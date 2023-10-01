# We will modify this file to create a lolrl trainer for llama on hh-rlhf dataset
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.optimization import AdamW, get_scheduler
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling,LlamaTokenizer

from trl import DPOTrainer, SFTTrainer
from tqdm import tqdm
import json
from metrics_hh import create_reward_fn

import logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
from time import time
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict, Counter
from utils import RANDOM_SEED, save_in_pickle, load_from_pickle, make_dir_if_not_exists, reduce_mean, save_in_jsonl, reduce_sum
import random

from rl_utils import ValueHeadMLP, ValueHeadAttention, numba_choice

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the SFT training script.
    """

    # data parameters
    # beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for SFT loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="huggyllama/llama-7b",metadata={"help": "the location of the SFT model name or path"})
    adapter_path: Optional[str] = field(default="timdettmers/qlora-hh-rlhf-7b", metadata={"help": "the location of the adapter model name or path"})
    
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    adam_beta1: float = field(default=0.9, metadata={"help": 'The beta1 parameter for AdamW'})
    adam_beta2: float = field(default=0.999, metadata={"help": 'The beta2 parameter for AdamW'})
    adam_epsilon: float = field(default=1e-6, metadata={"help": 'The epsilon parameter for AdamW'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_steps: int = field(default=300, metadata={"help": 'Number of warmup steps for learning rate schedule'})
    # warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})

    # ['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad']
    # Somebody suggested using adamw_bnb_8bit: https://gist.github.com/pacman100/1731b41f7a90a87b457e8c5415ff1c14?permalink_comment_id=4610607#gistcomment-4610607

    per_device_eval_batch_size: Optional[int] = field(default=16, metadata={"help": "eval batch size per device"})

    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    max_memory_MB: int = field(default=40000,metadata={"help": "Free memory per gpu."})

    max_prompt_length: Optional[int] = field(default=768 - 128, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=768, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=2500, metadata={"help": "the saving frequency"})
    # max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
    max_steps: Optional[int] = field(default=9000, metadata={"help": "max number of training steps"})
    eval_steps: Optional[int] = field(default=900, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    # Extra args from PRO setup
    block_size: Optional[int] = field(default=512, metadata={"help": "the max_length for the dataset"})
    train_file_path: Optional[str] = field(default="data/hh_train_len2", metadata={"help": "the path to the training data"})
    eval_file_path: Optional[str] = field(default="data/hh_dev/sampled_dev.json", metadata={"help": "the path to the evaluation data"})
    training_stage_num: Optional[int] = field(default=2, metadata={"help": "the number of training stages"})

    # Extra args for LoLRL
    # a2c_n_value_head_epochs: Optional[int] = field(default=5, metadata={"help": "the number of epochs to train the value head for"})
    a2c_n_value_head_epochs: Optional[int] = field(default=10, metadata={"help": "the number of epochs to train the value head for"})
    algorithm: Optional[str] = field(default="nll", metadata={"help": "the algorithm to use for training choices=['nll', 'wbc', 'r_gold', 'r_lol', 'a_lol', 'a_lol_ref_free', 'a_lol_seq', 'a_lol_kl']"})
    ppo_clip: Optional[float] = field(default=0.9, metadata={"help": "the clipping parameter for PPO"})
    kl_beta: Optional[float] = field(default=0.0, metadata={"help": "the beta parameter for KL penalty"})
    sampling_strategy: Optional[str] = field(default=None, metadata={"help": "the sampling strategy to use for advantage LoL RL methods"})
    cache_dir: Optional[str] = field(default="cache/", metadata={"help": "the cache directory"})
    seed: Optional[int] = field(default=RANDOM_SEED, metadata={"help": "the random seed"})

def evaluate_on_validation(args, model, tokenizer, get_score, reward_batch_size):
    model.eval()
    
    logging.info(f"Loading the evaluation dataset from {args.eval_file_path}")
    with open(args.eval_file_path, "r", encoding='utf-8') as f:
        infer_data = [json.loads(l) for l in f.readlines()]
    
    # Subsampling for debugging
    # infer_data = infer_data[:100]
    prefixes = list()
    fixed_prefixes = list()
    gen_suffixes = list()

    origin_state = (tokenizer.padding_side, tokenizer.truncation_side)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    prompt_prefix = (f"A chat between a curious human and an artificial intelligence assistant. "
                    f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(infer_data), args.per_device_eval_batch_size), desc=f"Generating responses with batch_size {args.per_device_eval_batch_size}"):
            current_batch_data = infer_data[i:i+args.per_device_eval_batch_size]
            current_batch_og_prefixes = [datum['prefix'][0] for datum in current_batch_data]
            current_batch_prefixes = ["".join(prefix) for prefix in current_batch_og_prefixes]
            # prefix_str.replace("<|prompter|>", "\n ### Human: ").replace("<|assistant|>", "\n ### Assistant: ").strip()
            current_batch_prefixes_fixed = [prompt_prefix + prefix_str.replace("<|prompter|>", " ### Human: ").replace("<|assistant|>", " ### Assistant: ").strip() for prefix_str in current_batch_prefixes]
            current_batch_prefixes_inputs = tokenizer(current_batch_prefixes_fixed, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.device)
            # breakpoint()
            # decoded = tokenizer.batch_decode(current_batch_prefixes_inputs["input_ids"], skip_special_tokens=True)
            # tokenizer.convert_ids_to_tokens([835])
            # human = tokenizer.encode("\nHuman:", add_special_tokens=False)
            # tokenizer.convert_ids_to_tokens(human)
            # (current_batch_prefixes_inputs["input_ids"] == 835)
            predicted_sents = model.generate(**current_batch_prefixes_inputs, max_new_tokens=128,pad_token_id=tokenizer.pad_token_id,num_beams=1,do_sample=False,num_return_sequences = 1, eos_token_id=[835, tokenizer.eos_token_id])
            repsonse_tokens = predicted_sents[:, current_batch_prefixes_inputs['input_ids'].shape[-1]:]
            responses = tokenizer.batch_decode(repsonse_tokens, skip_special_tokens=True)
            # Normalize responses
            # Only for the baseline checkpoint. Split at \nHuman:
            responses_normalized = [resp.split("\n Human:")[0].split("\nHuman:")[0].split("\n### Human")[0].strip() for resp in responses]
            responses_normalized = [resp.replace("###", "").strip() if resp.endswith("###") else resp.strip() for resp in responses_normalized]
            # check if any response_normalized ends with colon
            # if any([resp.endswith(":") for resp in responses_normalized]):
            #     breakpoint()
            # TODO: Check if tokenizer.eos_token is a part of any response
            prefixes.extend(current_batch_og_prefixes)
            fixed_prefixes.extend(current_batch_prefixes_fixed)
            gen_suffixes.extend(responses_normalized)
    all_val_rewards = list()
    torch.cuda.empty_cache()
    for i in tqdm(range(0, len(gen_suffixes), reward_batch_size), desc="Calculating rewards"):
        batch_suffixes = gen_suffixes[i:i+reward_batch_size]
        batch_prefixes = prefixes[i:i+reward_batch_size]
        batch_rewards = torch.sigmoid(get_score(batch_prefixes, batch_suffixes)).cpu().detach().numpy().tolist()
        all_val_rewards.extend(batch_rewards)
    avg_reward = np.mean(all_val_rewards)
    
    tokenizer.padding_side, tokenizer.truncation_side = origin_state
    # model_evaluator = DecoderUtils(self.model, args.world_size)
    # # Perform decoding and loss calculations here
    # _, wer = model_evaluator.evaluate(eval_dataloader, tokenizer)
    metrics = {'avg_val_reward': avg_reward}
    logging.info(f"Average reward on validation set of {len(gen_suffixes)} samples is {avg_reward:.4f}")
    all_val_rewards = np.array(all_val_rewards)
    model.train()
    return all_val_rewards, prefixes, fixed_prefixes, gen_suffixes


def train_value_function_on_val_predictions(value_function_model, model, tokenizer, prefixes, fixed_prefixes, suffixes, all_gen_rewards, args):
    # 1. Create a new optimizer for value head
    device = model.device
    value_fn_optimizer = AdamW([{'params': value_function_model.parameters()}], lr=1e-5)
    n_value_head_epochs = args.a2c_n_value_head_epochs
    logging.info(f"Estimating baseline policy value function for {n_value_head_epochs} epochs on dev set...")
    best_value_mse = float("inf")
    best_value_function_model = None
    best_epoch = None
    value_function_model.train()
    all_ids = list(range(len(prefixes)))
    id_to_gen_response_and_rewards = {id_: {"response": gen_resp, "rewards": gen_rewards} for id_, gen_resp, gen_rewards in zip(all_ids, suffixes, all_gen_rewards)}

    origin_state = (tokenizer.padding_side, tokenizer.truncation_side)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    model.eval()
    tqdm_flag = True
    for epoch in range(n_value_head_epochs):
        current_epoch_ids = deepcopy(all_ids)
        logging.info(f"Shuffling IDs for epoch {epoch+1}")
        random.shuffle(current_epoch_ids)
        batches = [current_epoch_ids[i:i+args.per_device_eval_batch_size] for i in range(0, len(current_epoch_ids), args.per_device_eval_batch_size)]
        pbar = tqdm(batches) if tqdm_flag else batches
        total_loss = 0.0
        total_steps = 0
        model_correct = 0.0
        total_instances = 0.0
        for batch in pbar:
            value_function_model.zero_grad()
            current_batch_prefixes = [fixed_prefixes[id_] for id_ in batch]
            current_batch_suffixes = [suffixes[id_] for id_ in batch]
            current_batch_rewards = [all_gen_rewards[id_] for id_ in batch]
            # Create prompt inputs for the model
            current_batch_prefixes_inputs = tokenizer(current_batch_prefixes, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.device)
            input_ids = current_batch_prefixes_inputs["input_ids"]
            attention_mask = current_batch_prefixes_inputs["attention_mask"]
            with torch.no_grad(): outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_layer_hidden_state = outputs.hidden_states[-1]
            val_outputs = value_function_model(last_layer_hidden_state)
            
            val_targets = torch.FloatTensor(current_batch_rewards).to(model.device)
            loss = F.mse_loss(val_outputs, val_targets)

            # 4. Backpropagate the loss
            loss.backward()
            value_fn_optimizer.step()
            # 5. Log the loss
            total_loss += loss.item()
            total_steps += 1
            avg_total_loss = total_loss / total_steps
            if tqdm_flag:
                pbar.set_postfix({"step_loss": loss.item(), "avg_total_loss": avg_total_loss})
        if avg_total_loss < best_value_mse:
            best_value_mse = avg_total_loss
            best_value_function_model = deepcopy(value_function_model)
            best_epoch = epoch+1
            logging.info(f"Found new best value head with MSE {best_value_mse} at epoch {best_epoch}")
        logging.info(f"Epoch {epoch+1} finished with MSE {avg_total_loss}")
    model.train()
    tokenizer.padding_side, tokenizer.truncation_side = origin_state
    # 6. Return the trained best value head
    return best_value_function_model, best_value_mse, best_epoch


def get_advantage_predictions_on_dataset(train_dataset, model, tokenizer, best_value_function_model, get_score, reward_batch_size, args):
    device = model.device
    with torch.no_grad():
        model.eval()
        all_good_advantages = list()
        all_bad_advantages = list()
        all_values = list()
        prefixes = list()
        fixed_prefixes = list()
        suffixes = list()
        total_value = 0.0
        total_good_reward = 0.0
        total_good_advantage = 0.0
        total_bad_reward = 0.0
        total_bad_advantage = 0.0

        origin_state = (tokenizer.padding_side, tokenizer.truncation_side)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        prompt_prefix = (f"A chat between a curious human and an artificial intelligence assistant. "
                        f"The assistant gives helpful, detailed, and polite answers to the user's questions.\n")
        pbar = tqdm(range(0, len(train_dataset), args.per_device_eval_batch_size), desc=f"Estimating train value and advantage with batch_size {args.per_device_eval_batch_size}")
        for i in pbar:
            current_batch_data = train_dataset[i:i+args.per_device_eval_batch_size]
            # current_batch_data = train_dataset[i:i+1]
            # current_batch_data.keys() = dict_keys(['prefix', 'suffix', 'reward', 'sft_index', 'meta'])
            current_batch_og_prefixes = [datum[0] for datum in current_batch_data['prefix']]
            current_batch_prefixes = ["".join(prefix) for prefix in current_batch_og_prefixes]
            # prefix_str.replace("<|prompter|>", "\n ### Human: ").replace("<|assistant|>", "\n ### Assistant: ").strip()
            current_batch_prefixes_fixed = [prompt_prefix + prefix_str.replace("<|prompter|>", " ### Human: ").replace("<|assistant|>", " ### Assistant: ").strip() for prefix_str in current_batch_prefixes]
            current_batch_prefixes_inputs = tokenizer(current_batch_prefixes_fixed, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.device)
            # forward the prefixes through model to get the last layer hidden state
            outputs = model(current_batch_prefixes_inputs['input_ids'], attention_mask=current_batch_prefixes_inputs['attention_mask'], output_hidden_states=True)
            last_layer_hidden_state = outputs.hidden_states[-1]
            # Calculate the value function estimates for current batch
            # Get the last layer encoder hidden state
            val_outputs = best_value_function_model(last_layer_hidden_state)
            current_batch_good_rewards = torch.sigmoid(torch.tensor([datum[0] for datum in current_batch_data['reward']]))
            current_batch_bad_rewards = torch.sigmoid(torch.tensor([datum[1] for datum in current_batch_data['reward']]))
            # Verify the rewards against the reward model.
            # current_batch_suffixes = [datum[0] for datum in current_batch_data['suffix']]
            # batch_rewards = torch.sigmoid(get_score(current_batch_og_prefixes, current_batch_suffixes)).cpu().detach().numpy().tolist()
            # current_batch_rewards.numpy() - np.array(batch_rewards)
            # NOTE: Sigmoid of given rewards is accurate w.r.t. the model
            
            # Calculate the advantage from rewards and value function estimates
            good_rewards = current_batch_good_rewards.to(model.device)
            bad_rewards = current_batch_bad_rewards.to(model.device)
            good_advantage = good_rewards - val_outputs
            bad_advantage = bad_rewards - val_outputs
            all_good_advantages.extend(good_advantage.tolist())
            all_bad_advantages.extend(bad_advantage.tolist())
            # Sometimes val_ouptuts can be a scalar float tensor
            # test_tensor = torch.FloatTensor(1.0)
            all_values.extend(val_outputs.tolist())

            # Update the total good and bad (reward, value and advantage) for logging
            total_value += val_outputs.sum().item()
            total_good_reward += good_rewards.sum().item()
            total_good_advantage += good_advantage.sum().item()
            total_bad_reward += bad_rewards.sum().item()
            total_bad_advantage += bad_advantage.sum().item()
            total_instances = i + len(current_batch_og_prefixes) 
            pbar.set_postfix({"#insts": total_instances, "avg val": total_value/total_instances, "avg good R": total_good_reward/total_instances, "avg good A": total_good_advantage/total_instances, "avg bad R": total_bad_reward/total_instances, "avg bad A": total_bad_advantage/total_instances})
        tokenizer.padding_side, tokenizer.truncation_side = origin_state
    model.train()
    return all_values, all_good_advantages, all_bad_advantages


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Official implementation of KL penalty in TRL PPOTrainer
# ref: https://github.com/huggingface/trl/blob/v0.7.1/trl/trainer/ppo_trainer.py#L1071C5-L1083C93
def _kl_penalty(config, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
    if config.kl_penalty == "kl":
        return logprob - ref_logprob

    if config.kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if config.kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if config.kl_penalty == "full":
        # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
        return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    torch.cuda.manual_seed_all(script_args.seed)
    logging.info(f"Set all seeds to {script_args.seed}")

    # Create the cache dir
    make_dir_if_not_exists(script_args.cache_dir)
    # logging.info(f"Loading model from {script_args.model_name_or_path}")
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    # 1. load a pretrained model
    # model = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,        # "meta-llama/Llama-2-7b-hf"
    #     quantization_config=bnb_config,
    #     device_map={"": 0},
    #     trust_remote_code=True,
    #     use_auth_token=True,
    # )
    # This code requires minimum of two gpus
    if torch.cuda.device_count() < 2:
        logging.error(f"Requires minimum of two gpus. Found {torch.cuda.device_count()}")
        exit(1)
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle0 = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    handle1 = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
    def print_gpu_info(handle, id):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        logging.info(f"GPU{id}: Total memory: {info.total} Free memory: {info.free} Used memory: {info.used}")
    # device = torch.device("cuda:0")
    # baseline_device = torch.device("cuda:1")
    start_time = time()
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    logging.info(f"Loading model from {script_args.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        load_in_4bit=True,
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        device_map="auto",
        # max_memory= {i: '40000MB' for i in range(torch.cuda.device_count())},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    logging.info(f"Loading adapter model from {script_args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, script_args.adapter_path, is_trainable=True)
    logging.info(f"Loaded the model in {time() - start_time} seconds")
    model.train()
    print_trainable_parameters(model)
    
    if script_args.algorithm in ["r_lol", "a_lol", "a_lol_seq"]:
        print_gpu_info(handle0, 0)
        print_gpu_info(handle1, 1)
        # Sharing backbone between training and behavior policy is not permitted yet: https://github.com/huggingface/peft/issues/854
        logging.info(f"Loading another base model from {script_args.model_name_or_path}")
        baseline_base_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            # max_memory= {i: '40000MB' for i in range(torch.cuda.device_count())},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                # bnb_4bit_compute_dtype=torch.bfloat16,
                # bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
        print_gpu_info(handle0, 0)
        print_gpu_info(handle1, 1)
        # Also initialize a behavior policy model
        logging.info(f"Initializing the behavior policy model from {script_args.adapter_path}")
        baseline_model = PeftModel.from_pretrained(baseline_base_model, script_args.adapter_path, is_trainable=False)
        baseline_model.eval()
        print_trainable_parameters(baseline_model)
        print_gpu_info(handle0, 0)
        print_gpu_info(handle1, 1)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False)
    logging.info(f"Loaded tokenizer from {script_args.model_name_or_path}")
    # Add pad token if missing
    if tokenizer._pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    # Load the reward model
    logging.info(f"Initializing the reward model.")
    start_time = time()
    get_score, reward_batch_size = create_reward_fn(reward_batch_size=script_args.per_device_eval_batch_size)
    logging.info(f"Initialized the reward model with batch size {reward_batch_size} in {time() - start_time} seconds")
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)

    # Load the train data
    logging.info(f"Loading train dataset from {script_args.train_file_path}")
    # train_dataset = load_dataset("json", data_dir = script_args.train_file_path, data_files = "train.json", streaming=False, split="train")
    train_dataset = load_dataset("json", data_dir = script_args.train_file_path, data_files = "cleaner_train.json", streaming=False, split="train")
    logging.info(f"Loaded train dataset with {len(train_dataset)} samples")
    all_good_suffixes = [datum[0] for datum in train_dataset['suffix']]
    all_good_suffixes_with_colon_end = [suffix for suffix in all_good_suffixes if suffix.endswith(":")]
    logging.info(f"Number of good suffixes that end with a colon: {len(all_good_suffixes_with_colon_end)}")
    all_good_suffixes_with_colon = [suffix for suffix in all_good_suffixes if ":" in suffix]
    logging.info(f"Number of good suffixes with colon: {len(all_good_suffixes_with_colon)}")
    all_good_suffixes_with_newline = [suffix for suffix in all_good_suffixes if "\n" in suffix]
    logging.info(f"Number of good suffixes with newline: {len(all_good_suffixes_with_newline)}")

    # Perform an initial validation
    eval_cache_file = os.path.join(script_args.cache_dir, "eval_cache.pkl")
    recompute_eval = False
    if os.path.exists(eval_cache_file) and not recompute_eval:
        logging.info(f"Loading the validation results from {eval_cache_file}")
        all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes = load_from_pickle(eval_cache_file)
        logging.info(f"Average reward on validation set of {len(gen_suffixes)} samples is {np.mean(all_val_gen_rewards):.4f}")
    else:
        logging.info(f"Evaluating on validation set for the first time.")
        all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes = evaluate_on_validation(script_args, model, tokenizer, get_score, reward_batch_size)
        save_in_pickle([all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes], eval_cache_file)
        logging.info(f"Saved the validation results in {eval_cache_file}")
    best_initial_avg_reward = np.mean(all_val_gen_rewards)
    logging.info(f"Best initial average reward on validation set of {len(gen_suffixes)} samples is {best_initial_avg_reward:.4f}")
    
    # Create an empty eval trajectory file
    make_dir_if_not_exists(script_args.output_dir)
    eval_trajectory_file = os.path.join(script_args.output_dir, "eval_trajectory.jsonl")
    open(eval_trajectory_file, "w").close()
    save_in_jsonl([{"step":0, "avg_reward": best_initial_avg_reward}], eval_trajectory_file, append=True)

    # TEMP: For debugging
    # all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes = evaluate_on_validation(script_args, model, tokenizer, get_score, reward_batch_size)
    if script_args.algorithm in ["a_lol", "a_lol_seq", "a_lol_ref_free"]:
        # Train value estimate
        val_model_save_path = os.path.join(script_args.cache_dir, "best_value_function_model.pt")
        value_function_estimator = ValueHeadAttention(model.config, max_value=1.0).to(model.device)
        recompute_val = False
        if os.path.exists(val_model_save_path) and not recompute_val:
            logging.info(f"Loading the best value function model from {val_model_save_path}")
            value_function_estimator.load_state_dict(torch.load(val_model_save_path))
            best_value_function_model = value_function_estimator
            logging.info(f"Loaded the best value function model from {val_model_save_path}")
        else:
            logging.info(f"Training the value function model on validation predictions for the first time.")
            best_value_function_model, best_value_mse, best_epoch = train_value_function_on_val_predictions(value_function_estimator, model, tokenizer, prefixes, fixed_prefixes, gen_suffixes, all_val_gen_rewards, script_args)
            # Save the best_value_function_model
            torch.save(best_value_function_model.state_dict(), val_model_save_path)
            logging.info(f"Saved the best value function model in {val_model_save_path}")
        

        
        # Get value and advantage estimates on train dataset
        train_advantage_cache_file = os.path.join(script_args.cache_dir, "train_advantage_cache.pkl")
        recompute_adv = False
        if os.path.exists(train_advantage_cache_file) and not recompute_adv:
            logging.info(f"Loading the train value, good and bad advantage estimates from {train_advantage_cache_file}")
            all_values, all_good_advantages, all_bad_advantages = load_from_pickle(train_advantage_cache_file)
            logging.info(f"Loaded {len(all_values)} the train value, good and bad advantage estimates from {train_advantage_cache_file}")
        else:
            all_values, all_good_advantages, all_bad_advantages = get_advantage_predictions_on_dataset(train_dataset, model, tokenizer, best_value_function_model, get_score, reward_batch_size, script_args)
            save_in_pickle([all_values, all_good_advantages, all_bad_advantages], train_advantage_cache_file)
            logging.info(f"Saved {len(all_values)} the train value, good and bad advantage estimates in {train_advantage_cache_file}")
        
        if script_args.algorithm in ["a_lol_seq"]:
            best_value_function_model = best_value_function_model.to(baseline_model.device)
        else:
            best_value_function_model = best_value_function_model.to(model.device)
        all_values = np.array(all_values)
        all_good_advantages = np.array(all_good_advantages)
        all_bad_advantages = np.array(all_bad_advantages)

    all_good_rewards = torch.sigmoid(torch.tensor([datum[0] for datum in train_dataset['reward']])).cpu().detach().numpy()
    all_bad_rewards = torch.sigmoid(torch.tensor([datum[1] for datum in train_dataset['reward']])).cpu().detach().numpy()
    # Train the model with a_lol


    # 2. Start custom training loop
    if script_args.algorithm == "nll":
        logging.info(f"Shuffling the train indices.")
        train_indices = list(range(len(train_dataset)))
        current_train_indices = deepcopy(train_indices)
        random.shuffle(current_train_indices)
        sampler = iter(current_train_indices)
    elif script_args.algorithm in ["wbc", "r_gold", "r_lol"]:
        logging.info(f"Percentiles for all good rewards: {np.percentile(all_good_rewards, [0, 25, 50, 75, 100])}")
        logging.info(f"Percentiles for all bad rewards: {np.percentile(all_bad_rewards, [0, 25, 50, 75, 100])}")
        if script_args.sampling_strategy == "good_priority":
            # Use the good rewards for sampling
            positive_ratio = 1.0
            logging.info(f"Using sampling strategy: {script_args.sampling_strategy} with positive_ratio = {positive_ratio}")
            sample_weights = all_good_rewards
    elif script_args.algorithm in ["a_lol", "a_lol_seq", "a_lol_ref_free"]:
        logging.info(f"Percentiles for all good advantages: {np.percentile(all_good_advantages, [0, 25, 50, 75, 100])}")
        positive_good_ratio = np.sum(all_good_advantages > 0) / len(all_good_advantages)
        logging.info(f"Percentage of positive advantages: {positive_good_ratio*100.0:.2f}%")
        logging.info(f"Percentiles for all bad advantages: {np.percentile(all_bad_advantages, [0, 25, 50, 75, 100])}")
        positive_bad_ratio = np.sum(all_bad_advantages > 0) / len(all_bad_advantages)
        logging.info(f"Percentage of positive bad advantages: {positive_bad_ratio*100.0:.2f}%")
        if script_args.sampling_strategy in ["good_priority", "good_random"]:
            positive_ratio = positive_good_ratio
            logging.info(f"Using sampling strategy: {script_args.sampling_strategy} with positive_ratio = {positive_ratio}")
        elif script_args.sampling_strategy == "all_priority":
            positive_ratio = positive_good_ratio + positive_bad_ratio
            logging.info(f"Using sampling strategy: {script_args.sampling_strategy} with positive_ratio = {positive_ratio} = {positive_good_ratio} + {positive_bad_ratio}")
        elif script_args.sampling_strategy is None:
            logging.info(f"Shuffling the train indices.")
            train_indices = list(range(len(train_dataset)))
            current_train_indices = deepcopy(train_indices)
            random.shuffle(current_train_indices)
            sampler = iter(current_train_indices)
            positive_ratio = 1.0
            logging.info(f"Using sampling strategy: {script_args.sampling_strategy} with positive_ratio = {positive_ratio}. Will be sampling on the the good responses")
            # Report avg advantage for good and bad
            logging.info(f"Average good advantage: {np.mean(all_good_advantages):.4f}")
            # logging.info(f"Average bad advantage: {np.mean(all_bad_advantages):.4f}")
        # Update the script_args.max_steps and script_args.eval_steps by positive_good_ratio
        logging.info(f"Previous max_steps = {script_args.max_steps} and eval_steps = {script_args.eval_steps}")
        script_args.max_steps = int(script_args.max_steps * positive_ratio)
        script_args.eval_steps = int(script_args.eval_steps * positive_ratio)
        logging.info(f"Updated max_steps = {script_args.max_steps} and eval_steps = {script_args.eval_steps}")
        
        # Use different sample weights for different sampling strategies
        if script_args.sampling_strategy in ["good_priority", "good_random"]:
            positive_all_good_advantages = np.array(all_good_advantages)
            positive_all_good_advantages[positive_all_good_advantages < 0] = 0.0
            # Print the percentiles for positive advantages
            logging.info(f"Percentiles for good positive advantages: {np.percentile(positive_all_good_advantages, [0, 25, 50, 75, 100])}")
            if script_args.sampling_strategy == "good_priority":
                sample_weights = positive_all_good_advantages
            else:
                # Give uniform weights to all positive advantages
                sample_weights = np.array(positive_all_good_advantages)
                sample_weights[sample_weights > 0] = 1.0
        elif script_args.sampling_strategy == "all_priority":
            positive_all_good_advantages = np.array(all_good_advantages)
            positive_all_good_advantages[positive_all_good_advantages < 0] = 0.0
            positive_all_bad_advantages = np.array(all_bad_advantages)
            positive_all_bad_advantages[positive_all_bad_advantages < 0] = 0.0
            # Print the percentiles for positive advantages
            logging.info(f"Percentiles for good positive advantages: {np.percentile(positive_all_good_advantages, [0, 25, 50, 75, 100])}")
            logging.info(f"Percentiles for bad positive advantages: {np.percentile(positive_all_bad_advantages, [0, 25, 50, 75, 100])}")
            # Concatenate the positive_all_good_advantages and positive_all_bad_advantages
            sample_weights = np.concatenate([positive_all_good_advantages, positive_all_bad_advantages])

    if script_args.sampling_strategy is not None:
        # Normalize the sample weights
        sample_probs = sample_weights / np.sum(sample_weights)
        logging.info(f"Total instances = {len(sample_probs)}")
        # Keep track of indices and non-zero probs
        sample_indices = np.where(sample_probs != 0.0)[0]
        logging.info(f"Total instances with non-zero probs = {len(sample_indices)}")
        # Also update the sample_probs accordingly
        sample_probs = sample_probs[sample_indices]
        logging.info(f"Total instances with non-zero probs after filtering = {len(sample_probs)}")
        sample_probs_csum = np.cumsum(sample_probs)
        logging.info(f"Total instances with non-zero probs after cumsum = {len(sample_probs_csum)}")
    pbar = tqdm(range(script_args.max_steps * script_args.gradient_accumulation_steps), desc=f"Training the model with {script_args.algorithm}")
    
    # 1. Create a new optimizer for peftmodel
    from transformers import Trainer
    script_args.optim_args = {}
    # script_args.adam_beta1 = 0.9
    # script_args.adam_beta2 = 0.999
    # script_args.adam_epsilon = 1e-6
    optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(script_args)
    # TODO: Check later if this is the right optimizer
    optimizer = optim_cls(model.parameters(), **optim_kwargs)
    script_args.warmup_steps = 300
    scheduler = get_scheduler(script_args.lr_scheduler_type, optimizer, num_warmup_steps=script_args.warmup_steps, num_training_steps=script_args.max_steps * script_args.gradient_accumulation_steps)
    
    prompt_prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    origin_state = (tokenizer.padding_side, tokenizer.truncation_side)
    total_loss = 0.0
    total_per_resp_loss_with_IS = 0.0
    total_per_resp_loss_without_IS = 0.0
    total_per_resp_loss_before_reward = 0.0
    total_kl_penalty = 0.0
    total_alol_loss = 0.0
    total_rlol_loss = 0.0
    total_advantage = 0.0
    total_reward = 0.0
    total_suffix_distribution = {0:0, 1:0}
    logging.info(f"Evaluating every {script_args.eval_steps * script_args.gradient_accumulation_steps} steps.")
    logging.info(f"Using algorithm: {script_args.algorithm}")
    if script_args.algorithm in ["a_lol", "a_lol_seq", "a_lol_ref_free"]:
        logging.info(f"Using PPO CLIP: {script_args.ppo_clip}")
        logging.info(f"Using KL beta: {script_args.kl_beta}")
    best_model = None
    for step in pbar:
        # 1. Get the next batch of data from sampler
        if script_args.algorithm == "nll":
            try:
                batch_indices = [next(sampler) for _ in range(script_args.per_device_train_batch_size)]
            except StopIteration:
                logging.info(f"Shuffling the train indices.")
                current_train_indices = deepcopy(train_indices)
                random.shuffle(current_train_indices)
                sampler = iter(current_train_indices)
                batch_indices = [next(sampler) for _ in range(script_args.per_device_train_batch_size)]
            batch_suffix_indices = [0] * len(batch_indices)
        elif script_args.algorithm in ["wbc", "r_gold", "r_lol"]:
            batch_indices = numba_choice(sample_indices, sample_probs_csum, 4).tolist()
            if script_args.sampling_strategy == "good_priority":
                batch_rewards = all_good_rewards[batch_indices]
                # batch_values = all_values[batch_indices]
                # batch_advantages = all_good_advantages[batch_indices]
                batch_suffix_indices = [0] * len(batch_indices)
            else:
                logging.warning(f"Not implemented for sampling strategy {script_args.sampling_strategy}")
                breakpoint()
        elif script_args.algorithm in ["a_lol", "a_lol_seq", "a_lol_ref_free"]:
            if script_args.sampling_strategy is None:
                # Use the same sampling as NLL
                try:
                    batch_indices = [next(sampler) for _ in range(script_args.per_device_train_batch_size)]
                except StopIteration:
                    logging.info(f"Shuffling the train indices.")
                    current_train_indices = deepcopy(train_indices)
                    random.shuffle(current_train_indices)
                    sampler = iter(current_train_indices)
                    batch_indices = [next(sampler) for _ in range(script_args.per_device_train_batch_size)]
                # Select good suffix indices
                batch_suffix_indices = [0] * len(batch_indices)
                # Get the batch advantages based on the suffix indices
                batch_advantages = np.array([all_good_advantages[idx] for i, idx in enumerate(batch_indices)])
                batch_values = all_values[batch_indices]
                batch_rewards = np.array([all_good_rewards[idx] for i, idx in enumerate(batch_indices)])
            elif script_args.sampling_strategy.endswith("_priority") or script_args.sampling_strategy == "good_random":
                batch_indices = numba_choice(sample_indices, sample_probs_csum, 4).tolist()
                if script_args.sampling_strategy in ["good_priority", "good_random"]:
                    batch_advantages = all_good_advantages[batch_indices]
                    batch_values = all_values[batch_indices]
                    batch_suffix_indices = [0] * len(batch_indices)
                    batch_rewards = all_good_rewards[batch_indices]
                elif script_args.sampling_strategy == "all_priority":
                    # Some batch indices will be > len(all_good_advantages). Those correspond to bad advantages
                    batch_suffix_indices = [0 if idx < len(all_good_advantages) else 1 for idx in batch_indices]
                    batch_advantages = np.array([all_good_advantages[idx] if idx < len(all_good_advantages) else all_bad_advantages[idx - len(all_good_advantages)] for idx in batch_indices])
                    new_batch_indices = [idx if idx < len(all_good_advantages) else idx - len(all_good_advantages) for idx in batch_indices]
                    batch_values = all_values[new_batch_indices]
                    batch_indices = new_batch_indices
                    batch_rewards = np.array([all_good_rewards[idx] if idx < len(all_good_advantages) else all_bad_rewards[idx - len(all_good_advantages)] for idx in batch_indices])
                assert np.all(batch_advantages >= 0.0), breakpoint()
        batch = [train_dataset[i] for i in batch_indices]
        # 2. Preprocess the batch data for model
        # 2.1. Get the batch prefix and suffix
        batch_og_prefixes = [datum['prefix'][0] for datum in batch]
        batch_prefixes = ["".join(prefix) for prefix in batch_og_prefixes]
        batch_prefixes_fixed = [prompt_prefix + prefix_str.replace("<|prompter|>", " ### Human: ").replace("<|assistant|>", " ### Assistant: ").strip() for prefix_str in batch_prefixes]
        batch_suffixes = [datum['suffix'][batch_suffix_indices[i]] for i, datum in enumerate(batch)]

        # batch_good_suffixes = [datum['suffix'][0] for datum in batch]
        # batch_bad_suffixes = [datum['suffix'][1] for datum in batch]
        batch_good_rewards = torch.sigmoid(torch.tensor([datum['reward'][0] for datum in batch]))
        # 2.2 Get left truncated prefix input
        tokenizer.truncation_side, tokenizer.padding_side = "left", "left"
        current_batch_prefixes_inputs = tokenizer(batch_prefixes_fixed, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.device)
        # 2.3 Get right truncated suffix input
        # NOTE: Adding tokenizer_eos token to the end of the suffixes
        batch_suffixes = [suffix + tokenizer.eos_token for suffix in batch_suffixes]
        tokenizer.truncation_side, tokenizer.padding_side = "right", "right"
        current_batch_suffixes_inputs = tokenizer(batch_suffixes, max_length = 128,truncation = True,add_special_tokens=False, padding = True, return_tensors="pt").to(model.device)
        # 2.4 Merge the prefix and suffix inputs
        input_ids = torch.cat([current_batch_prefixes_inputs["input_ids"], current_batch_suffixes_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([current_batch_prefixes_inputs["attention_mask"], current_batch_suffixes_inputs["attention_mask"]], dim=1)
        # tokenizer.convert_ids_to_tokens(input_ids[0])
        # tokenizer.convert_ids_to_tokens(current_batch_suffixes_inputs["input_ids"][0])
        # 3. Forward pass through model
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_size, query_seq_len = current_batch_prefixes_inputs["input_ids"].shape
        resp_logits = logits[:, (query_seq_len - 1):-1, :]
        # Gather per label logits
        labels = current_batch_suffixes_inputs["input_ids"]
        response_ids = current_batch_suffixes_inputs["input_ids"]
        response_mask = current_batch_suffixes_inputs["attention_mask"]
        # Logits are before softmax. So, we need to apply softmax to get the probabilities
        # logits is of the shape [batch_size, seq_len, vocab_size]
        # labels is of the shape [batch_size, seq_len]
        resp_log_probs = F.log_softmax(resp_logits, dim=-1)
        per_token_log_probs = -torch.gather(resp_log_probs, 2, labels[:, :, None]).squeeze(2)
        postfix_dict = dict()
        # Update the suffix distribution
        for suffix_idx in batch_suffix_indices:
            total_suffix_distribution[suffix_idx] += 1
        # Update the postfix_dict
        postfix_dict["suffix_dist"] = total_suffix_distribution
        
        if script_args.algorithm == "nll":
            # per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1) / torch.sum(response_mask, dim=1)
            # NOTE: Consistently sum is better than mean
            per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1)
            loss = torch.mean(per_response_loss)
        elif script_args.algorithm in ["wbc"]:
            # per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1) / torch.sum(response_mask, dim=1)
            # NOTE: Consistently sum is better than mean
            per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1)
            total_per_resp_loss_before_reward += np.mean(per_response_loss.cpu().detach().numpy())
            postfix_dict["avg_loss_/wo_reward"] = total_per_resp_loss_before_reward / (step+1)
            total_reward += np.mean(batch_rewards)
            postfix_dict["avg_reward"] = total_reward / (step+1)
            loss = torch.mean(per_response_loss * torch.FloatTensor(batch_rewards).to(model.device))
            # total_advantage += np.mean(batch_advantages)
            # postfix_dict["debug avg_adv"] = total_advantage / (step+1)
        elif script_args.algorithm in ["r_gold"]:
            # pi * delta log pi * R
            # They also lowerbound pi as max(u, pi) where u is 0.1, 0.15 or .2 for various tasks
            with torch.no_grad():
                per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1)
                total_per_resp_loss_before_reward += np.mean(per_response_loss.cpu().detach().numpy())
                postfix_dict["avg_loss_/wo_reward"] = total_per_resp_loss_before_reward / (step+1)
                per_token_probs = torch.exp(-per_token_log_probs.detach())
                # Lowerbound the importance weight by applying a max on each token to 0.1
                lower_bound_per_token_probs = torch.max(per_token_probs, torch.tensor(0.1).to(model.device))
            # Multiply the loss with the probs
            lm_loss = torch.mul(per_token_log_probs, lower_bound_per_token_probs)
            per_response_loss_gold_objective = torch.sum(lm_loss * response_mask, dim=1)
            # Multiply the loss with the rewards
            total_reward += np.mean(batch_rewards)
            postfix_dict["avg_reward"] = total_reward / (step+1)
            loss = torch.mean(per_response_loss_gold_objective * torch.FloatTensor(batch_rewards).to(model.device))
        elif script_args.algorithm in ["r_lol"]:
            # Reward * Importance weight * log pi
            with torch.no_grad():
                # Compute the baseline model log probs
                total_per_resp_loss_without_IS += np.mean(torch.sum(per_token_log_probs * response_mask, dim=1).cpu().detach().numpy())
                postfix_dict["avg_loss_/wo_IS_or_KL"] = total_per_resp_loss_without_IS / (step+1)
                baseline_input_ids = deepcopy(input_ids).to(baseline_model.device)
                baseline_attention_mask = deepcopy(attention_mask).to(baseline_model.device)
                baseline_outputs = baseline_model(baseline_input_ids, attention_mask=baseline_attention_mask)
                batch_size, query_seq_len = current_batch_prefixes_inputs["input_ids"].shape
                baseline_resp_logits = baseline_outputs.logits[:, (query_seq_len - 1):-1, :]
                baseline_log_probs = F.log_softmax(baseline_resp_logits, dim=-1)
                baseline_labels = deepcopy(labels).to(baseline_model.device)
                baseline_per_token_log_probs = -torch.gather(baseline_log_probs, 2, baseline_labels[:, :, None]).squeeze(2)
                baseline_response_mask = deepcopy(response_mask).to(baseline_model.device)

                baseline_full_seq_log_probs = torch.sum(baseline_per_token_log_probs * baseline_response_mask, dim=1)
                # Compute the seq importance sampling ratio over each word
                # Both per_token_log_probs and baseline_per_token_log_probs are negative log probs
                copy_per_token_log_probs = deepcopy(per_token_log_probs.detach()).to(baseline_model.device)
                copy_full_seq_log_probs = torch.sum(copy_per_token_log_probs * baseline_response_mask, dim=1)
                importance_sampling_ratio = torch.exp(baseline_full_seq_log_probs - copy_full_seq_log_probs.detach())
                if script_args.ppo_clip is not None and script_args.ppo_clip > 0.0:
                    # Clamp the importance sampling ratio
                    importance_sampling_ratio_clamped = torch.clamp(importance_sampling_ratio, 1 - script_args.ppo_clip, 1 + script_args.ppo_clip)
                    # return_dict['importance_sampling_ratio_clamped'] = importance_sampling_ratio_clamped
                    importance_sampling_ratio = importance_sampling_ratio_clamped
            total_reward += np.mean(batch_rewards)
            postfix_dict["avg_reward"] = total_reward / (step+1)
            if script_args.kl_beta > 0.0:
                assert script_args.ppo_clip == 0.0, breakpoint()
                # per_token_log_probs is negative log probs
                # baseline_full_seq_log_probs is negative log probs
                per_seq_neg_log_probs = torch.sum(per_token_log_probs * response_mask, dim=1)
                kl_penalty = script_args.kl_beta * (baseline_full_seq_log_probs - per_seq_neg_log_probs).mean()
                total_kl_penalty += kl_penalty.cpu().detach().item()
                postfix_dict["avg_kl_penalty"] = total_kl_penalty / (step+1)
                # Multiply the per_token_log_probs with the reward
                rlol_loss = torch.mean(per_seq_neg_log_probs * torch.FloatTensor(batch_rewards).to(model.device))
                total_rlol_loss += rlol_loss.cpu().detach().item()
                postfix_dict["avg_rlol_loss"] = total_rlol_loss / (step+1)
                loss = rlol_loss + kl_penalty
            else:
                # Multiply importance sampling with log probs
                importance_sampling_ratio = importance_sampling_ratio.to(model.device)
                per_response_loss_with_importance_sampling = torch.sum(per_token_log_probs * response_mask, dim=1) * importance_sampling_ratio
                total_per_resp_loss_with_IS += np.mean(per_response_loss_with_importance_sampling.cpu().detach().numpy())
                postfix_dict["avg_loss_/w_IS"] = total_per_resp_loss_with_IS / (step+1)
                # Get the rewards from extra args
                # extra_data is a dict with dict_keys(['texts', 'responses', 'batch', 'rewards'])
                # Convert list of rewards from extra_data['rewards'] to tensor
                # rewards = torch.tensor(batch_rewards).to(model.device)

                # Multiply the loss with the reward
                loss = torch.mean(per_response_loss_with_importance_sampling * torch.FloatTensor(batch_rewards).to(model.device))
        elif script_args.algorithm in ["a_lol"]:
            # Calculate baseline policy value
            with torch.no_grad():
                # Compute the baseline model log probs
                total_per_resp_loss_without_IS += np.mean(torch.sum(per_token_log_probs * response_mask, dim=1).cpu().detach().numpy())
                postfix_dict["avg_loss_/wo_IS_or_KL"] = total_per_resp_loss_without_IS / (step+1)
                baseline_input_ids = deepcopy(input_ids).to(baseline_model.device)
                baseline_attention_mask = deepcopy(attention_mask).to(baseline_model.device)
                baseline_outputs = baseline_model(baseline_input_ids, attention_mask=baseline_attention_mask, output_hidden_states=True)
                batch_size, query_seq_len = current_batch_prefixes_inputs["input_ids"].shape
                baseline_resp_logits = baseline_outputs.logits[:, (query_seq_len - 1):-1, :]
                last_layer_hidden_state = baseline_outputs.hidden_states[-1][:, :query_seq_len, :]
                val_outputs = best_value_function_model(last_layer_hidden_state).detach().cpu().numpy()
                # baseline_outputs2 = baseline_model(current_batch_prefixes_inputs["input_ids"], attention_mask=current_batch_prefixes_inputs["attention_mask"], output_hidden_states=True)
                # last_layer_hidden_state2 = baseline_outputs2.hidden_states[-1]
                # val_outputs2 = best_value_function_model(last_layer_hidden_state2)
                baseline_log_probs = F.log_softmax(baseline_resp_logits, dim=-1)
                baseline_labels = deepcopy(labels).to(baseline_model.device)
                baseline_per_token_log_probs = -torch.gather(baseline_log_probs, 2, baseline_labels[:, :, None]).squeeze(2)
                baseline_response_mask = deepcopy(response_mask).to(baseline_model.device)

                baseline_full_seq_log_probs = torch.sum(baseline_per_token_log_probs * baseline_response_mask, dim=1)
                # Compute the seq importance sampling ratio over each word
                # Both per_token_log_probs and baseline_per_token_log_probs are negative log probs
                copy_per_token_log_probs = deepcopy(per_token_log_probs.detach()).to(baseline_model.device)
                copy_full_seq_log_probs = torch.sum(copy_per_token_log_probs * baseline_response_mask, dim=1)
                importance_sampling_ratio = torch.exp(baseline_full_seq_log_probs - copy_full_seq_log_probs.detach())
                if script_args.ppo_clip is not None and script_args.ppo_clip > 0.0:
                    # Clamp the importance sampling ratio
                    importance_sampling_ratio_clamped = torch.clamp(importance_sampling_ratio, 1 - script_args.ppo_clip, 1 + script_args.ppo_clip)
                    # return_dict['importance_sampling_ratio_clamped'] = importance_sampling_ratio_clamped
                    importance_sampling_ratio = importance_sampling_ratio_clamped
            advantage = torch.tensor(batch_rewards - val_outputs).to(model.device)
            advantage = torch.clamp(advantage, min=0.0)
            total_advantage += torch.mean(advantage).cpu().detach().item()
            postfix_dict["avg_advantage"] = total_advantage / (step+1)
            if script_args.kl_beta > 0.0:
                assert script_args.ppo_clip == 0.0, breakpoint()
                # per_token_log_probs is negative log probs
                # baseline_full_seq_log_probs is negative log probs
                per_seq_neg_log_probs = torch.sum(per_token_log_probs * response_mask, dim=1)
                kl_penalty = script_args.kl_beta * (baseline_full_seq_log_probs - per_seq_neg_log_probs).mean()
                total_kl_penalty += kl_penalty.cpu().detach().item()
                postfix_dict["avg_kl_penalty"] = total_kl_penalty / (step+1)
                # Multiply the per_token_log_probs with the advantage
                alol_loss = torch.mean(per_seq_neg_log_probs * advantage)
                total_alol_loss += alol_loss.cpu().detach().item()
                postfix_dict["avg_alol_loss"] = total_alol_loss / (step+1)
                loss = alol_loss + kl_penalty
            else:
                # Multiply importance sampling with log probs
                importance_sampling_ratio = importance_sampling_ratio.to(model.device)
                per_response_loss_with_importance_sampling = torch.sum(per_token_log_probs * response_mask, dim=1) * importance_sampling_ratio
                total_per_resp_loss_with_IS += np.mean(per_response_loss_with_importance_sampling.cpu().detach().numpy())
                postfix_dict["avg_loss_/w_IS"] = total_per_resp_loss_with_IS / (step+1)
                # Get the rewards from extra args
                # extra_data is a dict with dict_keys(['texts', 'responses', 'batch', 'rewards'])
                # Convert list of rewards from extra_data['rewards'] to tensor
                # rewards = torch.tensor(batch_rewards).to(model.device)

                # Multiply the loss with the advantage
                loss = torch.mean(per_response_loss_with_importance_sampling * advantage)
        elif script_args.algorithm in ["a_lol_seq"]:
            # Calculate baseline policy value
            with torch.no_grad():
                total_per_resp_loss_without_IS += np.mean(torch.sum(per_token_log_probs * response_mask, dim=1).cpu().detach().numpy())
                postfix_dict["avg_loss_/wo_IS_or_KL"] = total_per_resp_loss_without_IS / (step+1)
                # Compute the baseline model log probs
                baseline_input_ids = deepcopy(input_ids).to(baseline_model.device)
                baseline_attention_mask = deepcopy(attention_mask).to(baseline_model.device)
                baseline_outputs = baseline_model(baseline_input_ids, attention_mask=baseline_attention_mask, output_hidden_states=True)
                batch_size, query_seq_len = current_batch_prefixes_inputs["input_ids"].shape
                baseline_resp_logits = baseline_outputs.logits[:, (query_seq_len - 1):-1, :]
                last_layer_hidden_state = baseline_outputs.hidden_states[-1][:, :query_seq_len, :]
                val_outputs = best_value_function_model(last_layer_hidden_state).detach().cpu().numpy()
                # baseline_outputs2 = baseline_model(current_batch_prefixes_inputs["input_ids"], attention_mask=current_batch_prefixes_inputs["attention_mask"], output_hidden_states=True)
                # last_layer_hidden_state2 = baseline_outputs2.hidden_states[-1]
                # val_outputs2 = best_value_function_model(last_layer_hidden_state2)
                baseline_log_probs = F.log_softmax(baseline_resp_logits, dim=-1)
                baseline_labels = deepcopy(labels).to(baseline_model.device)
                baseline_per_token_log_probs = -torch.gather(baseline_log_probs, 2, baseline_labels[:, :, None]).squeeze(2)
                # Compute the seq importance sampling ratio over each word
                # Both per_token_log_probs and baseline_per_token_log_probs are negative log probs
                copy_per_token_log_probs = deepcopy(per_token_log_probs.detach()).to(baseline_model.device)
                baseline_response_mask = deepcopy(response_mask).to(baseline_model.device)
                importance_sampling_ratio_per_token = torch.exp(baseline_per_token_log_probs - copy_per_token_log_probs.detach()) * baseline_response_mask
                if script_args.ppo_clip is not None:
                    # Clamp the importance sampling ratio
                    importance_sampling_ratio_per_token_clamped = torch.clamp(importance_sampling_ratio_per_token, 1 - script_args.ppo_clip, 1 + script_args.ppo_clip) * baseline_response_mask
                    # return_dict['importance_sampling_ratio_per_token_clamped'] = importance_sampling_ratio_per_token_clamped
                    importance_sampling_ratio_per_token = importance_sampling_ratio_per_token_clamped
            # Multiply importance sampling with log probs
            importance_sampling_ratio_per_token = importance_sampling_ratio_per_token.to(model.device)
            lm_loss = torch.mul(per_token_log_probs, importance_sampling_ratio_per_token)
            # per_response_loss_with_importance_sampling = reduce_mean(lm_loss, response_mask, axis=1)
            per_response_loss_with_importance_sampling = reduce_sum(lm_loss, response_mask, axis=1)
            total_per_resp_loss_with_IS += np.mean(per_response_loss_with_importance_sampling.cpu().detach().numpy())
            postfix_dict["avg_loss_/w_IS"] = total_per_resp_loss_with_IS / (step+1)
            # Get the rewards from extra args
            # extra_data is a dict with dict_keys(['texts', 'responses', 'batch', 'rewards'])
            # Convert list of rewards from extra_data['rewards'] to tensor
            # rewards = torch.tensor(batch_rewards).to(model.device)

            # Multiply the loss with the advantage
            advantage = torch.tensor(batch_rewards - val_outputs).to(model.device)
            advantage = torch.clamp(advantage, min=0.0)
            total_advantage += torch.mean(advantage).cpu().detach().item()
            postfix_dict["avg_advantage"] = total_advantage / (step+1)
            loss = torch.mean(per_response_loss_with_importance_sampling * advantage)
        elif script_args.algorithm == "a_lol_ref_free":
            # per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1) / torch.sum(response_mask, dim=1)
            # NOTE: experimenting with sum instead of mean because PRO and DPO use sum
            per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1)
            # Compute the baseline model log probs
            total_per_resp_loss_without_IS += np.mean(per_response_loss.detach().cpu().numpy())
            postfix_dict["avg_loss_/wo_adv"] = total_per_resp_loss_without_IS / (step+1)
            total_advantage += np.mean(batch_advantages)
            postfix_dict["avg_advantage"] = total_advantage / (step+1)
            loss = torch.mean(per_response_loss * torch.FloatTensor(batch_advantages).to(model.device))
        # 4. Backpropagate the loss and update the model parameters if gradient accumulation is done
        loss.backward()
        if (step+1) % script_args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), script_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # 5. Log the loss
        total_loss += loss.item()
        postfix_dict["step_loss"] = loss.item()
        # logging.info(f"Before clear cache")
        # print_gpu_info(handle0, 0)
        # print_gpu_info(handle1, 1)
        del loss, logits
        if script_args.algorithm == "a_lol_seq":
            del per_response_loss_with_importance_sampling, input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
        # logging.info(f"After clear cache")
        # print_gpu_info(handle0, 0)
        # print_gpu_info(handle1, 1)
        avg_total_loss = total_loss / (step+1)
        postfix_dict["avg_total_loss"] = avg_total_loss
        pbar.set_postfix(postfix_dict)
        if (step+1) % (script_args.eval_steps * script_args.gradient_accumulation_steps) == 0:
            torch.cuda.empty_cache()
            all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes = evaluate_on_validation(script_args, model, tokenizer, get_score, reward_batch_size)
            current_avg_reward = np.mean(all_val_gen_rewards)
            save_in_jsonl([{"step":(step+1), "avg_reward": current_avg_reward}], eval_trajectory_file, append=True)
            torch.cuda.empty_cache()
            if current_avg_reward > best_initial_avg_reward:
                logging.info(f"Achieved new best average reward of {current_avg_reward:.4f} on validation set. Compared to previous best of {best_initial_avg_reward:.4f}")
                best_initial_avg_reward = current_avg_reward
                # Save the peft model
                logging.info(f"Saving model at {script_args.output_dir}")
                model.save_pretrained(script_args.output_dir)
                logging.info(f"Model saved")
                best_model = deepcopy(model).to("cpu")
            else:
                logging.info(f"Average reward of {current_avg_reward:.4f} on validation set is not better than previous best of {best_initial_avg_reward:.4f}")
            model.train()
    if best_model is None:
        # Save the last checkpoint with warning
        logging.warning(f"Saving the last checkpoint at {script_args.output_dir}")
        model.save_pretrained(script_args.output_dir)
        logging.warning(f"Last checkpoint saved")
    else:
        logging.info(f"Best model already saved at {script_args.output_dir}")
    breakpoint()
    # To Hoard
    breakpoint()
