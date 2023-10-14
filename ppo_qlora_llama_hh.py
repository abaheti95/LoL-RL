# We will modify this file to create a lolrl trainer for llama on hh-rlhf dataset
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


import os
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Adafactor, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import math
from scipy import stats as scipy_stats

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



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
from utils.utils import RANDOM_SEED, save_in_pickle, load_from_pickle, make_dir_if_not_exists, reduce_mean, save_in_jsonl, reduce_sum
import random

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
    # save_steps: Optional[int] = field(default=2500, metadata={"help": "the saving frequency"})
    # max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})

    output_dir: Optional[str] = field(default="checkpoints/ppo_qlora_llama", metadata={"help": "the output directory"})
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
    algorithm: Optional[str] = field(default="nll", metadata={"help": "the algorithm to use for training choices=['nll', 'wbc', 'a_lol', 'a_lol_seq', 'a_lol_simple']"})
    ppo_clip: Optional[float] = field(default=0.2, metadata={"help": "the clipping parameter for PPO"})
    kl_beta: Optional[float] = field(default=0.2, metadata={"help": "the beta parameter for KL penalty"})
    sampling_strategy: Optional[str] = field(default=None, metadata={"help": "the sampling strategy to use for advantage LoL RL methods"})
    cache_dir: Optional[str] = field(default="cache/", metadata={"help": "the cache directory"})

    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    total_steps: Optional[int] = field(default=1000, metadata={"help": "total steps of training"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    # learning_rate: float = field(default=0.00002, metadata={"help": 'The learning rate'})
    # learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    # ppo_rollout_batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_rollout_batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    seed: Optional[int] = field(default=RANDOM_SEED, metadata={"help": "the random seed"})

def evaluate_on_validation(args, ppo_trainer, model, tokenizer, get_score, reward_batch_size):
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
            if hasattr(model, "device"):
                current_batch_prefixes_inputs = tokenizer(current_batch_prefixes_fixed, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.device)
            else:
                current_batch_prefixes_inputs = tokenizer(current_batch_prefixes_fixed, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.pretrained_model.device)
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
            prefixes.extend(current_batch_og_prefixes)
            fixed_prefixes.extend(current_batch_prefixes_fixed)
            gen_suffixes.extend(responses_normalized)
            # TEMP: to debug
            # if i > 10:
            #     break
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
    logging.info(f"Average reward on validation set of {len(gen_suffixes)} samples is {avg_reward:.4f}")
    all_val_rewards = np.array(all_val_rewards)
    model.train()
    return all_val_rewards, prefixes, fixed_prefixes, gen_suffixes


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

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    torch.cuda.manual_seed_all(script_args.seed)
    set_seed(script_args.seed)
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
    peft_model = PeftModel.from_pretrained(base_model, script_args.adapter_path, is_trainable=True)
    logging.info(f"Loaded the model in {time() - start_time} seconds")
    peft_model.train()
    print_trainable_parameters(peft_model)
    logging.info(f"Loading AutoModelForCausalLMWithValueHead from peft_model obtained from {script_args.adapter_path}")
    start_time = time()
    peft_config = peft_model.peft_config['default']
    # Convert PeftModel to AutoModelForCausalLMWithValueHead
    model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model, peft_config=peft_config, device_map="auto")
    print_trainable_parameters(model)
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    logging.info(f"Loaded the model with value head in {time() - start_time} seconds")

    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    # Load the reference model for DPO
    if script_args.target_kl != 0:
        # Sharing backbone between training and behavior policy is not permitted yet: https://github.com/huggingface/peft/issues/854
        logging.info(f"Loading another base model for reference model from {script_args.model_name_or_path}")
        ref_model_base_model = AutoModelForCausalLM.from_pretrained(
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
        # Initialize a reference policy model
        logging.info(f"Initializing untrainable the reference policy model from {script_args.adapter_path}")
        ref_model = PeftModel.from_pretrained(ref_model_base_model, script_args.adapter_path, is_trainable=False)
        ref_model.eval()
        print_trainable_parameters(ref_model)
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
        all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes = evaluate_on_validation(script_args, None, model, tokenizer, get_score, reward_batch_size)
        save_in_pickle([all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes], eval_cache_file)
        logging.info(f"Saved the validation results in {eval_cache_file}")
    best_initial_avg_reward = np.mean(all_val_gen_rewards)
    logging.info(f"Best initial average reward on validation set of {len(gen_suffixes)} samples is {best_initial_avg_reward:.4f}")
    
    # Create an empty eval trajectory file
    make_dir_if_not_exists(script_args.output_dir)
    eval_trajectory_file = os.path.join(script_args.output_dir, "eval_trajectory.jsonl")
    open(eval_trajectory_file, "w").close()
    save_in_jsonl([{"step":0, "avg_reward": best_initial_avg_reward}], eval_trajectory_file, append=True)

    all_good_rewards = torch.sigmoid(torch.tensor([datum[0] for datum in train_dataset['reward']])).cpu().detach().numpy()
    all_bad_rewards = torch.sigmoid(torch.tensor([datum[1] for datum in train_dataset['reward']])).cpu().detach().numpy()
    
    
    config = PPOConfig(
        steps=script_args.total_steps,
        learning_rate=script_args.learning_rate,
        batch_size=script_args.ppo_rollout_batch_size,
        mini_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=script_args.target_kl,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed,
        use_score_scaling = True,
        use_score_norm = True
    )
    

    def collator(data):
        breakpoint()
        return dict((key, [d[key] for d in data]) for key in data[0])

    # 1. Create a new optimizer for peftmodel
    from transformers import Trainer
    script_args.optim_args = {}
    # script_args.adam_beta1 = 0.9
    # script_args.adam_beta2 = 0.999
    # script_args.adam_epsilon = 1e-6
    optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(script_args)
    # TODO: Check later if this is the right optimizer
    optimizer = optim_cls(model.parameters(), **optim_kwargs)
    # NOTE: PPOTrainer doesn't use scheduler
    # script_args.warmup_steps = 300
    # scheduler = get_scheduler(script_args.lr_scheduler_type, optimizer, num_warmup_steps=script_args.warmup_steps, num_training_steps=script_args.max_steps * script_args.gradient_accumulation_steps)
    
    # Train the model with PPO
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        # dataset=train_dataset,
        # data_collator=collator,
        optimizer=optimizer,
    )


    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "do_sample": True,
        "top_p": 0.95,
        "max_new_tokens": 128,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": [835, tokenizer.eos_token_id],
        "num_return_sequences": 1,
    }

    logging.info(f"Shuffling the train indices.")
    train_indices = list(range(len(train_dataset)))
    current_train_indices = deepcopy(train_indices)
    random.shuffle(current_train_indices)
    sampler = iter(current_train_indices)

    pbar = tqdm(range(script_args.total_steps), desc=f"Training the model with PPO")
    
    prompt_prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    origin_state = (tokenizer.padding_side, tokenizer.truncation_side)
    total_loss = 0.0
    total_per_resp_loss_before_reward = 0.0
    total_kl_penalty = 0.0
    total_alol_loss = 0.0
    total_reward = 0.0
    total_suffix_distribution = {0:0, 1:0}
    logging.info(f"Evaluating every {script_args.eval_steps} steps.")
    logging.info(f"Using algorithm: PPO")
    logging.info(f"Using learning rate: {script_args.learning_rate}")
    best_model = None
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_ppo_benefit = 0.0
    total_ppo_advantage = 0.0
    for step in pbar:
        # Generate batch_size responses in parallel for PPO
        try:
            batch_indices = [next(sampler) for _ in range(script_args.ppo_rollout_batch_size)]
        except StopIteration:
            logging.info(f"Shuffling the train indices.")
            current_train_indices = deepcopy(train_indices)
            random.shuffle(current_train_indices)
            sampler = iter(current_train_indices)
            batch_indices = [next(sampler) for _ in range(script_args.ppo_rollout_batch_size)]
    
        batch = [train_dataset[i] for i in batch_indices]
        # 2. Preprocess the batch data for model
        # 2.1. Get the batch prefix and suffix
        batch_og_prefixes = [datum['prefix'][0] for datum in batch]
        batch_prefixes = ["".join(prefix) for prefix in batch_og_prefixes]
        batch_prefixes_fixed = [prompt_prefix + prefix_str.replace("<|prompter|>", " ### Human: ").replace("<|assistant|>", " ### Assistant: ").strip() for prefix_str in batch_prefixes]
            
        # 2.2 Get left truncated prefix input
        tokenizer.truncation_side, tokenizer.padding_side = "left", "left"
        # Tokenize each prefix individually and save in list of tensors
        ppo_prefix_tensors = list()
        for fixed_prefix in batch_prefixes_fixed:
            fixed_prefix_tokenized = tokenizer(fixed_prefix, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt")
            ppo_prefix_tensors.append(fixed_prefix_tokenized["input_ids"][0])
        
        # logging.info(f"Generating responses for the batch of size {script_args.ppo_rollout_batch_size}")
        # start_time = time()
        response_tensors = ppo_trainer.generate(ppo_prefix_tensors,return_prompt=False,**generation_kwargs)
        # logging.info(f"Generated responses in {time() - start_time} seconds")
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        responses_normalized = [resp.split("\n Human:")[0].split("\nHuman:")[0].split("\n### Human")[0].strip() for resp in responses]
        responses_normalized = [resp.replace("###", "").strip() if resp.endswith("###") else resp.strip() for resp in responses_normalized]
        
        # Calculate rewards
        ppo_batch_rewards = list()
        # logging.info(f"Calculating rewards for the batch of size {script_args.ppo_rollout_batch_size} with reward batch size {reward_batch_size}")
        # start_time = time()
        for i in range(0, len(responses_normalized), reward_batch_size):
            batch_suffixes = responses_normalized[i:i+reward_batch_size]
            batch_prefixes = batch_og_prefixes[i:i+reward_batch_size]
            batch_rewards = torch.sigmoid(get_score(batch_prefixes, batch_suffixes)).cpu().detach().numpy().tolist()
            ppo_batch_rewards.extend(batch_rewards)
        # logging.info(f"Calculated rewards in {time() - start_time} seconds")
        
        # Run PPO step
        ppo_rewards = [torch.tensor(reward).float() for reward in ppo_batch_rewards]
        stats = ppo_trainer.step(ppo_prefix_tensors, response_tensors, ppo_rewards)
        # stats contain dict_keys(['objective/kl', 'objective/kl_dist', 'objective/logprobs', 'objective/ref_logprobs', 'objective/kl_coef', 'objective/entropy', 'ppo/mean_non_score_reward', 'ppo/mean_scores', 'ppo/std_scores', 'tokens/queries_len_mean', 'tokens/queries_len_std', 'tokens/queries_dist', 'tokens/responses_len_mean', 'tokens/responses_len_std', 'tokens/responses_dist', 'ppo/loss/policy', 'ppo/loss/value', 'ppo/loss/total', 'ppo/policy/entropy', 'ppo/policy/approxkl', 'ppo/policy/policykl', 'ppo/policy/clipfrac', 'ppo/policy/advantages', 'ppo/policy/advantages_mean', 'ppo/policy/ratio', 'ppo/returns/mean', 'ppo/returns/var', 'ppo/val/vpred', 'ppo/val/error', 'ppo/val/clipfrac', 'ppo/val/mean', 'ppo/val/var', 'ppo/val/var_explained', 'ppo/learning_rate', 'time/ppo/forward_pass', 'time/ppo/compute_rewards', 'time/ppo/compute_advantages', 'time/ppo/optimize_step', 'time/ppo/calc_stats', 'time/ppo/total'])
        postfix_dict = dict()
        total_kl_penalty += stats["objective/kl"]
        postfix_dict["kl_penalty"] = f"{total_kl_penalty / (step + 1):.4f}"
        total_entropy += stats["objective/entropy"]
        postfix_dict["entropy"] = f"{total_entropy / (step + 1):.4f}"
        total_policy_loss += stats["ppo/loss/policy"]
        total_value_loss += stats["ppo/loss/value"]
        total_ppo_advantage += stats["ppo/policy/advantages_mean"]
        # ppo_learning_rate = stats["ppo/learning_rate"]
        postfix_dict["policy_loss"] = f"{total_policy_loss / (step + 1):.4f}"
        postfix_dict["value_loss"] = f"{total_value_loss / (step + 1):.4f}"
        postfix_dict["advantage"] = f"{total_ppo_advantage / (step + 1):.4f}"
        # postfix_dict["lr"] = f"{ppo_learning_rate:.4e}"

        # Compare with gold rewards
        ppo_rewards = np.array(ppo_batch_rewards)
        gold_rewards = torch.sigmoid(torch.tensor([datum['reward'][0] for datum in batch])).cpu().detach().numpy()
        postfix_dict["gold_vs_ppo"] = f"{np.mean(gold_rewards):.4f} vs {np.mean(ppo_rewards):.4f}"
        # total_ppo_benefit += np.mean(ppo_rewards - gold_rewards)
        postfix_dict["ppo_time"] = stats["time/ppo/total"]
        pbar.set_postfix(postfix_dict)
        if (step+1) % script_args.eval_steps == 0:
            torch.cuda.empty_cache()
            all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes = evaluate_on_validation(script_args, ppo_trainer, model, tokenizer, get_score, reward_batch_size)
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

    # To Hoard
    breakpoint()
