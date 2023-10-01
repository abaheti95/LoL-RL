# We will modify this file to create a qlora llama evaluator on hh-rlhf dataset
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
from metrics_hh import create_reward_fn_2 as create_reward_fn

import logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
from time import time
import pandas as pd
import numpy as np
from copy import deepcopy
from utils import RANDOM_SEED, save_in_jsonl, make_dir_if_not_exists
import random
random.seed(RANDOM_SEED)

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
    
    max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
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

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    max_memory_MB: int = field(default=40000,metadata={"help": "Free memory per gpu."})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=2500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output dir where we will save the test results"})
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
    algorithm: Optional[str] = field(default="nll", metadata={"help": "the algorithm to use for training choices=['nll', 'a_lol_simple']"})


def evaluate_on_filepath(filepath, args, model, tokenizer, get_score, reward_batch_size):
    model.eval()
    
    logging.info(f"Loading the evaluation dataset from {filepath}")
    with open(filepath, "r", encoding='utf-8') as f:
        infer_data = [json.loads(l) for l in f.readlines()]
    
    # Subsampling for debugging
    # infer_data = infer_data[:32]
    prefixes = list()
    fixed_prefixes = list()
    gen_suffixes = list()

    origin_state = (tokenizer.padding_side, tokenizer.truncation_side)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    
    with torch.no_grad():
        for i in tqdm(range(0, len(infer_data), args.per_device_eval_batch_size), desc=f"Generating responses with batch_size {args.per_device_eval_batch_size}"):
            current_batch_data = infer_data[i:i+args.per_device_eval_batch_size]
            current_batch_og_prefixes = [datum['prefix'][0] for datum in current_batch_data]
            current_batch_prefixes = ["".join(prefix) for prefix in current_batch_og_prefixes]
            # prefix_str.replace("<|prompter|>", "\n ### Human: ").replace("<|assistant|>", "\n ### Assistant: ").strip()
            current_batch_prefixes_fixed = [prefix_str.replace("<|prompter|>", "\nHuman: ").replace("<|assistant|>", "\nAssistant: ").strip() for prefix_str in current_batch_prefixes]
            current_batch_prefixes_inputs = tokenizer(current_batch_prefixes_fixed, max_length = 512 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.device)
            # breakpoint()
            # decoded = tokenizer.batch_decode(current_batch_prefixes_inputs["input_ids"], skip_special_tokens=True)
            # tokenizer.convert_ids_to_tokens([835])
            # human = tokenizer.encode("\nHuman:", add_special_tokens=False)
            # tokenizer.convert_ids_to_tokens(human)
            # (current_batch_prefixes_inputs["input_ids"] == 835)
            predicted_sents = model.generate(**current_batch_prefixes_inputs, max_new_tokens=128,pad_token_id=tokenizer.pad_token_id,num_beams=1,do_sample=False,num_return_sequences = 1)
            repsonse_tokens = predicted_sents[:, current_batch_prefixes_inputs['input_ids'].shape[-1]:]
            responses = tokenizer.batch_decode(repsonse_tokens, skip_special_tokens=True)
            # Normalize responses
            # Only for the baseline checkpoint. Split at \nHuman:
            responses_normalized = [resp.split("\n Human:")[0].split("\nHuman:")[0].split("\n Assistant:")[0].split("\nAssistant:")[0].strip() for resp in responses]
            responses_normalized = [resp.replace("###", "").strip() if resp.endswith("###") else resp.strip() for resp in responses_normalized]
            # responses_normalized = [resp.replace("###", "").strip() if resp.endswith("###") else resp.strip() for resp in responses]
            prefixes.extend(current_batch_og_prefixes)
            fixed_prefixes.extend(current_batch_prefixes_fixed)
            gen_suffixes.extend(responses_normalized)
            # TEMP: For debugging
            # if i >= 100:
            #     break
    all_val_rewards = list()
    torch.cuda.empty_cache()
    for i in tqdm(range(0, len(gen_suffixes), reward_batch_size), desc="Calculating rewards"):
        batch_suffixes = gen_suffixes[i:i+reward_batch_size]
        batch_prefixes = prefixes[i:i+reward_batch_size]
        batch_rewards = torch.sigmoid(get_score(batch_prefixes, batch_suffixes)).cpu().detach().numpy().tolist()
        torch.cuda.empty_cache()
        all_val_rewards.extend(batch_rewards)
    avg_reward = np.mean(all_val_rewards)
    
    tokenizer.padding_side, tokenizer.truncation_side = origin_state
    # model_evaluator = DecoderUtils(self.model, args.world_size)
    # # Perform decoding and loss calculations here
    # _, wer = model_evaluator.evaluate(eval_dataloader, tokenizer)
    metrics = {'avg_val_reward': avg_reward}
    logging.info(f"Average reward on test set of {len(gen_suffixes)} samples is {avg_reward:.4f}")
    all_val_rewards = np.array(all_val_rewards)

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
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle0 = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    handle1 = nvidia_smi.nvmlDeviceGetHandleByIndex(1)
    def print_gpu_info(handle, id):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        logging.info(f"GPU{id}: Total memory: {info.total} Free memory: {info.free} Used memory: {info.used}")
    start_time = time()
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory= {i: '40000MB' for i in range(torch.cuda.device_count())},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ))    
    logging.info(f"Loaded the model in {time() - start_time} seconds")
    print_trainable_parameters(model)
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    # breakpoint()
    # Load the tokenizer
    if script_args.model_name_or_path == "reciprocate/ppo_hh_neox-20B":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=True)
        logging.info(f"Loaded tokenizer from EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=True)
        logging.info(f"Loaded tokenizer from {script_args.model_name_or_path}")
    # tokenizer = GPTNeoXTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False)
    # Add pad token if missing
    if tokenizer._pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the reward model
    logging.info(f"Initializing the reward model.")
    start_time = time()
    get_score, reward_batch_size = create_reward_fn(reward_batch_size=4)
    logging.info(f"Initialized the reward model with batch size {reward_batch_size} in {time() - start_time} seconds")
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)

    # Perform an initial validation
    test_dir = "data/hh_test/"
    test_files = ["harmless_base.json", "helpful_base.json", "helpful_online.json", "helpful_rejection.json"]
    test_filepaths = [os.path.join(test_dir, test_file) for test_file in test_files]
    make_dir_if_not_exists(script_args.output_dir)
    for test_file, test_filepath in zip(test_files, test_filepaths):
        save_filebase = test_file.split(".")[0]
        logging.info(f"Evaluating {test_filepath}")
        save_filepath = os.path.join(script_args.output_dir, save_filebase + "_eval_results.jsonl")
        # if os.path.exists(save_filepath):
        #     logging.info(f"Skipping {test_filepath} as {save_filepath} already exists.")
        #     continue
        all_val_gen_rewards, prefixes, fixed_prefixes, gen_suffixes = evaluate_on_filepath(test_filepath, script_args, model, tokenizer, get_score, reward_batch_size)
        # script_args.output_filepath = "./checkpoints/sft_qlora_llama/eval_results.jsonl"
        # Combine fixed prefixes, gen suffixes and rewards into a list of dictionary
        eval_results = [{"prefix": prefix, "suffix": suffix, "reward": reward} for prefix, suffix, reward in zip(fixed_prefixes, gen_suffixes, all_val_gen_rewards)]

        save_in_jsonl(eval_results, save_filepath)
        logging.info(f"Saved {len(eval_results)} evaluation results in {save_filepath}")
    breakpoint()
    
