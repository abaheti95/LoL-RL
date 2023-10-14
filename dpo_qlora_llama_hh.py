# We will modify this file to create a dpo trainer for llama on hh-rlhf dataset
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
from utils.utils import RANDOM_SEED, save_in_pickle, load_from_pickle, make_dir_if_not_exists, reduce_mean, save_in_jsonl
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
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    # LR in original paper
    # learning_rate: float = field(default=1e-6, metadata={"help": 'The learning rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_steps: int = field(default=300, metadata={"help": 'Number of warmup steps for learning rate schedule'})
    # warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})

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
    max_steps: Optional[int] = field(default=9000, metadata={"help": "max number of training steps"})
    # max_steps: Optional[int] = field(default=10000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=2500, metadata={"help": "the saving frequency"})
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

    # Extra args for DPO
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    cache_dir: Optional[str] = field(default="cache/", metadata={"help": "the cache directory"})
    reference_free: Optional[bool] = field(default=False, metadata={"help": "whether to use reference-free DPO"})
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

def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    torch.cuda.manual_seed_all(script_args.seed)
    logging.info(f"Set all seeds to {script_args.seed}")
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
            bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_compute_dtype=torch.bfloat16,
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
    
    print_gpu_info(handle0, 0)
    print_gpu_info(handle1, 1)
    # Load the reference model for DPO
    if not script_args.reference_free:
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
    make_dir_if_not_exists(script_args.cache_dir)
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

    
    # Get value and advantage estimates on train dataset
    all_good_rewards = torch.sigmoid(torch.tensor([datum[0] for datum in train_dataset['reward']])).cpu().detach().numpy()
    all_bad_rewards = torch.sigmoid(torch.tensor([datum[1] for datum in train_dataset['reward']])).cpu().detach().numpy()
    # Train the model with a_lol

    # 1. Create a new optimizer for peftmodel
    from transformers import Trainer
    script_args.optim_args = {}
    # script_args.adam_beta1 = 0.9
    # script_args.adam_beta2 = 0.999
    # script_args.adam_epsilon = 1e-6
    optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(script_args)
    logging.info(f"Using optimizer {optim_cls} with kwargs {optim_kwargs}")
    # TODO: Check later if this is the right optimizer
    optimizer = optim_cls(model.parameters(), **optim_kwargs)
    script_args.warmup_steps = 300
    scheduler = get_scheduler(script_args.lr_scheduler_type, optimizer, num_warmup_steps=script_args.warmup_steps, num_training_steps=script_args.max_steps* script_args.gradient_accumulation_steps)
    
    # 2. Start custom training loop
    logging.info(f"Shuffling the train indices.")
    train_indices = list(range(len(train_dataset)))
    current_train_indices = deepcopy(train_indices)
    random.shuffle(current_train_indices)
    sampler = iter(current_train_indices)
    
    pbar = tqdm(range(script_args.max_steps * script_args.gradient_accumulation_steps), desc=f"Training the model with DPO")

    prompt_prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    origin_state = (tokenizer.padding_side, tokenizer.truncation_side)
    total_loss = 0.0
    total_per_resp_loss_with_IS = 0.0
    total_advantage = 0.0
    total_reward = 0.0
    total_suffix_distribution = {0:0, 1:0}
    logging.info(f"Evaluating every {script_args.eval_steps * script_args.gradient_accumulation_steps} steps.")
    loss_trajectories = defaultdict(list)
    best_model = None
    for step in pbar:
        # 1. Get the next batch of data from sampler
        try:
            batch_indices = [next(sampler) for _ in range(script_args.per_device_train_batch_size)]
        except StopIteration:
            logging.info(f"Shuffling the train indices.")
            current_train_indices = deepcopy(train_indices)
            random.shuffle(current_train_indices)
            sampler = iter(current_train_indices)
            batch_indices = [next(sampler) for _ in range(script_args.per_device_train_batch_size)]
        
        batch = [train_dataset[i] for i in batch_indices]
        # 2. Preprocess the batch data for model
        # 2.1. Get the batch prefix and suffix
        batch_og_prefixes = [datum['prefix'][0] for datum in batch]
        batch_prefixes = ["".join(prefix) for prefix in batch_og_prefixes]
        batch_prefixes_fixed = [prompt_prefix + prefix_str.replace("<|prompter|>", " ### Human: ").replace("<|assistant|>", " ### Assistant: ").strip() for prefix_str in batch_prefixes]
        batch_good_suffixes = [datum['suffix'][0] + tokenizer.eos_token for i, datum in enumerate(batch)]
        batch_bad_suffixes = [datum['suffix'][1] + tokenizer.eos_token for i, datum in enumerate(batch)]
        all_prefixes = batch_prefixes_fixed + batch_prefixes_fixed
        all_suffixes = batch_good_suffixes + batch_bad_suffixes
        # batch_good_suffixes = [datum['suffix'][0] for datum in batch]
        # batch_bad_suffixes = [datum['suffix'][1] for datum in batch]
        batch_good_rewards = torch.sigmoid(torch.tensor([datum['reward'][0] for datum in batch]))
        batch_bad_rewards = torch.sigmoid(torch.tensor([datum['reward'][1] for datum in batch]))
        # 2.2 Get left truncated prefix input
        tokenizer.truncation_side, tokenizer.padding_side = "left", "left"
        current_batch_prefixes_inputs = tokenizer(all_prefixes, max_length = 768 - 128,truncation = True,add_special_tokens=True, padding = True, return_tensors="pt").to(model.device)
        # 2.3 Get right truncated suffix input
        # NOTE: Adding tokenizer_eos token to the end of the suffixes
        tokenizer.truncation_side, tokenizer.padding_side = "right", "right"
        current_batch_suffixes_inputs = tokenizer(all_suffixes, max_length = 128,truncation = True,add_special_tokens=False, padding = True, return_tensors="pt").to(model.device)

        # Merge the chosen and rejected inputs into a single batch
        input_ids = torch.cat([current_batch_prefixes_inputs["input_ids"], current_batch_suffixes_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([current_batch_prefixes_inputs["attention_mask"], current_batch_suffixes_inputs["attention_mask"]], dim=1)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, query_seq_len = current_batch_prefixes_inputs["input_ids"].shape
        resp_logits = logits[:, (query_seq_len - 1):-1, :]

        # Gather per label logits
        labels = current_batch_suffixes_inputs["input_ids"]
        response_ids = current_batch_suffixes_inputs["input_ids"]
        response_mask = current_batch_suffixes_inputs["attention_mask"]
        # Logits are before softmax. So, we need to apply softmax to get the probabilities
        # logits is of the shape [batch_size, seq_len, vocab_size]
        # labels is of the shape [batch_size, seq_len]
        resp_log_probs = F.log_softmax(resp_logits, dim=-1)
        per_token_log_probs = torch.gather(resp_log_probs, 2, labels[:, :, None]).squeeze(2)
        resp_logps = (per_token_log_probs * response_mask).sum(-1)
        batch_size = len(batch)
        chosen_logps = resp_logps[:batch_size]
        rejected_logps = resp_logps[batch_size:]
        
        # Similarly compute reference model chosen and rejected logps
        if not script_args.reference_free:
            with torch.no_grad():
                reference_outputs = ref_model(input_ids, attention_mask=attention_mask)
                reference_logits = reference_outputs.logits
                reference_resp_logits = reference_logits[:, (query_seq_len - 1):-1, :]
                reference_resp_log_probs = F.log_softmax(reference_resp_logits, dim=-1)
                reference_per_token_log_probs = torch.gather(reference_resp_log_probs, 2, labels[:, :, None]).squeeze(2)
                reference_resp_logps = (reference_per_token_log_probs * response_mask).sum(-1)
                reference_chosen_logps = reference_resp_logps[:batch_size]
                reference_rejected_logps = reference_resp_logps[batch_size:]
            
            losses, chosen_rewards, rejected_rewards = dpo_loss(chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, beta=script_args.beta, reference_free=False)
        else:
            losses, chosen_rewards, rejected_rewards = dpo_loss(chosen_logps, rejected_logps, 0.0, 0.0, beta=script_args.beta, reference_free=True)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        margins = chosen_rewards - rejected_rewards
        loss = losses.mean()

        loss_trajectories["loss"].extend(losses.cpu().detach().numpy().tolist())
        loss_trajectories["chosen_reward"].extend(chosen_rewards.cpu().numpy().tolist())
        loss_trajectories["rejected_reward"].extend(rejected_rewards.cpu().numpy().tolist())
        loss_trajectories["reward_accuracy"].extend(reward_accuracies.cpu().numpy().tolist())
        loss_trajectories["margin"].extend(margins.cpu().numpy().tolist())
        
        postfix_dict = dict()
        postfix_dict["loss"] = np.mean(loss_trajectories["loss"][-500:])
        postfix_dict["chosen_reward"] = np.mean(loss_trajectories["chosen_reward"][-500:])
        postfix_dict["rejected_reward"] = np.mean(loss_trajectories["rejected_reward"][-500:])
        postfix_dict["reward_accuracy"] = np.mean(loss_trajectories["reward_accuracy"][-500:])
        postfix_dict["margin"] = np.mean(loss_trajectories["margin"][-500:])


        # 4. Backpropagate the loss and update the model parameters if gradient accumulation is done
        loss.backward()
        if (step+1) % script_args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), script_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # 5. Log the loss
        total_loss += loss.item()
        avg_total_loss = total_loss / (step+1)
        postfix_dict["step_loss"] = loss.item()
        # logging.info(f"Before clear cache")
        # print_gpu_info(handle0, 0)
        # print_gpu_info(handle1, 1)
        del loss, logits
        torch.cuda.empty_cache()
        # logging.info(f"After clear cache")
        # print_gpu_info(handle0, 0)
        # print_gpu_info(handle1, 1)
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
