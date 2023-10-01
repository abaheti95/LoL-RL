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
from utils import RANDOM_SEED, save_in_pickle, load_from_pickle, make_dir_if_not_exists, save_in_jsonl
import random
random.seed(RANDOM_SEED)

from rl_utils import ValueHeadMLP, ValueHeadAttention, numba_choice
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the Analysis script.
    """
    train_file_path: Optional[str] = field(default="data/hh_train_len2", metadata={"help": "the path to the training data"})
    eval_file_path: Optional[str] = field(default="data/hh_dev/sampled_dev.json", metadata={"help": "the path to the evaluation data"})
    output_dir: Optional[str] = field(default="analysis/", metadata={"help": "the path to the output directory"})
    cache_dir: Optional[str] = field(default="cache/", metadata={"help": "the path to the cache directory"})

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    make_dir_if_not_exists(args.output_dir)
    # Load the train data
    logging.info(f"Loading train dataset from {args.train_file_path}")
    train_dataset = load_dataset("json", data_dir = args.train_file_path, data_files = "cleaner_train.json", streaming=False, split="train")
    # train_dataset = load_dataset("json", data_dir = args.train_file_path, data_files = "train.json", streaming=False, split="train")
    logging.info(f"Loaded train dataset with {len(train_dataset)} samples")

    # Get value and advantage estimates on train dataset
    train_advantage_cache_file = os.path.join(args.cache_dir, "train_advantage_cache.pkl")
    logging.info(f"Loading the train value, good and bad advantage estimates from {train_advantage_cache_file}")
    all_values, all_good_advantages, all_bad_advantages = load_from_pickle(train_advantage_cache_file)
    logging.info(f"Loaded {len(all_values)} the train value, good and bad advantage estimates from {train_advantage_cache_file}")
    all_values = np.array(all_values)
    all_good_rewards = torch.sigmoid(torch.tensor([rewards[0] for rewards in train_dataset["reward"]])).numpy()
    all_bad_rewards = torch.sigmoid(torch.tensor([rewards[1] for rewards in train_dataset["reward"]])).numpy()
    all_good_advantages = np.array(all_good_advantages)
    all_bad_advantages = np.array(all_bad_advantages)
    # Report percentiles of values, good and bad rewards and advantages
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    logging.info(f"Percentiles = {percentiles}")
    logging.info(f"Value percentiles: {np.percentile(all_values, percentiles)}")
    logging.info(f"Good reward percentiles: {np.percentile(all_good_rewards, percentiles)}")
    logging.info(f"Bad reward percentiles: {np.percentile(all_bad_rewards, percentiles)}")
    logging.info(f"Good advantage percentiles: {np.percentile(all_good_advantages, percentiles)}")
    logging.info(f"Bad advantage percentiles: {np.percentile(all_bad_advantages, percentiles)}")

    def make_data_dicts_readable(train_dataset, indices, all_values, all_good_rewards, all_bad_rewards, all_good_advantages, all_bad_advantages):
        instances = [train_dataset[i] for i in indices]
        readable_data_dicts = []
        for instance, index in zip(instances, indices):
            prefix = "".join(instance["prefix"][0]).replace("<|prompter|>", " ### Human: ").replace("<|assistant|>", " ### Assistant: ").strip()
            suffix_good = instance["suffix"][0]
            suffix_bad = instance["suffix"][1]
            value = all_values[index]
            good_reward = all_good_rewards[index]
            bad_reward = all_bad_rewards[index]

            good_advantage = all_good_advantages[index]
            bad_advantage = all_bad_advantages[index]
            readable_data_dicts.append({"prefix": prefix, "suffix_good": suffix_good, "suffix_bad": suffix_bad, "value": value, "good_reward": good_reward, "bad_reward": bad_reward, "good_advantage": good_advantage, "bad_advantage": bad_advantage})
        return readable_data_dicts

    # Save instances with bottom 20 values
    bottom_20_values_indices = np.argsort(all_values)[:20].tolist()
    bottom_20_values_readable_instances = make_data_dicts_readable(train_dataset, bottom_20_values_indices, all_values, all_good_rewards, all_bad_rewards, all_good_advantages, all_bad_advantages)
    save_file = os.path.join(args.output_dir, "bottom_20_values_readable_instances.json")
    save_in_jsonl(bottom_20_values_readable_instances, save_file)
    logging.info(f"Saved {len(bottom_20_values_readable_instances)} bottom values readable instances to {save_file}")
    # Save instances with top 20 values
    top_20_values_indices = np.argsort(all_values)[-20:].tolist()
    top_20_values_readable_instances = make_data_dicts_readable(train_dataset, top_20_values_indices, all_values, all_good_rewards, all_bad_rewards, all_good_advantages, all_bad_advantages)
    save_file = os.path.join(args.output_dir, "top_20_values_readable_instances.json")
    save_in_jsonl(top_20_values_readable_instances, save_file)
    logging.info(f"Saved {len(top_20_values_readable_instances)} top values readable instances to {save_file}")
    # Save instances with bottom 20 good advantages
    bottom_20_good_advantages_indices = np.argsort(all_good_advantages)[:20].tolist()
    bottom_20_good_advantages_readable_instances = make_data_dicts_readable(train_dataset, bottom_20_good_advantages_indices, all_values, all_good_rewards, all_bad_rewards, all_good_advantages, all_bad_advantages)
    save_file = os.path.join(args.output_dir, "bottom_20_good_advantages_readable_instances.json")
    save_in_jsonl(bottom_20_good_advantages_readable_instances, save_file)
    logging.info(f"Saved {len(bottom_20_good_advantages_readable_instances)} bottom good advantages readable instances to {save_file}")
    # Save instances with top 20 good advantages
    top_20_good_advantages_indices = np.argsort(all_good_advantages)[-20:].tolist()
    top_20_good_advantages_readable_instances = make_data_dicts_readable(train_dataset, top_20_good_advantages_indices, all_values, all_good_rewards, all_bad_rewards, all_good_advantages, all_bad_advantages)
    save_file = os.path.join(args.output_dir, "top_20_good_advantages_readable_instances.json")
    save_in_jsonl(top_20_good_advantages_readable_instances, save_file)
    logging.info(f"Saved {len(top_20_good_advantages_readable_instances)} top good advantages readable instances to {save_file}")
    # Save instances with bottom 20 bad advantages
    bottom_20_bad_advantages_indices = np.argsort(all_bad_advantages)[:20].tolist()
    bottom_20_bad_advantages_readable_instances = make_data_dicts_readable(train_dataset, bottom_20_bad_advantages_indices, all_values, all_good_rewards, all_bad_rewards, all_good_advantages, all_bad_advantages)
    save_file = os.path.join(args.output_dir, "bottom_20_bad_advantages_readable_instances.json")
    save_in_jsonl(bottom_20_bad_advantages_readable_instances, save_file)
    logging.info(f"Saved {len(bottom_20_bad_advantages_readable_instances)} bottom bad advantages readable instances to {save_file}")
    # Save instances with top 20 bad advantages
    top_20_bad_advantages_indices = np.argsort(all_bad_advantages)[-20:].tolist()
    top_20_bad_advantages_readable_instances = make_data_dicts_readable(train_dataset, top_20_bad_advantages_indices, all_values, all_good_rewards, all_bad_rewards, all_good_advantages, all_bad_advantages)
    save_file = os.path.join(args.output_dir, "top_20_bad_advantages_readable_instances.json")
    save_in_jsonl(top_20_bad_advantages_readable_instances, save_file)
    logging.info(f"Saved {len(top_20_bad_advantages_readable_instances)} top bad advantages readable instances to {save_file}")
    
    # Create violin plots
    # Estimate the advantage for all train examples
    logging.info(f"Estimating the advantage for all train examples")
    components = list()
    scores = list()
    sources = list()
    reward_plus_value = all_good_rewards + all_values
    reward_plus_value[all_good_advantages > 0.3] = np.nan
    for source, reward_or_advantage_list in [("values", all_values), ("good_rewards", all_good_rewards), ("good_advantages", all_good_advantages),  ("V + Good R", reward_plus_value), ("bad_rewards", all_bad_rewards), ("bad_advantages", all_bad_advantages)]:
        for reward_or_advantage in reward_or_advantage_list:
            scores.append(reward_or_advantage)
            sources.append(source)
    df = pd.DataFrame({"score": scores, "source": sources})
    # Plot violin plot
    # Set figure size
    plt.figure(figsize=(10, 9))
    violin_plot = sns.violinplot(data=df, x="source", y="score")
    violin_plot.set(xlabel="Segment Name", ylabel="Reward/Value/Advantage distribution")
    # Add total counts/percentage of instances for each threshold window
    # violin_plot.set_xticklabels(violin_plot.get_xticklabels(), rotation=90)
    xticklabels = ["Values", "Good Rewards", "Good Advantages", "V + Good R", "Bad Rewards", "Bad Advantages"]
    violin_plot.set_xticklabels(xticklabels, rotation=70)
    violin_plot.set_title(f"Good and Bad reward, value and advantage distribution for {len(all_values)} instances")
    violin_plot_save_file = os.path.join(args.output_dir, f"train_reward_vs_advantage_distribution.png")
    # Tight layout
    plt.tight_layout()
    violin_plot.figure.savefig(violin_plot_save_file, dpi=300)
    logging.info(f"Saved violin plot to {violin_plot_save_file}")
    plt.clf()
    plt.cla()

    # Count non nans in reward_plus_value
    neg_reward_plus_value = -reward_plus_value[~np.isnan(reward_plus_value)]
    # Percentiles of softmax_reward_plus_value
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    logging.info(f"neg_reward_plus_value percentiles: {np.percentile(neg_reward_plus_value, percentiles)}")
    softmax = np.exp(neg_reward_plus_value)/np.sum(np.exp(neg_reward_plus_value))
    logging.info(f"Softmax neg_reward_plus_value percentiles: {np.percentile(softmax, percentiles)}")
    norm = neg_reward_plus_value/np.sum(neg_reward_plus_value)
    logging.info(f"Norm neg_reward_plus_value percentiles: {np.percentile(norm, percentiles)}")
    breakpoint()

