# We will combine the results from generation task off-policy algo variations evaluations into readable csv files.

import os
from collections import defaultdict, Counter
from ast import literal_eval
import ast
import re
from time import time
from tqdm import tqdm
import math
from copy import deepcopy

from utils.utils import load_from_pickle, load_from_jsonl, RANDOM_SEED, make_dir_if_not_exists, save_in_pickle, save_in_jsonl, log_list, save_in_json, distinctness, remove_multiple_space, save_list_of_tuples_to_tsv
from utils.attributes_utils import metric_to_key_mapping

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-bmps", "--baseline_model_prefixes", help="Dictionary of Baseline models prefixes for current task with bool variable indicating presence of a2c", type=str, required=True)
parser.add_argument("-tn", "--task_name", help="Name of the RL4LMs task useful to keep track of reward functions", type=str, required=True)
parser.add_argument("-o", "--output_file", help="Output csv file", type=str, required=True)
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import numpy as np
import random
random.seed(RANDOM_SEED)

def main():
    # Read the baseline_model_prefixes
    baseline_model_prefixes = literal_eval(args.baseline_model_prefixes)
    logging.info(f"baseline_model_prefixes: {baseline_model_prefixes}")
    task_name = args.task_name
    task_name_to_task_key = {"Xsum": "xsum", 
                             "CNNDailyMail": "cnn", 
                             "IWSLT2017EnDe": "iwslt17ende", 
                             "IMDBForSeq2Seq": "imdb_pos",
                             "DailyDialog": "daily_dialog",
                             "COMET": "comet",
                             "WOW": "wow",
                             "reddit_pos": "reddit_pos",
                             "reddit_neg": "reddit_neg",}
    task_key = task_name_to_task_key[task_name]
    logging.info(f"task_name: {task_name} task_key: {task_key}")
    # alg_suffixes = ["", "nll", "pg", "offline_pg_seq", "offline_pg_seq_but", "offline_pg_seq_clip", "offline_pg_seq_clip_but"]
    # NOTE: Removing the but versions of the offline_pg_seq because they usually don't perform well
    # alg_suffixes = ["", "nll", "pg", "offline_pg_seq", "offline_pg_seq_clip"]
    alg_suffixes = ["", "nll", "pg", "offline_pg_seq_clip_sample"]
    # a2c_alg_suffixes = ["offline_a2c", "offline_a2c_clip"]
    a2c_alg_suffixes = ["offline_a2c_clip_sample"]
    
    success_metric = "meteor"
    if task_name == "IMDBForSeq2Seq":
        success_metric = "sentiment"
    elif task_name == "COMET":
        success_metric = "p_valid_model"
    elif task_name == "WOW":
        success_metric = "faithdial"
    elif task_name in ["reddit_pos", "reddit_neg"]:
        success_metric = "final_reward"
    elif task_name == "DailyDialog":
        success_metric = "-ve perplexity"
    
    header_row = ["task_name", "baseline_model", "algorithm", "best e", "best s", f"best V {success_metric}", "worst e", "worst s", f"worst V {success_metric}", f"val_{success_metric}", f"test_{success_metric}"]
    all_csv_rows = list()
    rewards_header = None
    for baseline_model_prefix, include_a2c in baseline_model_prefixes.items():
        current_model_alg_suffixes = deepcopy(alg_suffixes)
        if include_a2c:
            current_model_alg_suffixes.extend(a2c_alg_suffixes)
        for alg_suffix in current_model_alg_suffixes:
            if alg_suffix == "":
                model_output_dir = f"final_results/GEM/{task_key}/{baseline_model_prefix}/train_log"
                model_output_test_dir = f"final_results/GEM/{task_key}/{baseline_model_prefix}/test_log"
            else:
                model_output_dir = f"final_results/GEM/{task_key}/{baseline_model_prefix}_{alg_suffix}/train_log"
                model_output_test_dir = f"final_results/GEM/{task_key}/{baseline_model_prefix}_{alg_suffix}/test_log"
            
            final_eval_stats_file = f"{model_output_test_dir}/test_final_eval_stats.jsonl"
            if not os.path.exists(final_eval_stats_file):
                final_eval_stats_file = f"{model_output_dir}/train_final_eval_stats.jsonl"
            # NOTE: Ignoring the test stats for now
            final_eval_stats_file = f"{model_output_dir}/train_final_eval_stats.jsonl"
            logging.info(f"final_eval_stats_file: {final_eval_stats_file}")
            
            current_model_alg_csv_row = [task_name, baseline_model_prefix, alg_suffix]
            # Check for per epoch eval stats
            validation_stats_file = f"{model_output_dir}/validation_stats.jsonl"
            
            if os.path.exists(validation_stats_file):
                # Read the file
                validation_stats = load_from_jsonl(validation_stats_file)
                # validation_stats is a list of dict with dict_keys(['epoch', 'step', 'meteor', 'gen_reward_avg', 'val_reward_avg'])
                # find the best and worst val meteor
                best_val_success_metric = -100
                worst_val_success_metric = 100
                best_val_success_metric_epoch = None
                best_val_success_metric_step = None
                worst_val_success_metric_epoch = None
                worst_val_success_metric_step = None
                for index, val_stat in enumerate(validation_stats):
                    if success_metric == "meteor":
                        current_val_success_metric = val_stat[success_metric]
                    elif success_metric in ["sentiment", "p_valid_model", "faithdial", "final_reward"]:
                        current_val_success_metric = val_stat["gen_reward_avg"][success_metric]
                    elif success_metric == "-ve perplexity":
                        current_val_success_metric = val_stat[success_metric] if success_metric in val_stat else val_stat["-ve perlexity"]

                    if current_val_success_metric > best_val_success_metric:
                        best_val_success_metric = current_val_success_metric
                        best_val_success_metric_epoch = val_stat["epoch"]
                        best_val_success_metric_step = val_stat["step"]
                    if current_val_success_metric < worst_val_success_metric:
                        worst_val_success_metric = current_val_success_metric
                        worst_val_success_metric_index = index
                        worst_val_success_metric_epoch = val_stat["epoch"]
                        worst_val_success_metric_step = val_stat["step"]
                current_model_alg_csv_row.extend([best_val_success_metric_epoch, best_val_success_metric_step, f"{best_val_success_metric:.3f}", worst_val_success_metric_epoch, worst_val_success_metric_step, f"{worst_val_success_metric:.3f}"])
            else:
                current_model_alg_csv_row.extend(["NA", "NA", "NA", "NA", "NA", "NA"])
            # Check if the file exists
            if not os.path.exists(final_eval_stats_file):
                logging.info(f"File {final_eval_stats_file} does not exist")
                val_success_metric = "NA"
                test_success_metric = "NA"
                reward_row = list()
            else:
                # Read the file
                train_final_eval_stats = load_from_jsonl(final_eval_stats_file)
                # Add success metric to the table
                val_stats = train_final_eval_stats[0]["val"]
                if success_metric == "meteor":
                    val_success_metric = f"{val_stats['meteor']:.3f}"
                elif success_metric in ["sentiment", "p_valid_model", "faithdial", "final_reward"]:
                    val_success_metric = f"{val_stats['gen_reward_avg'][success_metric]:.3f}"
                elif success_metric == "-ve perplexity":
                    val_success_metric = val_stats[success_metric] if success_metric in val_stats else val_stats["-ve perlexity"]
                test_stats = train_final_eval_stats[0]["test"]
                if success_metric == "meteor":
                    test_success_metric = f"{test_stats['meteor']:.3f}"
                elif success_metric in ["sentiment", "p_valid_model", "faithdial", "final_reward"]:
                    test_success_metric = f"{test_stats['gen_reward_avg'][success_metric]:.3f}"
                elif success_metric == "-ve perplexity":
                    test_success_metric = test_stats[success_metric] if success_metric in test_stats else test_stats["-ve perlexity"]

                # Also add the rewards to the table
                reward_types = list(val_stats["gen_reward_avg"].keys())
                if rewards_header is None:
                    # Prepare the rewards header and add them to the header row
                    # val gen rewards, val gold rewards, test gen rewards, test gold rewards
                    rewards_header = list()
                    for segment in ["val_gen", "test_gen", "val_gold", "test_gold"]:
                        for reward_type in reward_types:
                            rewards_header.append(f"{segment} {reward_type}")
                    header_row.extend(rewards_header)
                val_gen_rewards = [f"{val_stats['gen_reward_avg'][reward_type]:.3f}" for reward_type in reward_types]
                test_gen_rewards = [f"{test_stats['gen_reward_avg'][reward_type]:.3f}" for reward_type in reward_types]
                val_gold_rewards = [f"{val_stats['val_reward_avg'][reward_type]:.3f}" for reward_type in reward_types]
                test_gold_rewards = [f"{test_stats['test_reward_avg'][reward_type]:.3f}" for reward_type in reward_types]
                reward_row = val_gen_rewards + test_gen_rewards + val_gold_rewards + test_gold_rewards
            # Add to csv rows
            current_model_alg_csv_row.extend([val_success_metric, test_success_metric])
            current_model_alg_csv_row.extend(reward_row)
            all_csv_rows.append(current_model_alg_csv_row)
        
    logging.info(header_row)
    log_list(all_csv_rows)
    # Save the csv file with header
    save_list_of_tuples_to_tsv(all_csv_rows, args.output_file, header=header_row, delimiter=',')
    logging.info(f"Saved {len(all_csv_rows)} rows to {args.output_file}")
        


if __name__ == "__main__":
    main()

