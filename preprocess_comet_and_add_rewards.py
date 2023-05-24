# We will batch preprocess COMET ATOMIC10X and ATOMIC2020 and add rewards to it

from utils.utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, save_list_of_tuples_to_tsv, load_from_tsv_to_list_of_list, save_in_jsonl, load_from_jsonl, get_ngrams_from_sentence
import pdb

from transformers import GPT2Tokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, RobertaPreTrainedModel, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM, RobertaModel
from sentence_transformers import SentenceTransformer, util
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import torch
torch.manual_seed(RANDOM_SEED+1)

import random
random.seed(RANDOM_SEED)

import os
import re
import math
from time import time
import copy
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr

from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rl4lms.data_pools.custom_text_generation_pools import Xsum, CNNDailyMail, ToTTo, IMDB, IMDBForSeq2Seq, IWSLT2017EnDe, DailyDialog
# CommonGen, CNNDailyMail, IMDB, IMDBForSeq2Seq, WMT, WMT14PreprocessedEnDe, WMT16NewsOnlyDatasetEnDe, IWSLT2017EnDe, CRD3DialogueGeneration, DailyDialog 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="directory containing ATOMIC 10X data", type=str, required=True)
parser.add_argument("-it", "--input_test_dir", help="directory containing ATOMIC2020 data", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all preprocessed language generation data and rewards in jsonl files", type=str, required=True)
# Arguments regarding the reward models
parser.add_argument("-ccm", "--comet_critic_model", help="Directory containing COMET critic model", type=str, required=True)

parser.add_argument("-bs", "--batch_size", help="batch size for reward models", type=int, default=32)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Also add the stream handler so that it logs on STD out as well
# Ref: https://stackoverflow.com/a/46098711/4535284
make_dir_if_not_exists(args.output_dir)
logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

# Other global constants required for the code
MAX_SEQ_THRESH = 512

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using {device} to generate rewarded data")

import datasets


from sentence_transformers import SentenceTransformer, util

# To supress truncation warning. Ref: https://github.com/huggingface/transformers/issues/14285#issuecomment-968839045
import transformers
transformers.logging.set_verbosity_error()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_reward_components_distribution(reward_components_list, segment_name):
    components = list()
    rewards = list()
    for reward_components in reward_components_list:
        for component, reward in reward_components.items():
            components.append(component)
            rewards.append(reward)
    df = pd.DataFrame({"components": components, "reward": rewards})
    # Plot violin plot
    # Set figure size
    plt.figure(figsize=(10, 9))
    violin_plot = sns.violinplot(data=df, x="components", y="reward")
    violin_plot.set(xlabel="Reward Components", ylabel="Reward distribution")
    # Add total counts/percentage of instances for each threshold window
    xticklabels = list(reward_components_list[0].keys())
    # violin_plot.set_xticklabels(violin_plot.get_xticklabels(), rotation=90)
    violin_plot.set_xticklabels(xticklabels, rotation=70)
    violin_plot.set_title(f"Per component reward distribution for segment = {segment_name}, containing {len(reward_components_list)} instances")
    violin_plot_save_file = os.path.join(args.output_dir, f"per_component_reward_distribution_plot_{segment_name}.png")
    # Tight layout
    plt.tight_layout()
    violin_plot.figure.savefig(violin_plot_save_file, dpi=300)
    logging.info(f"Saved violin plot to {violin_plot_save_file}")
    plt.clf()
    plt.cla()


from utils.attributes_utils import get_cola_fluency_score
from utils.rl_utils import get_batch_comet_reward
from utils.comet_utils import RobertaClassificationHead, _RELATIONS
def main():
    # 1. Load the ATOMIC dataset
    start_time = time()
    atomic_jsonl_file = os.path.join(args.input_dir, "ATOMIC10X.jsonl")
    logging.info(f"Loading the ATOMIC dataset from {atomic_jsonl_file}")
    atomic_dataset = load_from_jsonl(atomic_jsonl_file)
    logging.info(f"Loaded the ATOMIC dataset containing {len(atomic_dataset)} instances in {time() - start_time:.2f} seconds")
    # 1.1 Load the ATOMIC2020 dataset
    start_time = time()
    atomic2020_jsonl_file = os.path.join(args.input_test_dir, "test.tsv")
    logging.info(f"Loading the ATOMIC2020 dataset from {atomic2020_jsonl_file}")
    atomic2020_dataset = load_from_tsv_to_list_of_list(atomic2020_jsonl_file)
    # atomic2020_prompt_list = [(d[0], d[1]) for d in atomic2020_dataset]
    # unique_atomic2020_prompt_list = list(set(atomic2020_prompt_list))
    # This list only contains 3 things, head, relation and tail
    logging.info(f"Loaded the ATOMIC2020 dataset containing {len(atomic2020_dataset)} instances in {time() - start_time:.2f} seconds")

    # 2. Load the COMET critic model and tokenizer
    reward_args = defaultdict(lambda: None)
    reward_args["device"] = device
    if args.comet_critic_model is not None:
        start_time = time()
        logging.info(f"Loading the pytorch version of COMET critic model, tokenizer and classification head from {args.comet_critic_model}")

        comet_roberta_model = RobertaModel.from_pretrained(args.comet_critic_model).to(device)
        comet_tokenizer = AutoTokenizer.from_pretrained(args.comet_critic_model)
        logging.info(f"Loaded the Comet Critic Roberta model and tokenizer from {args.comet_critic_model} in {time() - start_time:.2f} seconds")

        comet_classification_head = RobertaClassificationHead(1024, 512, 1)
        classification_head_torch_save_path = os.path.join(args.comet_critic_model, "custom_roberta_classification_head.pt")
        comet_classification_head.load_state_dict(torch.load(classification_head_torch_save_path))
        comet_classification_head.to(device)
        logging.info(f"Loaded the comet_classification_head from {classification_head_torch_save_path}")


        reward_args["comet_critic_model_name"] = args.comet_critic_model
        reward_args["comet_roberta_model"] = comet_roberta_model
        reward_args["comet_tokenizer"] = comet_tokenizer
        reward_args["comet_classification_head"] = comet_classification_head
        logging.info(f"Loaded the COMET Critic model and tokenizer in {time() - start_time:.2f} seconds")

    # Checking if the model predictions matches the expected output
    # atomic_test_dict = {"head": "PersonX decides to see a therapist", "relation": "xEffect", "tail": "feels better", "split": "train", "rec_0.6": True, "rec_0.9": True, "rec_0.5": True, "rec_0.7": True, "rec_0.8": True, "p_valid_model": 0.9934503436088562}
    # atomic_test_dict["valid"] = 1.0
    # test_x, label = get_comet_keras_input_and_labels([atomic_test_dict], comet_critic_tokenizer)

    # We need to first filter out the relations that are not present in SKD
    filtered_atomic2020_dataset = [d for d in atomic2020_dataset if d[1] in _RELATIONS]
    logging.info(f"Filtered the atomic2020 dataset to {len(filtered_atomic2020_dataset)} instances, from original {len(atomic2020_dataset)} instances")
    atomic2020_dataset = filtered_atomic2020_dataset
    # First create a test set using the atomic2020 dataset test file
    # We need to make batch predictions on the 130K test instances
    batch_size = args.batch_size
    atomic2020_test_dicts = list()
    for i in tqdm(range(0, len(atomic2020_dataset), batch_size), desc="Batch prediction on atomic2020 test set"):
        batch_data = atomic2020_dataset[i:i+batch_size]
        heads = [d[0] for d in batch_data]
        relations = [d[1] for d in batch_data]
        tails = [d[2] for d in batch_data]
        # Create the prompt
        batch_prompts = [f"<head> {head.strip()} </head> <relation> {relation} </relation> [GEN] " for head, relation in zip(heads, relations)]
        comet_critic_rewards = get_batch_comet_reward(batch_prompts, tails, reward_args)
        current_batch_dicts = [{"head": head, "relation": relation, "tail": tail, "p_valid_model": comet_critic_reward} for head, relation, tail, comet_critic_reward in zip(heads, relations, tails, comet_critic_rewards)]
        atomic2020_test_dicts.extend(current_batch_dicts)
    # within these dicts we need to pick the highest p_valid_model for each unique head, relation pair
    unique_atomic2020_test_dicts = dict()
    for atomic2020_test_dict in atomic2020_test_dicts:
        head, relation =  atomic2020_test_dict["head"], atomic2020_test_dict["relation"]
        if (head, relation) not in unique_atomic2020_test_dicts:
            unique_atomic2020_test_dicts[(head, relation)] = atomic2020_test_dict
        else:
            if atomic2020_test_dict["p_valid_model"] > unique_atomic2020_test_dicts[(head, relation)]["p_valid_model"]:
                unique_atomic2020_test_dicts[(head, relation)] = atomic2020_test_dict
    logging.info(f"Total number of unique head, relation pairs in atomic2020 test set = {len(unique_atomic2020_test_dicts)}")
    final_atomic2020_test_dicts = list(unique_atomic2020_test_dicts.values())
    # comet_critic_pred = comet_critic_model.predict(test_x)
    # logging.info(f"COMET critic prediction = {comet_critic_pred}")
    # enable tqdm enumerate
    segment_dicts = defaultdict(list)
    for index, atomic_dict in enumerate(tqdm(atomic_dataset, desc="Split ATOMIC dataset by split")):
        # atomic_dict contains dict_keys(['head', 'relation', 'tail', 'split', 'rec_0.6', 'rec_0.9', 'rec_0.5', 'rec_0.7', 'rec_0.8', 'p_valid_model'])
        split = atomic_dict["split"]
        segment_dicts[split].append(atomic_dict)
    # segment_dicts.keys() are dict_keys(['train', 'val', 'test'])
    train_dicts = segment_dicts["train"]
    val_dicts = segment_dicts["val"]
    test_dicts = segment_dicts["test"]
    logging.info(f"Total number of train, val and test instances = {len(train_dicts)}, {len(val_dicts)}, {len(test_dicts)}")
    logging.info(f"We will be using atomic2020 test set of size {len(atomic2020_test_dicts)} instead of atomic10x test set of size {len(test_dicts)}")

    # 3. Create the prompt, target and rewards for the dataset
    def process_dicts_for_segment(segment_dicts, segment_name):
        final_segment_dicts = list()
        for i, segment_dict in enumerate(tqdm(segment_dicts, desc=f"Processing {segment_name} segment")):
            # Each segment_dict contains dict_keys(['head', 'relation', 'tail', 'split', 'rec_0.6', 'rec_0.9', 'rec_0.5', 'rec_0.7', 'rec_0.8', 'p_valid_model'])
            # We need to convert the atomic dicts to the format "id", "prompt_or_input_text", "references", "meta_data", "reward_components"
            # Change reward function based on input_task mapping
            event = segment_dict['head']
            # one of ['xEffect','xAttr','xReact', 'xWant','xIntent', 'xNeed', 'HinderedBy']
            relation = segment_dict['relation']
            # This prompt is copied from: https://github.com/peterwestai2/symbolic-knowledge-distillation/blob/main/generate_COMET_distill.ipynb
            inp_text = f"<head> {event.strip()} </head> <relation> {relation} </relation> [GEN] "
            references = segment_dict['tail']
            meta_data = {"original_dict": segment_dict}
            reward_components = {"p_valid_model": segment_dict["p_valid_model"], "final_reward": segment_dict["p_valid_model"]}
            new_segment_dict = {"id": i, "prompt_or_input_text": inp_text, "references": references, "meta_data": meta_data, "reward_components": reward_components}
            final_segment_dicts.append(new_segment_dict)
        return final_segment_dicts
    
    # final_test_dicts = process_dicts_for_segment(test_dicts, "test")
    final_test_dicts = process_dicts_for_segment(test_dicts, "test")
    test_save_file = os.path.join(args.output_dir, "test_atomic10x.jsonl")
    logging.info(f"Saving {len(final_test_dicts)} atomic10x test dicts to {test_save_file}")
    save_in_jsonl(final_test_dicts, test_save_file)
    # Plot the test reward components distribution
    test_reward_components = [d["reward_components"] for d in final_test_dicts]
    plot_reward_components_distribution(test_reward_components, "atomic10x test")

    
    final_test_dicts = process_dicts_for_segment(final_atomic2020_test_dicts, "test")
    test_save_file = os.path.join(args.output_dir, "test.jsonl")
    logging.info(f"Saving {len(final_test_dicts)} test dicts to {test_save_file}")
    save_in_jsonl(final_test_dicts, test_save_file)
    # Plot the test reward components distribution
    test_reward_components = [d["reward_components"] for d in final_test_dicts]
    plot_reward_components_distribution(test_reward_components, "test")

    final_val_dicts = process_dicts_for_segment(val_dicts, "val")
    val_save_file = os.path.join(args.output_dir, "val.jsonl")
    logging.info(f"Saving {len(final_val_dicts)} val dicts to {val_save_file}")
    save_in_jsonl(final_val_dicts, val_save_file)
    # Plot the val reward components distribution
    val_reward_components = [d["reward_components"] for d in final_val_dicts]
    plot_reward_components_distribution(val_reward_components, "val")

    final_train_dicts = process_dicts_for_segment(train_dicts, "train")
    train_save_file = os.path.join(args.output_dir, "train.jsonl")
    logging.info(f"Saving {len(final_train_dicts)} train dicts to {train_save_file}")
    save_in_jsonl(final_train_dicts, train_save_file)
    # Plot the train reward components distribution
    train_reward_components = [d["reward_components"] for d in final_train_dicts]
    plot_reward_components_distribution(train_reward_components, "train")



if __name__ == '__main__':
    main()
    