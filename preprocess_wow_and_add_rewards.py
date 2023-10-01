# We will batch preprocess Faithdial and Wizards of Wikipedia and add rewards to it

from utils.utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, save_list_of_tuples_to_tsv, load_from_tsv_to_list_of_list, save_in_jsonl, load_from_jsonl, get_ngrams_from_sentence
import pdb

from transformers import AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, RobertaPreTrainedModel, AutoModelForSequenceClassification, pipeline
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
from copy import deepcopy
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_task", help="Name of the task", type=str, required=True, choices=["faithdial", "wow", "faithdial_wow"])
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all preprocessed language generation data and rewards in jsonl files", type=str, required=True)
parser.add_argument("-pd", "--plot_violin_distribution", help="Plot violin distribution for reward components", action="store_true")
parser.add_argument("-m", "--model_name", help="Name of the model to be used for tokenization", type=str, required=True)
parser.add_argument("-cm", "--cola_classifier_model", help="Name of the CoLA classifier model to use for fluency metric", type=str, default=None)
parser.add_argument("-fcm", "--faithdial_critic_model", help="Name of the FaithDial Critic model to check the faithfulness of the Knowledge-Grounded Dialog responses.", type=str, default=None)
parser.add_argument("-dcm", "--depth_dialogRPT_model", help="Name of the DialogRPT model that predicts if the response is engaging given previous utterance as context", type=str, default=None)

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

from utils.attributes_utils import get_cola_fluency_score
from utils.rl_utils import task_to_reward_fn_map, get_text_and_responses_from_task_prompts_and_references, get_batch_rewards, plot_reward_components_distribution

from sentence_transformers import SentenceTransformer, util

# To supress truncation warning. Ref: https://github.com/huggingface/transformers/issues/14285#issuecomment-968839045
import transformers
transformers.logging.set_verbosity_error()


import matplotlib.pyplot as plt
import seaborn as sns

from utils.comet_utils import get_comet_keras_input_and_labels

def extract_wow_dialogs(data):
    dialog_idx = 0
    count_no_passages_used = 0
    final_data_dicts = list()
    for dialog_data in data:
        # dialog is dictionary with dict_keys(['chosen_topic', 'persona', 'wizard_eval', 'dialog', 'chosen_topic_passage'])
        dialog = dialog_data["dialog"]
        # dialog is a list of utterances
        history = list()
        for utterance in dialog:
            # utterance is a dict with dict_keys(['speaker', 'text', 'checked_sentence', 'checked_passage', 'retrieved_passages', 'retrieved_topics'])
            speaker = utterance["speaker"]
            text = utterance["text"]
            if "Wizard" in speaker:
                response = text
                if len(utterance["checked_sentence"]) == 0:
                    continue
                checked_sentence = list(utterance["checked_sentence"].values())[0]
                knowledge = checked_sentence
                if knowledge == "no_passages_used":
                    count_no_passages_used += 1
                if len(history) != 0 and knowledge != "no_passages_used":
                    final_data_dicts.append({"dialog_idx": dialog_idx, "response": response, "history": deepcopy(history), "knowledge": knowledge})
                    dialog_idx += 1
                history.append(response)
            elif "Apprentice" in speaker:
                history.append(text)
            else:
                logging.warning(f"Unknown speaker = {speaker}")
                breakpoint()
    logging.info(f"Total no passages used = {count_no_passages_used}")
    return final_data_dicts

def compute_rewards_for_wizard_utterances(task_name, wizard_utterances, reward_args, segment_name):
    batch_size = args.batch_size
    tokenizer = reward_args["tokenizer"]
    reward_component_to_fn_map = task_to_reward_fn_map[task_name]
    all_rewarded_data = list()
    eos_token = "<|endoftext|>"
    total_reduced_history = 0
    for i in tqdm(range(0, len(wizard_utterances), batch_size), desc=f"Computing rewards for {segment_name} split"):
        # batch_wizard_utterances = wizard_utterances[i:i+batch_size]
        batch_wizard_utterances = [wizard_utterances[j] for j in range(i, min(i+batch_size, len(wizard_utterances)))]
        id_list = [wizard_utterance["dialog_idx"] for wizard_utterance in batch_wizard_utterances]
        batch_knowledge = [wizard_utterance["knowledge"].strip() for wizard_utterance in batch_wizard_utterances]
        batch_history = [eos_token.join(wizard_utterance["history"]).strip() for wizard_utterance in batch_wizard_utterances]
        # Compress history such that it is less than 400 tokens
        initial_batch_history = deepcopy(batch_history)
        initial_history_tokenized = tokenizer(initial_batch_history, padding=True, return_tensors="pt")
        initial_history_input_ids = initial_history_tokenized["input_ids"]
        initial_history_mask = initial_history_tokenized["attention_mask"]
        history_tokenized = tokenizer(batch_history, padding=True, return_tensors="pt")
        history_input_ids = history_tokenized["input_ids"]
        count_n_reduced = None
        while history_input_ids.size(1) > 400:
            history_mask = history_tokenized["attention_mask"]
            batch_seq_len = history_mask.sum(dim=1)
            if count_n_reduced is None:
                count_n_reduced = sum(batch_seq_len > 400).item()
            # Remove first utterance for sequences that are longer than 400 tokens
            batch_history = [history.split(eos_token, 1)[1] if seq_len > 400 else history for history, seq_len in zip(batch_history, batch_seq_len)]
            history_tokenized = tokenizer(batch_history, padding=True, return_tensors="pt")
            history_input_ids = history_tokenized["input_ids"]
        if count_n_reduced is not None:
            total_reduced_history += count_n_reduced
        batch_references = [wizard_utterance["response"].strip() + eos_token for wizard_utterance in batch_wizard_utterances]
        prompt_or_input_text_list = [f"Knowledge: {knowledge}{eos_token}{history}{eos_token}" for knowledge, history in zip(batch_knowledge, batch_history)]
        batch_texts, batch_responses = get_text_and_responses_from_task_prompts_and_references(task_name, prompt_or_input_text_list, batch_references)
        batch_reward_components = get_batch_rewards(batch_texts, batch_responses, reward_args, reward_component_to_fn_map)
        final_data_dicts = [{"id": id, "prompt_or_input_text": prompt_or_input_text, "references": references, "meta_data": {"knowledge": knowledge, "history": history}, "reward_components": reward_components} for id, prompt_or_input_text, knowledge, history, references, reward_components in zip(id_list, prompt_or_input_text_list, batch_knowledge, batch_history, batch_references, batch_reward_components)]
        all_rewarded_data.extend(final_data_dicts)
    logging.info(f"Total reduced history = {total_reduced_history}/{len(wizard_utterances)}")
    return all_rewarded_data

def convert_faithdial_dataset_to_list(faithdial_dataset):
    all_data_dicts = list()
    for i in range(len(faithdial_dataset)):
        data_dict = faithdial_dataset[i]
        data_dict['dialog_idx'] = i
        all_data_dicts.append(data_dict)
    return all_data_dicts

def main():
    args.device = device
    # Load the FaithDail dataset
    from datasets import load_dataset
    dataset = load_dataset("McGill-NLP/FaithDial")
    # dict_keys(['test', 'test_random_split', 'test_topic_split', 'train', 'validation', 'valid_random_split', 'valid_topic_split'])
    faithdial_train_data = dataset["train"]
    faithdial_val_data = dataset["validation"]
    faithdial_test_data = dataset["test"]
    # Convert all faithdial datasets to list while fixing the ids
    faithdial_train_data = convert_faithdial_dataset_to_list(faithdial_train_data)
    faithdial_val_data = convert_faithdial_dataset_to_list(faithdial_val_data)
    faithdial_test_data = convert_faithdial_dataset_to_list(faithdial_test_data)
    logging.info(f"train size = {len(faithdial_train_data)}, val size = {len(faithdial_val_data)}, test size = {len(faithdial_test_data)}")
    # train size = 18357, val size = 3417, test size = 3539
    # Each datapoint contains dict_keys(['dialog_idx', 'response', 'original_response', 'history', 'knowledge', 'BEGIN', 'VRM'])
    # {'dialog_idx': 0, 'response': 'Yeah, but once the access to the internet was a rare thing. do you remember?', 'original_response': "No I could not! I couldn't imagine living when internet access was rare and very few people had it!", 'history': ['Can you imagine the world without internet access?'], 'knowledge': 'Internet access was once rare, but has grown rapidly.', 'BEGIN': ['Hallucination'], 'VRM': ['Disclosure', 'Ack.']}
    
    # Load the wow dataset from input_dir
    wow_dir = "data/wow"
    start_time = time()
    train_file = os.path.join(wow_dir, "train.json")
    train_data = load_from_json(train_file)
    logging.info(f"Loaded {len(train_data)} instances from {train_file}")
    val_seen_file = os.path.join(wow_dir, "valid_random_split.json")
    val_seen_data = load_from_json(val_seen_file)
    logging.info(f"Loaded {len(val_seen_data)} instances from {val_seen_file}")
    val_unseen_file = os.path.join(wow_dir, "valid_topic_split.json")
    val_unseen_data = load_from_json(val_unseen_file)
    logging.info(f"Loaded {len(val_unseen_data)} instances from {val_unseen_file}")
    test_seen_file = os.path.join(wow_dir, "test_random_split.json")
    test_seen_data = load_from_json(test_seen_file)
    logging.info(f"Loaded {len(test_seen_data)} instances from {test_seen_file}")
    test_unseen_file = os.path.join(wow_dir, "test_topic_split.json")
    test_unseen_data = load_from_json(test_unseen_file)
    logging.info(f"Loaded {len(test_unseen_data)} instances from {test_unseen_file}")
    logging.info(f"Loaded wow dataset in {time() - start_time} seconds")

    # Extract the dialogs from the wow dataset
    train_wizard_utterances = extract_wow_dialogs(train_data)
    logging.info(f"Extracted {len(train_wizard_utterances)} wizard utterances from {len(train_data)} instances in train split")
    val_seen_wizard_utterances = extract_wow_dialogs(val_seen_data)
    logging.info(f"Extracted {len(val_seen_wizard_utterances)} wizard utterances from {len(val_seen_data)} instances in val seen split")
    val_unseen_wizard_utterances = extract_wow_dialogs(val_unseen_data)
    logging.info(f"Extracted {len(val_unseen_wizard_utterances)} wizard utterances from {len(val_unseen_data)} instances in val unseen split")
    test_seen_wizard_utterances = extract_wow_dialogs(test_seen_data)
    logging.info(f"Extracted {len(test_seen_wizard_utterances)} wizard utterances from {len(test_seen_data)} instances in test seen split")
    test_unseen_wizard_utterances = extract_wow_dialogs(test_unseen_data)
    logging.info(f"Extracted {len(test_unseen_wizard_utterances)} wizard utterances from {len(test_unseen_data)} instances in test unseen split")

    # Load the reward models
    reward_args = defaultdict(lambda: None)
    reward_args["device"] = device
    # Load the CoLA model as pipeline
    if args.cola_classifier_model is not None:
        start_time = time()
        logging.info(f"Loading the CoLA model pipeline from {args.cola_classifier_model}")
        cola_pipeline = pipeline("sentiment-analysis",model=args.cola_classifier_model, device=0, batch_size=args.batch_size)
        logging.info(f"Loaded the CoLA model pipeline in {time() - start_time:.2f} seconds")
        reward_args["cola_classifier_model"] = args.cola_classifier_model
        reward_args["cola_pipeline"] = cola_pipeline
    # Load the FaithDial Critic model
    if args.faithdial_critic_model is not None:
        start_time = time()
        logging.info(f"Loading the FaithDial Critic model from {args.faithdial_critic_model}")
        faithdial_tokenizer = AutoTokenizer.from_pretrained(args.faithdial_critic_model)
        faithdial_critic_model = AutoModelForSequenceClassification.from_pretrained(args.faithdial_critic_model).to(device)
        faithdial_critic_model.eval()
        logging.info(f"Loaded the FaithDial Critic model in {time() - start_time:.2f} seconds")
        reward_args["faithdial_critic_model_name"] = args.faithdial_critic_model
        reward_args["faithdial_tokenizer"] = faithdial_tokenizer
        reward_args["faithdial_critic_model"] = faithdial_critic_model
    if args.depth_dialogRPT_model is not None:
        start_time = time()
        logging.info(f"Loading the microsoft/DialogRPT-depth model from {args.depth_dialogRPT_model}")
        depth_dialogRPT_tokenizer = AutoTokenizer.from_pretrained(args.depth_dialogRPT_model)
        depth_dialogRPT_model = AutoModelForSequenceClassification.from_pretrained(args.depth_dialogRPT_model).to(device)
        depth_dialogRPT_model.eval()
        logging.info(f"Loaded the microsoft/DialogRPT-depth model in {time() - start_time:.2f} seconds")
        reward_args["depth_dialogRPT_model_name"] = args.depth_dialogRPT_model
        reward_args["depth_dialogRPT_tokenizer"] = depth_dialogRPT_tokenizer
        reward_args["depth_dialogRPT_model"] = depth_dialogRPT_model
    # Prepare TFiDF reward models from train responses and save them
    logging.info(f"Precomputing tf-idf scores from {len(train_wizard_utterances)} train wizard utterances")
    start_time = time()
    train_responses = [d["response"] for d in train_wizard_utterances]
    from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
    cv=CountVectorizer() 
    # this steps generates word counts for the words in your docs 
    word_count_vector=cv.fit_transform(train_responses)
    # word_count_vector.shape = (35781, 12255)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # Save the tfidf_transformer and cv
    tfidf_transformer_save_file = os.path.join(args.output_dir, "tfidf_transformer.pkl")
    cv_save_file = os.path.join(args.output_dir, "cv.pkl")
    logging.info(f"Saving tfidf_transformer to {tfidf_transformer_save_file}")
    save_in_pickle(tfidf_transformer, tfidf_transformer_save_file)
    logging.info(f"Saving cv to {cv_save_file}")
    save_in_pickle(cv, cv_save_file)
    reward_args["cv"] = cv
    reward_args["tfidf_transformer"] = tfidf_transformer
    logging.info(f"Precomputed tf-idf scores in {time() - start_time:.2f} seconds")
    


    # Load the tokenizer to compress the history
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Loaded the tokenizer from {args.model_name}")
    reward_args["tokenizer"] = tokenizer

    def preprocess_task_segment(wizard_utterances, segment_name):
        task_name = "WOW"
        # Convert utterance data into prompt and response pairs for reward computations
        wizard_rewarded = compute_rewards_for_wizard_utterances(task_name, wizard_utterances, reward_args, segment_name)
        logging.info(f"Computed rewards for {len(wizard_rewarded)} wizard utterances in train split")
        save_file = os.path.join(args.output_dir, f"{segment_name}.jsonl")
        logging.info(f"Saving {len(wizard_rewarded)} {segment_name} dicts to {save_file}")
        save_in_jsonl(wizard_rewarded, save_file)
        # Plot the train reward components distribution
        reward_components = [d["reward_components"] for d in wizard_rewarded]
        plot_reward_components_distribution(reward_components, segment_name, args)

    if args.input_task == "wow":
        preprocess_task_segment(train_wizard_utterances, "train")
        preprocess_task_segment(val_unseen_wizard_utterances, "val")
        preprocess_task_segment(faithdial_test_data, "faithdial_test")
    elif args.input_task == "faithdial":
        preprocess_task_segment(faithdial_train_data, "train")
        preprocess_task_segment(faithdial_val_data, "val")
        preprocess_task_segment(faithdial_test_data, "test")
    elif args.input_task == "faithdial_wow":
        # Combine train and val splits of faithdial and wow
        n_wizard_dialogs = len(train_wizard_utterances)
        faithdial_train_data_list = list()
        for i, d in enumerate(faithdial_train_data):
            d["dialog_id"] = i + 1 + n_wizard_dialogs
            faithdial_train_data_list.append(d)
        train_utterances = train_wizard_utterances + faithdial_train_data_list
        random.shuffle(train_utterances)
        logging.info(f"New merged train data (WOW) {len(train_wizard_utterances)} and (FaithDial) {len(faithdial_train_data_list)} = {len(train_utterances)}")
        preprocess_task_segment(train_utterances, "train")
        n_val_wizard_dialogs = len(val_unseen_wizard_utterances)
        faithdial_val_data_list = list()
        for i, d in enumerate(faithdial_val_data):
            d["dialog_id"] = i + 1 + n_val_wizard_dialogs
            faithdial_val_data_list.append(d)
        logging.info(f"New merged val data (FaithDial) {len(faithdial_val_data_list)} and (WOW) {len(val_unseen_wizard_utterances)} = {len(faithdial_val_data_list) + len(val_unseen_wizard_utterances)}")
        preprocess_task_segment(val_unseen_wizard_utterances + faithdial_val_data_list, "val")
        preprocess_task_segment(faithdial_test_data, "test")
    else:
        logging.error(f"Unknown input task {args.input_task}")
        breakpoint()

    """
    task_name = "WOW"
    # Convert utterance data into prompt and response pairs for reward computations
    train_wizard_rewarded = compute_rewards_for_wizard_utterances(task_name, train_wizard_utterances, reward_args, "train")
    logging.info(f"Computed rewards for {len(train_wizard_rewarded)} wizard utterances in train split")
    train_save_file = os.path.join(args.output_dir, "train.jsonl")
    logging.info(f"Saving {len(train_wizard_rewarded)} train dicts to {train_save_file}")
    save_in_jsonl(train_wizard_rewarded, train_save_file)
    # Plot the train reward components distribution
    train_reward_components = [d["reward_components"] for d in train_wizard_rewarded]
    plot_reward_components_distribution(train_reward_components, "train", args)


    # Also keep track of faithdail test set
    faithdial_test_rewarded = compute_rewards_for_wizard_utterances(task_name, faithdial_test_data, reward_args, "faithdial_test")
    logging.info(f"Computed rewards for {len(faithdial_test_rewarded)} wizard utterances in faithdial test split")
    faithdial_test_save_file = os.path.join(args.output_dir, "faithdial_test.jsonl")
    logging.info(f"Saving {len(faithdial_test_rewarded)} faithdial test dicts to {faithdial_test_save_file}")
    save_in_jsonl(faithdial_test_rewarded, faithdial_test_save_file)
    # Plot the faithdial test reward components distribution
    faithdial_test_reward_components = [d["reward_components"] for d in faithdial_test_rewarded]
    plot_reward_components_distribution(faithdial_test_reward_components, "faithdial_test", args)


    val_seen_wizard_rewarded = compute_rewards_for_wizard_utterances(task_name, val_seen_wizard_utterances, reward_args, "val_seen")
    logging.info(f"Computed rewards for {len(val_seen_wizard_rewarded)} wizard utterances in val seen split")
    val_seen_save_file = os.path.join(args.output_dir, "val_seen.jsonl")
    logging.info(f"Saving {len(val_seen_wizard_rewarded)} val seen dicts to {val_seen_save_file}")
    save_in_jsonl(val_seen_wizard_rewarded, val_seen_save_file)
    # Plot the val seen reward components distribution
    val_seen_reward_components = [d["reward_components"] for d in val_seen_wizard_rewarded]
    plot_reward_components_distribution(val_seen_reward_components, "val_seen", args)

    val_wizard_rewarded = compute_rewards_for_wizard_utterances(task_name, val_unseen_wizard_utterances, reward_args, "val")
    logging.info(f"Computed rewards for {len(val_wizard_rewarded)} wizard utterances in val unseen split")
    val_save_file = os.path.join(args.output_dir, "val.jsonl")
    logging.info(f"Saving {len(val_wizard_rewarded)} val dicts to {val_save_file}")
    save_in_jsonl(val_wizard_rewarded, val_save_file)
    # Plot the val unseen reward components distribution
    val_reward_components = [d["reward_components"] for d in val_wizard_rewarded]
    plot_reward_components_distribution(val_reward_components, "val", args)

    test_seen_wizard_rewarded = compute_rewards_for_wizard_utterances(task_name, test_seen_wizard_utterances, reward_args, "test_seen")
    logging.info(f"Computed rewards for {len(test_seen_wizard_rewarded)} wizard utterances in test seen split")
    test_seen_save_file = os.path.join(args.output_dir, "test_seen.jsonl")
    logging.info(f"Saving {len(test_seen_wizard_rewarded)} test seen dicts to {test_seen_save_file}")
    save_in_jsonl(test_seen_wizard_rewarded, test_seen_save_file)
    # Plot the test seen reward components distribution
    test_seen_reward_components = [d["reward_components"] for d in test_seen_wizard_rewarded]
    plot_reward_components_distribution(test_seen_reward_components, "test_seen", args)
    
    test_wizard_rewarded = compute_rewards_for_wizard_utterances(task_name, test_unseen_wizard_utterances, reward_args, "test")
    logging.info(f"Computed rewards for {len(test_wizard_rewarded)} wizard utterances in test unseen split")
    test_save_file = os.path.join(args.output_dir, "test.jsonl")
    logging.info(f"Saving {len(test_wizard_rewarded)} test dicts to {test_save_file}")
    save_in_jsonl(test_wizard_rewarded, test_save_file)
    # Plot the test unseen reward components distribution
    test_reward_components = [d["reward_components"] for d in test_wizard_rewarded]
    plot_reward_components_distribution(test_reward_components, "test", args)
    """


if __name__ == '__main__':
    main()
    