# We will batch preprocess Reddit Comment Scores and add rewards to it

from utils.utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, save_list_of_tuples_to_tsv, load_from_tsv_to_list_of_list, save_in_jsonl, load_from_jsonl, get_ngrams_from_sentence, remove_multiple_space, url_regex, replace_urls
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="directory Reddit Comment Scores data", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all preprocessed language generation data and rewards in jsonl files", type=str, required=True)
parser.add_argument("-pd", "--plot_violin_distribution", help="Plot violin distribution for reward components", action="store_true")
parser.add_argument("-m", "--model_name", help="Name of the model to be saved", type=str, required=True)

parser.add_argument("-cm", "--cola_classifier_model", help="Name of the CoLA classifier model to use for fluency metric", type=str, default=None)
parser.add_argument("-dcm", "--depth_dialogRPT_model", help="Name of the DialogRPT model that predicts if the response is engaging given previous utterance as context", type=str, default=None)
parser.add_argument("-ucm", "--updown_dialogRPT_model", help="Name of the DialogRPT model that predicts if the response relative up-down ratio", type=str, default=None)
parser.add_argument("-om", "--offensive_model_dir", help="Directory containing saved ToxiChat DGPT offensive classifier", type=str, default=None)




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
from utils.rl_utils import task_to_class_mapping, general_batch_prompt_and_reward_generator

from sentence_transformers import SentenceTransformer, util

# To supress truncation warning. Ref: https://github.com/huggingface/transformers/issues/14285#issuecomment-968839045
import transformers
transformers.logging.set_verbosity_error()


import matplotlib.pyplot as plt
import seaborn as sns

# OLD and not needed now
def get_dgpt_regression_model_data(comments_positive_df, comments_negative_df):
    
    # Calculate 10 percentile of positive and negative comments "score"
    positive_comments_score_10_percentile = [comments_positive_df["score"].quantile(el/2) for el in range(0, 3)]
    negative_comments_score_10_percentile = [comments_negative_df["score"].quantile(el/2) for el in range(0, 3)]
    logging.info(f"Positive comments score 10 percentile: {positive_comments_score_10_percentile}")
    logging.info(f"Negative comments score 10 percentile: {negative_comments_score_10_percentile}")
    
    # Prepare comments of the format "id", "sentence", "attr_score", "domain"
    # Initialize roberta tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def get_quantile_for_score(score, quantiles):
        score_quantile = 0
        for i, el in enumerate(quantiles):
            if score >= el:
                score_quantile = i+1
        if score_quantile == 3:
            score_quantile = 2
        # if score_quantile == 0:
        #     score_quantile = 1
        return float(score_quantile)

    def dgpt_classifier_data(comments_df, quantiles, comment_type):
        
        dgpt_classifier_data = list()
        mismatched_count = 0
        nan_text = 0
        too_big = 0
        pbar = tqdm(comments_df.iterrows(), desc=f"Converting {comment_type} comments to roberta classifier data")
        for index, row in pbar:
            # Each row contains the keys:
            # Index(['id', 'parent_id', 'subreddit_id', 'link_id', 'text', 'score', 'ups',
                    # 'author', 'controversiality', 'parent_link_id', 'parent_text',
                    # 'parent_score', 'parent_ups', 'parent_author',
                    # 'parent_controversiality'],
                    # dtype='object')
            parent_text = row["parent_text"]
            text = row["text"]
            if type(text) == float and math.isnan(text):
                nan_text += 1
                continue
            if type(parent_text) == float and math.isnan(parent_text):
                nan_text += 1
                continue
            n_parent_words = len(parent_text.split())
            n_words = len(text.split())
            if n_parent_words > 100 or n_words > 100:
                too_big += 1
                continue
            score = row["score"]
            parent_score = row["parent_score"]
            dgpt_input = tokenizer(parent_text + tokenizer.eos_token + text, return_tensors="pt")
            full_sentence = tokenizer.decode(dgpt_input["input_ids"][0], skip_special_tokens=False)
            if len(dgpt_input["input_ids"][0]) > 300:
                too_big += 1
                continue
            # retokenize full sentence as in the ridge regression classifier script
            all_model_inputs_tokenized = tokenizer.batch_encode_plus([full_sentence], padding=True, add_special_tokens=False, return_tensors="pt")
            if all_model_inputs_tokenized["input_ids"].shape != dgpt_input["input_ids"].shape or not (all_model_inputs_tokenized["input_ids"] == dgpt_input["input_ids"]).all().item():
                # Keep track of these
                mismatched_count += 1
            # tokenizer.convert_ids_to_tokens(all_model_inputs_tokenized["input_ids"][0])
            # tokenizer.convert_ids_to_tokens(dgpt_input["input_ids"][0])
            # Convert score to quantile based on the positive comments score 10 percentile
            # score_quantile = get_quantile_for_score(score, quantiles)
            # if comment_type == "negative":
            #     score_quantile = - 3 + score_quantile
            # [NEW] Only binary lables
            score_quantile = 1.0 if comment_type == "positive" else -1.0
            # Assert that score and score_quantile are of the same sign
            assert(score_quantile * score > 0), breakpoint()
            final_dict = {"id": row["id"], "sentence": full_sentence, "attr_score": score_quantile, "domain": "reddit", "original_score": score, "parent_score": parent_score}
            pbar.set_postfix({"Mismatched": mismatched_count, "Nan text": nan_text, "Too big": too_big})
            dgpt_classifier_data.append(final_dict)
        return dgpt_classifier_data
    # We only need to do this for a sample for data that will be used for robert regressor training
    sample_size = 30000
    comments_positive_df_sample = comments_positive_df.sample(sample_size, random_state=RANDOM_SEED)
    sampled_postive_comment_classifier_data = dgpt_classifier_data(comments_positive_df_sample, positive_comments_score_10_percentile, "positive")
    comments_negative_df_sample = comments_negative_df.sample(sample_size, random_state=RANDOM_SEED)
    sampled_negative_comment_classifier_data = dgpt_classifier_data(comments_negative_df_sample, negative_comments_score_10_percentile, "negative")
    # Split the sampled data into train val and test
    # 2500 for val and test
    random.shuffle(sampled_postive_comment_classifier_data)
    random.shuffle(sampled_negative_comment_classifier_data)
    classifier_test_data = sampled_postive_comment_classifier_data[:2500] + sampled_negative_comment_classifier_data[:2500]
    classifier_val_data = sampled_postive_comment_classifier_data[2500:5000] + sampled_negative_comment_classifier_data[2500:5000]
    classifier_train_data = sampled_postive_comment_classifier_data[5000:] + sampled_negative_comment_classifier_data[5000:]
    random.shuffle(classifier_train_data)
    save_dir = os.path.join(args.output_dir, "dgpt_classifier_data")
    make_dir_if_not_exists(save_dir)
    train_save_file = os.path.join(save_dir, "train.jsonl")
    save_in_jsonl(classifier_train_data, train_save_file)
    logging.info(f"Saved {len(classifier_train_data)} instances to {train_save_file}")
    train_attr_score_counts = Counter([el["attr_score"] for el in classifier_train_data])
    logging.info(f"Train attr score counts: {train_attr_score_counts}")
    val_save_file = os.path.join(save_dir, "val.jsonl")
    save_in_jsonl(classifier_val_data, val_save_file)
    logging.info(f"Saved {len(classifier_val_data)} instances to {val_save_file}")
    val_attr_score_counts = Counter([el["attr_score"] for el in classifier_val_data])
    logging.info(f"Val attr score counts: {val_attr_score_counts}")
    test_save_file = os.path.join(save_dir, "test.jsonl")
    save_in_jsonl(classifier_test_data, test_save_file)
    logging.info(f"Saved {len(classifier_test_data)} instances to {test_save_file}")
    test_attr_score_counts = Counter([el["attr_score"] for el in classifier_test_data])
    logging.info(f"Test attr score counts: {test_attr_score_counts}")


from utils.comet_utils import get_comet_keras_input_and_labels
from utils.rl_utils import task_to_reward_fn_map, get_text_and_responses_from_task_prompts_and_references, get_batch_rewards, plot_reward_components_distribution
from utils.attributes_utils import GPT2ForOC_S_offensive

def main():
    # Load the positive and negative comments from the Reddit Comment Scores dataset
    start_time = time()
    comments_positive_csv = os.path.join(args.input_dir, "comments_positive.csv")
    comments_negative_csv = os.path.join(args.input_dir, "comments_negative.csv")
    comments_positive_df = pd.read_csv(comments_positive_csv)
    comments_negative_df = pd.read_csv(comments_negative_csv)
    logging.info(f"Loaded {len(comments_positive_df)} positive and {len(comments_negative_df)} negative comments from the Reddit Comment Scores dataset in {time() - start_time:.2f} seconds")
    # Filter out rows with nan values in "parent_text" or "text"
    count_parent_text_nan = comments_positive_df["parent_text"].isna().sum()
    count_text_nan = comments_positive_df["text"].isna().sum()
    logging.info(f"Positive comments: {count_parent_text_nan} parent_text and {count_text_nan} text rows have nan values")
    count_parent_text_nan = comments_negative_df["parent_text"].isna().sum()
    count_text_nan = comments_negative_df["text"].isna().sum()
    logging.info(f"Negative comments: {count_parent_text_nan} parent_text and {count_text_nan} text rows have nan values")
    logging.info(f"Shapes before filtering: {comments_positive_df.shape}, {comments_negative_df.shape}")
    comments_positive_df = comments_positive_df.dropna(subset=["parent_text", "text"])
    comments_negative_df = comments_negative_df.dropna(subset=["parent_text", "text"])
    logging.info(f"Shapes after filtering: {comments_positive_df.shape}, {comments_negative_df.shape}")

    # min, max, mean, median, std for positive and negative comments "score"
    logging.info(f"Positive comments score stats: {comments_positive_df['score'].describe()}")
    logging.info(f"Negative comments score stats: {comments_negative_df['score'].describe()}")
    # Violin plot of positive and negative comments "score"
    if args.plot_violin_distribution:
        # Set figure size
        plt.figure(figsize=(10, 9))
        violin_plot = sns.violinplot(data=comments_positive_df, x="score")
        violin_plot.set(xlabel="Positive comments score", ylabel="Score distribution")
        violin_plot.set_title(f"Positive comments score distribution, containing {len(comments_positive_df)} instances")
        violin_plot_save_file = os.path.join(args.output_dir, "positive_comments_score_distribution.png")
        # Tight layout
        plt.tight_layout()
        violin_plot.figure.savefig(violin_plot_save_file, dpi=300)
        logging.info(f"Saved violin plot to {violin_plot_save_file}")
        plt.clf()
        plt.cla()
        # Set figure size
        plt.figure(figsize=(10, 9))
        violin_plot = sns.violinplot(data=comments_negative_df, x="score")
        violin_plot.set(xlabel="Negative comments score", ylabel="Score distribution")
        violin_plot.set_title(f"Negative comments score distribution, containing {len(comments_negative_df)} instances")
        violin_plot_save_file = os.path.join(args.output_dir, "negative_comments_score_distribution.png")
        # Tight layout
        plt.tight_layout()
        violin_plot.figure.savefig(violin_plot_save_file, dpi=300)
        logging.info(f"Saved violin plot to {violin_plot_save_file}")
        plt.clf()
        plt.cla()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Extract dictionaries of text and responses for positive and negative comments
    def extract_correct_dictionaries_from_comments_df(comments_df):
        filtered_data = list()
        mismatched_count = 0
        too_big = 0
        url_text_count = 0
        url_parent_count = 0
        pbar = tqdm(comments_df.iterrows(), desc=f"Filtering long comments and parents")
        for index, row in pbar:
            # Each row contains the keys:
            # Index(['id', 'parent_id', 'subreddit_id', 'link_id', 'text', 'score', 'ups',
                    # 'author', 'controversiality', 'parent_link_id', 'parent_text',
                    # 'parent_score', 'parent_ups', 'parent_author',
                    # 'parent_controversiality'],
                    # dtype='object')
            parent_text = remove_multiple_space(row["parent_text"])
            text = remove_multiple_space(row["text"])
            n_parent_words = len(parent_text.split())
            n_words = len(text.split())
            if n_parent_words > 100 or n_words > 100:
                too_big += 1
                continue
            # Check for URLs present in text using url_regex
            if re.search(url_regex, text):
                url_text_count += 1
            if re.search(url_regex, parent_text):
                url_parent_count += 1
            parent_text, _ = replace_urls(parent_text)
            text, _ = replace_urls(text)

            score = row["score"]
            parent_score = row["parent_score"]
            try:
                dgpt_input = tokenizer(parent_text + tokenizer.eos_token + text + tokenizer.eos_token, return_tensors="pt")
            except TypeError:
                breakpoint()
            full_sentence = tokenizer.decode(dgpt_input["input_ids"][0], skip_special_tokens=False)
            if len(dgpt_input["input_ids"][0]) > 300:
                too_big += 1
                continue
            # retokenize full sentence as in the ridge regression classifier script
            all_model_inputs_tokenized = tokenizer.batch_encode_plus([full_sentence], padding=True, add_special_tokens=False, return_tensors="pt")
            if all_model_inputs_tokenized["input_ids"].shape != dgpt_input["input_ids"].shape or not (all_model_inputs_tokenized["input_ids"] == dgpt_input["input_ids"]).all().item():
                # Keep track of these
                mismatched_count += 1

            # prompts = [parent_text + tokenizer.eos_token]
            # responses = [text + tokenizer.eos_token]
            # tokenizer.padding_side = "left"
            # tokenizer.truncation_side = "left"
            # prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # tokenizer.padding_side = "right"
            # tokenizer.truncation_side = "right"
            # response_inputs = tokenizer(text_target=responses, return_tensors="pt", padding=True, truncation=True, max_length=128)
            # tokenizer.convert_ids_to_tokens(prompt_inputs["input_ids"][0])
            # tokenizer.convert_ids_to_tokens(response_inputs["input_ids"][0])
            # breakpoint()
            # tokenizer.convert_ids_to_tokens(all_model_inputs_tokenized["input_ids"][0])
            # tokenizer.convert_ids_to_tokens(dgpt_input["input_ids"][0])
            # Convert score to quantile based on the positive comments score 10 percentile
            # score_quantile = get_quantile_for_score(score, quantiles)
            # if comment_type == "negative":
            #     score_quantile = - 3 + score_quantile
            # [NEW] Only binary lables
            final_dict = {"id": row["id"], "prompt_or_input_text": parent_text + tokenizer.eos_token, "references": text + tokenizer.eos_token, "meta_data": {"score": score, "parent_score": parent_score}}
            pbar.set_postfix({"Mismatched": mismatched_count, "Too big": too_big, "parent url": url_parent_count, "text url": url_text_count})
            filtered_data.append(final_dict)
        return filtered_data
    
    # Split the comments into equal sized train, val and test sets
    # train sample size = 500000, val and test sample size = 1000 each
    sample_size = 102000
    # Debugging with smaller sample
    # sample_size = 10000
    comments_positive_df_sample = comments_positive_df.sample(sample_size, random_state=RANDOM_SEED)
    comments_negative_df_sample = comments_negative_df.sample(sample_size, random_state=RANDOM_SEED)
    filtered_positive_dicts = extract_correct_dictionaries_from_comments_df(comments_positive_df_sample)
    logging.info(f"Total positive comments after filtering: {len(filtered_positive_dicts)}/{len(comments_positive_df_sample)}")
    filtered_negative_dicts = extract_correct_dictionaries_from_comments_df(comments_negative_df_sample)
    logging.info(f"Total negative comments after filtering: {len(filtered_negative_dicts)}/{len(comments_negative_df_sample)}")

    # Split the filtered data into train val and test
    test_positive_dicts = filtered_positive_dicts[:1000]
    val_positive_dicts = filtered_positive_dicts[1000:2000]
    train_positive_dicts = filtered_positive_dicts[2000:]
    logging.info(f"Final positive comments: train = {len(train_positive_dicts)}, val = {len(val_positive_dicts)}, test = {len(test_positive_dicts)}")
    test_negative_dicts = filtered_negative_dicts[:1000]
    val_negative_dicts = filtered_negative_dicts[1000:2000]
    train_negative_dicts = filtered_negative_dicts[2000:]
    logging.info(f"Final negative comments: train = {len(train_negative_dicts)}, val = {len(val_negative_dicts)}, test = {len(test_negative_dicts)}")
    
    joint_val_set = val_positive_dicts + val_negative_dicts
    logging.info(f"Joint final val set: {len(joint_val_set)}")
    joint_test_set = test_positive_dicts + test_negative_dicts
    logging.info(f"Joint final test set: {len(joint_test_set)}")

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
    if args.updown_dialogRPT_model is not None:
        start_time = time()
        logging.info(f"Loading the microsoft/DialogRPT-updown model from {args.updown_dialogRPT_model}")
        updown_dialogRPT_tokenizer = AutoTokenizer.from_pretrained(args.updown_dialogRPT_model)
        updown_dialogRPT_model = AutoModelForSequenceClassification.from_pretrained(args.updown_dialogRPT_model).to(device)
        updown_dialogRPT_model.eval()
        logging.info(f"Loaded the microsoft/DialogRPT-updown model in {time() - start_time:.2f} seconds")
        reward_args["updown_dialogRPT_model_name"] = args.updown_dialogRPT_model
        reward_args["updown_dialogRPT_tokenizer"] = updown_dialogRPT_tokenizer
        reward_args["updown_dialogRPT_model"] = updown_dialogRPT_model
    if args.offensive_model_dir is not None:
        start_time = time()
        offensive_model = GPT2ForOC_S_offensive.from_pretrained(args.offensive_model_dir).half().to(device)
        offensive_tokenizer = AutoTokenizer.from_pretrained(args.offensive_model_dir)
        logging.info(f"Loaded offensive model from {args.offensive_model_dir} in {time() - start_time:.2f} seconds")
        reward_args["offensive_model"] = offensive_model
        reward_args["offensive_tokenizer"] = offensive_tokenizer

    # Prepare TFiDF reward models from positive and negative train and save them
    logging.info(f"Precomputing tf-idf scores from {len(train_positive_dicts)} positive and {len(train_negative_dicts)} negative train comments")
    start_time = time()
    train_responses = [d["references"] for d in train_positive_dicts + train_negative_dicts]
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
    
    # Test loading them and checking if they are the same
    # loaded_tfidf_transformer = load_from_pickle(tfidf_transformer_save_file)
    # loaded_cv = load_from_pickle(cv_save_file)
    # test_sentence = "This was a really delicious meal with scrumptious food! Very tasty"
    # test_sentence_vector = loaded_cv.transform([test_sentence])
    # test_sentence_tfidf_vector = loaded_tfidf_transformer.transform(test_sentence_vector).data
    # test_sentence_tfidf_vector_2 = tfidf_transformer.transform(test_sentence_vector).data
    # tfidf_means = [np.mean(tfidf_vector[i].data)
    # assert(np.allclose(test_sentence_tfidf_vector, test_sentence_tfidf_vector_2)), breakpoint()
    # breakpoint()


    
    logging.info(f"Saving tfidf_transformer to {tfidf_transformer_save_file}")
    save_in_pickle(tfidf_transformer, tfidf_transformer_save_file)
    logging.info(f"Saving cv to {cv_save_file}")
    save_in_pickle(cv, cv_save_file)
    
    
    daily_dialog_extra_args = {"tfidf_transformer": tfidf_transformer, "cv": cv}
    reward_args["cv"] = cv
    reward_args["tfidf_transformer"] = tfidf_transformer
    logging.info(f"Precomputed tf-idf scores in {time() - start_time:.2f} seconds")

    def compute_rewards_for_reddit_comments_data(task_name, comments_data, reward_args, segment_name):
        batch_size = args.batch_size
        reward_component_to_fn_map = task_to_reward_fn_map[task_name]
        all_rewarded_data = list()
        # eos_token = "<|endoftext|>"
        total_correct_updown_predictions = 0
        for i in tqdm(range(0, len(comments_data), batch_size), desc=f"Computing rewards for {segment_name} split"):
            batch_comments_data = comments_data[i:i+batch_size]
            prompt_or_input_text_list = [data["prompt_or_input_text"] for data in batch_comments_data]
            batch_responses = [data["references"] for data in batch_comments_data]
            batch_texts, batch_responses = get_text_and_responses_from_task_prompts_and_references(task_name, prompt_or_input_text_list, batch_responses)
            batch_reward_components = get_batch_rewards(batch_texts, batch_responses, reward_args, reward_component_to_fn_map)
            # Check accuracy of updown reward with comment score
            updown_rewards = np.array([el["updown"] for el in batch_reward_components])
            comment_scores = np.array([el["meta_data"]['score'] for el in batch_comments_data])
            # check if comment score sign matches with updown > 0.5 or not
            comment_rewards_sign = np.sign(comment_scores)
            updown_rewards_sign = np.array([1 if el >= 0.5 else -1 for el in updown_rewards])
            count_correct = np.sum(comment_rewards_sign == updown_rewards_sign)
            total_correct_updown_predictions += count_correct

            # Simply append the reward components to the data dicts
            final_data_dicts = [data_dict | {"reward_components": reward_components} for data_dict, reward_components in zip(batch_comments_data, batch_reward_components)]
            all_rewarded_data.extend(final_data_dicts)
        logging.info(f"Total correct updown predictions for {segment_name} split = {total_correct_updown_predictions}/{len(comments_data)} = {total_correct_updown_predictions/len(comments_data)*100.0:.2f}%")
        return all_rewarded_data
    
    
    task_name = "reddit_pos"
    joint_test_rewarded = compute_rewards_for_reddit_comments_data(task_name, joint_test_set, reward_args, "joint test")
    test_save_file = os.path.join(args.output_dir, "test.jsonl")
    logging.info(f"Saving {len(joint_test_rewarded)} test dicts to {test_save_file}")
    save_in_jsonl(joint_test_rewarded, test_save_file)
    # Plot the test reward components distribution
    test_reward_components = [d["reward_components"] for d in joint_test_rewarded]
    plot_reward_components_distribution(test_reward_components, "test", args)

    joint_val_rewarded = compute_rewards_for_reddit_comments_data(task_name, joint_val_set, reward_args, "joint val")
    val_save_file = os.path.join(args.output_dir, "val.jsonl")
    logging.info(f"Saving {len(joint_val_rewarded)} val dicts to {val_save_file}")
    save_in_jsonl(joint_val_rewarded, val_save_file)
    # Plot the val reward components distribution
    val_reward_components = [d["reward_components"] for d in joint_val_rewarded]
    plot_reward_components_distribution(val_reward_components, "val", args)

    positive_train_rewarded = compute_rewards_for_reddit_comments_data(task_name, train_positive_dicts, reward_args, "positive train")
    pos_train_save_file = os.path.join(args.output_dir, "pos_train.jsonl")
    logging.info(f"Saving {len(positive_train_rewarded)} train dicts to {pos_train_save_file}")
    save_in_jsonl(positive_train_rewarded, pos_train_save_file)
    # Plot the train reward components distribution
    train_reward_components = [d["reward_components"] for d in positive_train_rewarded]
    plot_reward_components_distribution(train_reward_components, "pos_train", args)

    negative_train_rewarded = compute_rewards_for_reddit_comments_data(task_name, train_negative_dicts, reward_args, "negative train")
    neg_train_save_file = os.path.join(args.output_dir, "neg_train.jsonl")
    logging.info(f"Saving {len(negative_train_rewarded)} train dicts to {neg_train_save_file}")
    save_in_jsonl(negative_train_rewarded, neg_train_save_file)
    # Plot the train reward components distribution
    train_reward_components = [d["reward_components"] for d in negative_train_rewarded]
    plot_reward_components_distribution(train_reward_components, "neg_train", args)


if __name__ == '__main__':
    main()
    