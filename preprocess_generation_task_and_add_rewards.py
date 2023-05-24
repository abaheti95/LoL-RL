# We will batch preprocess RL4LMs tasks and add rewards to them

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

# CommonGen, CNNDailyMail, IMDB, IMDBForSeq2Seq, WMT, WMT14PreprocessedEnDe, WMT16NewsOnlyDatasetEnDe, IWSLT2017EnDe, CRD3DialogueGeneration, DailyDialog 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_task", help="RL4LMs predefined data class", type=str, required=True, choices=["Xsum", "CNNDailyMail", "ToTTo", "IMDB", "IMDBForSeq2Seq", "IWSLT2017EnDe", "DailyDialog", "CommonGen", "FaithDial"])
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all preprocessed language generation data and rewards in jsonl files", type=str, required=True)

# Arguments regarding the reward models
parser.add_argument("-cm", "--cola_classifier_model", help="Name of the CoLA classifier model to use for fluency metric", type=str, default=None)
parser.add_argument("-esm", "--embedding_similarity_model", help="Name of the embedding similarity model to use for comparing semantic relevance", type=str, default=None)
parser.add_argument("-scm", "--sentiment_classification_model", help="Name of sentiment classifier model used for IMDB continuation", type=str, default=None)
parser.add_argument("-dnlim", "--doc_nli_model", help="Name of the document NLI model to check the summary entailment", type=str, default=None)

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
from utils.rl_utils import task_to_class_mapping, general_batch_prompt_and_reward_generator, plot_reward_components_distribution

from sentence_transformers import SentenceTransformer, util

# To supress truncation warning. Ref: https://github.com/huggingface/transformers/issues/14285#issuecomment-968839045
import transformers
transformers.logging.set_verbosity_error()


def main():
    # 1. Load the RL4LMs dataset
    logging.info(f"Loading the dataset for task = {args.input_task}")
    data_class = task_to_class_mapping[args.input_task]["class"]
    prepare_args = task_to_class_mapping[args.input_task].get("prepare_args", {})
    # Load the rl4lms version of the dataset
    train = data_class.prepare("train", **prepare_args)
    if args.input_task in ["Xsum"]:
        val = data_class.prepare("validation", **prepare_args)
    else:
        val = data_class.prepare("val", **prepare_args)
    test = data_class.prepare("test", **prepare_args)
    logging.info(f"Total train samples = {len(train)}")
    logging.info(f"Total validation samples = {len(val)}")
    logging.info(f"Total test samples = {len(test)}")

    prepare_args['task_name'] = args.input_task

    reward_args = defaultdict(lambda: None)
    reward_args["device"] = device
    # 1.1. For DailyDialog compute the tf-idf scores on all train responses
    if args.input_task == "DailyDialog":
        logging.info(f"Precomputing tf-idf scores for all train responses for task = {args.input_task}")
        train_responses = [sample[0].references[0].replace(" <EOU>", "").strip() for sample in train] 
        from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
        cv=CountVectorizer() 
        # this steps generates word counts for the words in your docs 
        word_count_vector=cv.fit_transform(train_responses)
        # word_count_vector.shape = (35781, 12255)
        tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(word_count_vector)
        daily_dialog_extra_args = {"tfidf_transformer": tfidf_transformer, "cv": cv}
        reward_args["cv"] = cv
        reward_args["tfidf_transformer"] = tfidf_transformer
    elif args.input_task == "CommonGen":
        import spacy
        nlp = spacy.load('en_core_web_sm')
        reward_args["nlp"] = nlp

    # 2. Load the reward models
    # Load the CoLA model as pipeline
    if args.cola_classifier_model is not None:
        start_time = time()
        logging.info(f"Loading the CoLA model pipeline from {args.cola_classifier_model}")
        cola_pipeline = pipeline("sentiment-analysis",model=args.cola_classifier_model, device=0, batch_size=args.batch_size)
        logging.info(f"Loaded the CoLA model pipeline in {time() - start_time:.2f} seconds")
        reward_args["cola_classifier_model"] = args.cola_classifier_model
        reward_args["cola_pipeline"] = cola_pipeline
    # Load the Sentiment Classification model as pipeline
    if args.sentiment_classification_model is not None:
        start_time = time()
        logging.info(f"Loading the Sentiment Classification model pipeline from {args.sentiment_classification_model}")
        sentiment_classification_pipeline = pipeline("sentiment-analysis",model=args.sentiment_classification_model, device=0, batch_size=args.batch_size)
        logging.info(f"Loaded the Sentiment Classification model pipeline in {time() - start_time:.2f} seconds")
        reward_args["sentiment_classification_model"] = args.sentiment_classification_model
        reward_args["sentiment_classification_pipeline"] = sentiment_classification_pipeline
    # Load the embedding similarity model
    if args.embedding_similarity_model is not None:
        start_time = time()
        logging.info(f"Loading the embedding similarity model from {args.embedding_similarity_model}")
        emb_sim_measure = SentenceTransformer(args.embedding_similarity_model, device=device)
        logging.info(f"Loaded the embedding similarity model in {time() - start_time:.2f} seconds")
        reward_args["embedding_similarity_model"] = args.embedding_similarity_model
        reward_args["emb_sim_measure"] = emb_sim_measure
    # Load the Document NLI model
    if args.doc_nli_model is not None:
        start_time = time()
        logging.info(f"Loading the Document NLI model from {args.doc_nli_model}")
        doc_nli_tokenizer = AutoTokenizer.from_pretrained(args.doc_nli_model)
        doc_nli_model = AutoModelForSequenceClassification.from_pretrained(args.doc_nli_model).to(device)
        logging.info(f"Loaded the Document NLI model in {time() - start_time:.2f} seconds")
        reward_args["doc_nli_model_name"] = args.doc_nli_model
        reward_args["doc_nli_tokenizer"] = doc_nli_tokenizer
        reward_args["doc_nli_model"] = doc_nli_model
    
    # 3. Create the prompt, target and rewards for the dataset
    def process_dicts_for_segment(segment, segment_name):
        # type(segment) = <class 'rl4lms.data_pools.custom_text_generation_pools.Xsum'>
        final_segment_dicts = list()
        for i in tqdm(range(0, len(segment), args.batch_size), desc=f"Computing reward components for {segment_name}"):
            # Each sample is a tuple of 2 elements
            # first element is a Sample object and second element is just 1.0 (probably unit reward?)
            # We will only keep track of Sample objects
            segment_batch = [segment[i][0] for i in range(i, min(i+args.batch_size, len(segment)))]
            # Change reward function based on input_task mapping
            new_segment_dicts = general_batch_prompt_and_reward_generator(segment_batch, reward_args, prepare_args)
            final_segment_dicts.extend(new_segment_dicts)
            # TEMP: For debugging
            # if i > 1000:
            #     break
        return final_segment_dicts
    final_test_dicts = process_dicts_for_segment(test, "test")
    test_save_file = os.path.join(args.output_dir, "test.jsonl")
    logging.info(f"Saving {len(final_test_dicts)} test dicts to {test_save_file}")
    save_in_jsonl(final_test_dicts, test_save_file)
    # Plot the test reward components distribution
    test_reward_components = [d["reward_components"] for d in final_test_dicts]
    plot_reward_components_distribution(test_reward_components, "test", args)

    final_val_dicts = process_dicts_for_segment(val, "val")
    val_save_file = os.path.join(args.output_dir, "val.jsonl")
    logging.info(f"Saving {len(final_val_dicts)} val dicts to {val_save_file}")
    save_in_jsonl(final_val_dicts, val_save_file)
    # Plot the val reward components distribution
    val_reward_components = [d["reward_components"] for d in final_val_dicts]
    plot_reward_components_distribution(val_reward_components, "val", args)

    final_train_dicts = process_dicts_for_segment(train, "train")
    train_save_file = os.path.join(args.output_dir, "train.jsonl")
    logging.info(f"Saving {len(final_train_dicts)} train dicts to {train_save_file}")
    save_in_jsonl(final_train_dicts, train_save_file)
    # Plot the train reward components distribution
    train_reward_components = [d["reward_components"] for d in final_train_dicts]
    plot_reward_components_distribution(train_reward_components, "train", args)



if __name__ == '__main__':
    main()
    