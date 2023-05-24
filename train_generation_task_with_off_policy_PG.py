# We will create a generalized recipe for off-policy PG training while comparing with standard Supervised Learning (SL) training.

from utils.utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss, log_TP_FP_FN_TN_from_binary_predictions, save_list_of_tuples_to_tsv, load_from_tsv_to_list_of_list, save_in_jsonl, load_from_jsonl, reduce_mean
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
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="Directory containing train, dev and test jsonl", type=str, required=True)
parser.add_argument("-tn", "--task_name", help="Name of the RL4LMs task useful to keep track of reward functions", type=str, required=True)
parser.add_argument("-s", "--save_dir", help="Path to the directory where we will save model and the tokenizer", type=str, required=True)
parser.add_argument("-o", "--output_dir", help="Path to the output directory where we will save all the model prediction and results", type=str, required=True)

# Argument regarding learning algorithm
# parser.add_argument("-bom", "--baseline_offensive_model_dir", help="Path to the directory containing the baseline supervised learning method", type=str, required=True)
parser.add_argument("-m", "--model_name", help="Name of the model to be saved", type=str, required=True)
parser.add_argument("-mt", "--model_tokenizer", help="Directory containing COMET-distill tokenizer", type=str, default=None)
parser.add_argument("-algo", "--learning_algorithm", help="Which type of to include reward in learning", type=str, default="nll", choices=["nll", "pg", "offline_pg", "offline_a2c", "offline_pg_seq", "offline_a2c_seq"])
parser.add_argument("-a2c_n", "--a2c_n_value_head_epochs", help="Number of epochs to train the value head in A2C", type=int, default=5)
parser.add_argument("-r", "--reward_function", help="What reward to use in learning", type=str, default="unit", choices=["unit", "diff_from_goal", "true_prob"])
parser.add_argument("-c", "--ppo_clip", help="PPO clip parameter value. If given the clipped version of importance sampling will be used", type=float, default=None)
parser.add_argument("-ts", "--train_sampling", help="Whether to use reward/advantage sampling during training", action="store_true")
parser.add_argument("-but", "--baseline_update_threshold", help="Update the baseline if new validation is greater than baseline validation + threshold", type=float, default=None)

# Arguments regarding the reward model
parser.add_argument("-cm", "--cola_classifier_model", help="Name of the CoLA classifier model to use for fluency metric", type=str, default=None)
parser.add_argument("-esm", "--embedding_similarity_model", help="Name of the embedding similarity model to use for comparing semantic relevance", type=str, default=None)
parser.add_argument("-scm", "--sentiment_classification_model", help="Name of sentiment classifier model used for IMDB continuation", type=str, default=None)
parser.add_argument("-dnlim", "--doc_nli_model", help="Name of the document NLI model to check the summary entailment", type=str, default=None)
parser.add_argument("-ccm", "--comet_critic_model", help="Directory containing COMET critic model", type=str, default=None)
parser.add_argument("-fcm", "--faithdial_critic_model", help="Name of the FaithDial Critic model to check the faithfulness of the Knowledge-Grounded Dialog responses.", type=str, default=None)
parser.add_argument("-dcm", "--depth_dialogRPT_model", help="Name of the DialogRPT model that predicts if the response is engaging given previous utterance as context", type=str, default=None)
parser.add_argument("-ucm", "--updown_dialogRPT_model", help="Name of the DialogRPT model that predicts if the response relative up-down ratio", type=str, default=None)
parser.add_argument("-om", "--offensive_model_dir", help="Directory containing saved ToxiChat DGPT offensive classifier", type=str, default=None)

parser.add_argument("-t", "--train", help="Flag that will indicate if the model needs to be trained", action="store_true")
parser.add_argument("-bs", "--batch_size", help="Train batch size for regression model", type=int, default=16)
parser.add_argument("-as", "--accumulation_steps", help="Number of steps to accumulate gradients", type=int, default=1)
parser.add_argument("-ev_b", "--eval_in_beginning", help="Should we evaluate the model before training", action="store_true")
parser.add_argument("-v_bs", "--val_batch_size", help="Validation and Test batch size for regression model", type=int, default=64)
parser.add_argument("-vf", "--val_log_frequency", help="How many times should we evaluate in each epoch", type=int, default=2)
parser.add_argument("-e", "--n_epochs", help="Number of epochs", type=int, default=4)
parser.add_argument("-lr", "--learning_rate", help="Number of epochs", type=float, default=2e-5)
parser.add_argument("-ml", "--max_resp_length", help="Maximum length of the input sequence", type=int, default=128)
args = parser.parse_args()

import logging
# Ref: https://stackoverflow.com/a/49202811/4535284
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
# Also add the stream handler so that it logs on STD out as well
# Ref: https://stackoverflow.com/a/46098711/4535284
make_dir_if_not_exists(args.output_dir)
if args.model_name != args.save_dir:
    # We don't want to make a new directory when evaluating pretrained model
    make_dir_if_not_exists(args.save_dir)
if args.train:
    logfile = os.path.join(args.output_dir, "train_output.log")
else:
    logfile = os.path.join(args.output_dir, "output.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(logfile, mode='w'), logging.StreamHandler()])

# Save the copy of arguments in the save directory
import json
with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# Other global constants required for the code
MAX_SEQ_THRESH = 1024
MAX_RESP_LENGTH = 128

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Using {device} to train")

from utils.rl_utils import ValueHeadMLP, LanguageGenerationListofDict, LanguageGenerationCollator, get_model_predictions, train_value_function_on_val_predictions, get_advantage_predictions_on_dataset

# Used to calculate val and test perplexity for DialyDialog evaluation
from utils.attributes_utils import get_batched_dialog_loss, GPT2ForOC_S_offensive
from utils.comet_utils import RobertaClassificationHead, get_period_stopping_critera

def main():
    args.device = device
    # 1. Load the jsonl dataset from input_dir
    logging.info(f"Loading the dataset from {args.input_dir}")
    start_time = time()
    # TEMP: Smaller number of instances for debugging
    # max_lines = 1000
    max_lines = None
    if args.task_name == "reddit_pos":
        train_file = os.path.join(args.input_dir, "pos_train.jsonl")
    elif args.task_name == "reddit_neg":
        train_file = os.path.join(args.input_dir, "neg_train.jsonl")
    else:
        train_file = os.path.join(args.input_dir, "train.jsonl")
    train_data = load_from_jsonl(train_file, max_lines=max_lines)
    logging.info(f"Loaded the train dataset from {train_file} with {len(train_data)} instances")
    
    val_file = os.path.join(args.input_dir, "val.jsonl")
    val_data = load_from_jsonl(val_file, max_lines=max_lines)
    # if len(val_data) > 100000:
    #     logging.info(f"Found more than 100000 val instances. Using 10000 instances for validation")
    # val_data = random.sample(val_data, 1000)
    logging.info(f"Loaded the val dataset from {val_file} with {len(val_data)} instances")
    if args.task_name == "COMET":
        # Reduce val data to 10000 instances
        val_data = random.sample(val_data, 10000)
        logging.info(f"[IMPORTANT] Reduced the val dataset to {len(val_data)} instances for COMET")
    # TEMP: Smaller number of instances for debugging
    # val_data = val_data[:100]
    # train_data = train_data[:1000]
    if args.task_name == "WOW":
        test_file = os.path.join(args.input_dir, "faithdial_test.jsonl")
    else:
        test_file = os.path.join(args.input_dir, "test.jsonl")
    test_data = load_from_jsonl(test_file, max_lines=max_lines)
    logging.info(f"Loaded the test dataset from {test_file} with {len(test_data)} instances")
    logging.info(f"Loaded all segments in {time() - start_time:.2f} seconds")

    # 1.1 Load the tokenizer
    if args.model_name == "rajkumarrrk/t5-fine-tuned-on-iwslt2017en_de":
        logging.info(f"Loading t5-base as tokenizer for {args.model_name} in task {args.task_name}")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
    elif args.model_tokenizer is not None:
        logging.info(f"Loading tokenizer from {args.model_tokenizer}")
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.task_name == "DailyDialog":
        tokenizer.add_tokens(['<EOU>'])
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    tokenize_collator = LanguageGenerationCollator(args, tokenizer)
    # 1.1 Create dataset and dataloaders
    train_dataset = LanguageGenerationListofDict(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=tokenize_collator)

    
    # NOTE: Checking the number of responses exceeding the max response length in CNNDailyMail: 14602/287113 = 5.1%
    # all_resp_len = list()
    # count_over_max = 0
    # for i in tqdm(range(len(train_dataset)), desc="Train response lengths"):
    #     datapoint = train_dataset[i]
    #     response = datapoint['references']
    #     response_inputs = tokenizer(response, return_tensors="pt")
    #     response_len = response_inputs['input_ids'].shape[1]
    #     if response_len > MAX_RESP_LENGTH:
    #         count_over_max += 1
    #     all_resp_len.append(response_len)
    # all_resp_len = np.array(all_resp_len)
    # print(f"Max response length: {all_resp_len.max()}, Min response length: {all_resp_len.min()}, Mean response length: {all_resp_len.mean()}, Count over max: {count_over_max}")
    # breakpoint()
    # Max response length: 3151, Min response length: 7, Mean response length: 74.6828112972941, Count over max: 14602
    
    reward_args = defaultdict(lambda: None)
    if args.task_name == "DailyDialog":
        logging.info(f"Precomputing tf-idf scores for all train responses for task = {args.task_name}")
        train_responses = [datapoint['references'].replace(" <EOU>", "") for datapoint in train_dataset]
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
     
    val_dataset = LanguageGenerationListofDict(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=tokenize_collator)
    test_dataset = LanguageGenerationListofDict(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=tokenize_collator)
    
    # 2. Load the reward models
    reward_args["task_name"] = args.task_name
    reward_args["device"] = device
    # Load the CoLA model as pipeline
    if args.cola_classifier_model is not None:
        start_time = time()
        logging.info(f"Loading the CoLA model pipeline from {args.cola_classifier_model}")
        cola_pipeline = pipeline("sentiment-analysis",model=args.cola_classifier_model, device=0, batch_size=args.val_batch_size)
        logging.info(f"Loaded the CoLA model pipeline in {time() - start_time:.2f} seconds")
        reward_args["cola_classifier_model"] = args.cola_classifier_model
        reward_args["cola_pipeline"] = cola_pipeline
    # Load the Sentiment Classification model as pipeline
    if args.sentiment_classification_model is not None:
        start_time = time()
        logging.info(f"Loading the Sentiment Classification model pipeline from {args.sentiment_classification_model}")
        sentiment_classification_pipeline = pipeline("sentiment-analysis",model=args.sentiment_classification_model, device=0, batch_size=args.val_batch_size)
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
    # Load the comet critic model
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
    # Load the DialogRPT-depth model
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
    # Load the DialogRPT-updown model
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
    # Load the Toxichat Offensive Classifier
    if args.offensive_model_dir is not None:
        start_time = time()
        offensive_model = GPT2ForOC_S_offensive.from_pretrained(args.offensive_model_dir).half().to(device)
        offensive_tokenizer = AutoTokenizer.from_pretrained(args.offensive_model_dir)
        logging.info(f"Loaded offensive model from {args.offensive_model_dir} in {time() - start_time:.2f} seconds")
        reward_args["offensive_model"] = offensive_model
        reward_args["offensive_tokenizer"] = offensive_tokenizer

    if args.task_name in ["WOW", "reddit_pos", "reddit_neg"]:
        logging.info(f"Loading tf-idf scores for all train responses for task = {args.task_name}")
        # Load the tfidf_transformer and cv
        tfidf_transformer_save_file = os.path.join(args.input_dir, "tfidf_transformer.pkl")
        cv_save_file = os.path.join(args.input_dir, "cv.pkl")
        tfidf_transformer = load_from_pickle(tfidf_transformer_save_file)
        cv = load_from_pickle(cv_save_file)
        reward_args["cv"] = cv
        reward_args["tfidf_transformer"] = tfidf_transformer
        logging.info(f"Precomputed tf-idf scores in {time() - start_time:.2f} seconds")

    #2.3 Load the model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name)
    # Depending on the config we need to choose between Seq2SeqLM and CausalLM
    if config.is_encoder_decoder:
        args.model_type = "seq2seq"
        AutoModelLM = AutoModelForSeq2SeqLM
    else:
        args.model_type = "causal"
        AutoModelLM = AutoModelForCausalLM
        tokenizer.pad_token = tokenizer.eos_token
    if args.train:
        # Create new model from scratch
        start_time = time()
        logging.info(f"Loading model from {args.model_name}...")
        
        model = AutoModelLM.from_pretrained(args.model_name, config=config).to(device)
        if args.task_name == "DailyDialog" and model.config.vocab_size != len(tokenizer):
            # Resize model embeddings to match the new vocabulary size
            model.resize_token_embeddings(len(tokenizer))
        logging.info(f"Loaded model in {time() - start_time:.2f} seconds")
        if args.learning_algorithm.startswith("offline_"):
            # Also create a baseline model
            logging.info(f"Also creating a baseline model from {args.model_name}")
            baseline_model = AutoModelLM.from_pretrained(args.model_name, config=config).to(device)
            if args.task_name == "DailyDialog" and baseline_model.config.vocab_size != len(tokenizer):
                # Resize model embeddings to match the new vocabulary size
                baseline_model.resize_token_embeddings(len(tokenizer))
            baseline_model.eval()
            if args.learning_algorithm == "offline_a2c":
                # Also initialize an external value function estimator for A2C
                logging.info(f"Also creating a value function estimator for the model {args.model_name}")
                # Change the max_value to task specific
                if args.task_name in ["IWSLT2017EnDe", "COMET"]:
                    args.max_value = 1.0
                elif args.task_name in ["Xsum", "CNNDailyMail", "IMDBForSeq2Seq", "DailyDialog"]:
                    args.max_value = 2.0
                elif args.task_name in ["WOW"]:
                    args.max_value = 4.0
                elif args.task_name in ["reddit_pos", "reddit_neg"]:
                    args.max_value = 5.0
                else:
                    logging.error(f"Task {args.task_name} not yet defined for value function estimator")
                    breakpoint()
                value_function_estimator = ValueHeadMLP(config, max_value=args.max_value).to(device)

    else:
        # Load from a previously trained model
        logging.info(f"Loading pretrained model and tokenizer from {args.save_dir}...")
        model = AutoModelLM.from_pretrained(args.save_dir, config=config).to(device)
        # tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    
    if args.baseline_update_threshold is not None or args.learning_algorithm == "offline_a2c":
        # Eval in beginning should be true to get baseline model performance
        assert args.eval_in_beginning, breakpoint()
    baseline_success_measure = None
    if args.task_name == "IMDBForSeq2Seq":
        args.success_measure_str = "Sentiment Accuracy"
    elif args.task_name == "COMET":
        args.success_measure_str = "COMET Critic Score"
    elif args.task_name == "DailyDialog":
        args.success_measure_str = "-ve perplexity"
    elif args.task_name in ["reddit_pos", "reddit_neg"]:
        args.success_measure_str = "Final reward"
    elif args.task_name == "WOW":
        args.success_measure_str = "FaithDial Critic Score"
    else:
        args.success_measure_str = "Meteor Score"
    if args.eval_in_beginning:
        logging.info(f"############## Running Validation before training...")
        if args.learning_algorithm in ["nll", "pg"]: baseline_model = model
        # Make predictions on val set
        all_ids, all_gen_responses, all_val_responses, all_gen_rewards, all_gold_rewards, meteor_score = get_model_predictions(val_dataloader, baseline_model, tokenizer, device, reward_args, args)
        # Get per reward average
        gen_reward_avg = {k: sum([e[k] for e in all_gen_rewards])/len(all_gen_rewards) for k in all_gen_rewards[0].keys()}
        gold_reward_avg = {k: sum([e[k] for e in all_gold_rewards])/len(all_gold_rewards) for k in all_gold_rewards[0].keys()}
        logging.info(f"val_reward_avg: {gold_reward_avg}")
        logging.info(f"gen_reward_avg: {gen_reward_avg}")
        logging.info(f"METEOR: {meteor_score}")
        if args.task_name == "IMDBForSeq2Seq":
            best_success_measure = gen_reward_avg["sentiment"]
        elif args.task_name == "COMET":
            best_success_measure = gen_reward_avg["p_valid_model"]
        elif args.task_name in ["reddit_pos", "reddit_neg"]:
            best_success_measure = gen_reward_avg["final_reward"]
        elif args.task_name == "WOW":
            best_success_measure = gen_reward_avg["faithdial"]
        elif args.task_name == "DailyDialog":
            # Calculate model perplexity on val set
            val_prompt_and_resps = [f"{e['prompt_or_input_text']} {e['references']}" for e in val_dataset]
            all_val_gold_resp_loss = get_batched_dialog_loss(val_prompt_and_resps, all_val_responses, baseline_model, tokenizer, device, batch_size=args.val_batch_size)
            val_perlexity = torch.exp(torch.tensor(all_val_gold_resp_loss).mean()).item()
            best_success_measure = -val_perlexity
            # logging.info(f"Baseline -ve val perplexity: {-val_perlexity}")
        else:
            best_success_measure = meteor_score
        logging.info(f"Initial {args.success_measure_str}: {best_success_measure}")
        if args.baseline_update_threshold is not None:
            logging.info(f"Also setting the baseline model {args.success_measure_str} to {best_success_measure} since we are using baseline update threshold of {args.baseline_update_threshold}")
            baseline_success_measure = best_success_measure
        if args.learning_algorithm == "offline_a2c":
            # Estimate the value function on the dev predictions of baseline model
            logging.info(f"Also estimating the value function on the initial dev predictions of baseline model")
            best_value_function_model, best_value_mse, best_epoch = train_value_function_on_val_predictions(value_function_estimator, baseline_model, tokenizer, val_dataloader, all_ids, all_gen_responses, all_gen_rewards, args)
            best_value_function_model.eval()
            logging.info(f"Best value function model MSE: {best_value_mse} at val epoch {best_epoch}")
            

    else:
        best_success_measure = -1000000.0
    best_epoch = -1
    best_model = None

    if args.train:
        # Trying to find out the callable methods from the model object
        # Ref: https://stackoverflow.com/a/34452/4535284
        # object_methods = [method_name for method_name in dir(model) if callable(getattr(model, method_name))]
        # print(object_methods)
        # exit()
        epochs = args.n_epochs
        # Add sampling weights right before training
        if args.train_sampling:
            if args.learning_algorithm in ["pg", "offline_pg_seq"]:
                if args.task_name in ["Xsum", "CNNDailyMail"]:
                    # NOTE: not using doc_nli reward right now. For CNN the doc_nli reward is mostly close to 0.
                    train_rewards = [datum['reward_components']['fluency'] + datum['reward_components']['text_sim'] for datum in train_dataset]
                    # rewards = [datum['reward_components']['fluency'] + datum['reward_components']['text_sim'] + datum['reward_components']['doc_nli_score'] for datum in batch]
                elif args.task_name in ["IWSLT2017EnDe", "IMDBForSeq2Seq", "DailyDialog", "COMET", "WOW", "reddit_pos", "reddit_neg"]:
                    train_rewards = [datum['reward_components']['final_reward'] for datum in train_dataset]
                train_rewards = np.array(train_rewards)
                train_dataset.set_sample_weights(train_rewards)
                logging.info(f"Using train_rewards for sampling instead of random sampling")
                total_steps = len(train_dataloader) * epochs
            elif args.learning_algorithm == "offline_a2c":
                all_advantages = get_advantage_predictions_on_dataset(train_dataset, tokenize_collator, baseline_model, best_value_function_model, args)
                # Find ratio of instances with negative advantage
                num_instances_with_negative_advantage = len([e for e in all_advantages if e < 0])
                ratio = num_instances_with_negative_advantage / len(all_advantages)
                logging.info(f"Ratio of instances with negative advantage: {ratio * 100.0:.2f}%")
                # convert all negative advantages to 0
                np_all_advantages = np.array([e if e > 0 else 0.0 for e in all_advantages])
                train_dataset.set_sample_weights(np_all_advantages)
                logging.info(f"Using positive advantages for sampling instead of random sampling")
                # Also change the len(train_dataloader) to len(train_dataloader) * (1-ratio)
                total_steps = int(len(train_dataloader) * epochs * (1-ratio))
        else:
            total_steps = len(train_dataloader) * epochs
        # Start training
        
        # Create optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=0.01)
        # NOTE: num_warmup_steps = 0 is the Default value in run_glue.py
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        logging.info(f"Created model optimizer with learning rate = {args.learning_rate}")

        # Log the algorithm parameters
        logging.info(f"Learning algorithm: {args.learning_algorithm}")
        logging.info(f"Baseline update threshold: {args.baseline_update_threshold}")
        logging.info(f"PPO clip: {args.ppo_clip}")
        logging.info(f"Batch size: {args.batch_size}")
        logging.info(f"Accumulation steps: {args.accumulation_steps}")
        logging.info(f"Number of epochs: {args.n_epochs}")
        logging.info(f"Learning rate: {args.learning_rate}")
        logging.info(f"Val log frequency: {args.val_log_frequency}")
        logging.info(f"Max response length: {args.max_resp_length}")
        logging.info(f"Train sampling: {args.train_sampling}")


        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = list()
        validation_stats = list()

        logging.info(f"Initiating training loop for {args.n_epochs} epochs...")
        # Measure the total training time for the whole run.
        total_start_time = time()

        # Find the accumulation steps
        accumulation_steps = args.accumulation_steps

        # Loss trajectory for epochs
        epoch_train_loss = list()
        # Val validation trajectory

        # Create empty train stats and validation stats jsonl files that will be appended to
        validation_stats_file = os.path.join(args.output_dir, "validation_stats.jsonl")
        training_stats_file = os.path.join(args.output_dir, "training_stats.jsonl")
        open(validation_stats_file, "w").close()
        open(training_stats_file, "w").close()

        for epoch in range(epochs):
            logging.info(f"Initiating Epoch {epoch+1}/{epochs}:")
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            # Reset the total loss for each epoch.
            total_train_loss = 0
            train_loss_trajectory = list()

            # Reset timer for each epoch
            start_time = time()
            model.train()
            model.zero_grad()

            val_log_frequency = args.val_log_frequency
            n_steps = len(train_dataloader)
            if args.train_sampling and args.learning_algorithm == "offline_a2c":
                # NOTE: We are sampling from the train dataset instead of random sampling
                n_steps = int(n_steps * (1-ratio))
            val_steps = int(n_steps / val_log_frequency)
            current_epoch_loss_trajectories = defaultdict(list)
            for step, batch in enumerate(pbar):
                prompt_inputs, response_inputs, extra_data = batch
                # Set response_inputs as labels for the model
                input_ids = prompt_inputs["input_ids"].to(device)
                attention_mask = prompt_inputs["attention_mask"].to(device)
                labels = response_inputs["input_ids"].to(device)

                # Tokenize the inputs in the batch and create input_ids and attention_mask for the model
                # Ref: https://github.com/huggingface/transformers/issues/3021
                if args.model_type == "seq2seq":
                    input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                    # Forward
                    outputs = model(**input_dict)
                    # loss = outputs.loss
                    logits = outputs.logits
                elif args.model_type == "causal":
                    # Merge the prompt and response inputs
                    input_ids = torch.cat([prompt_inputs["input_ids"], response_inputs["input_ids"]], dim=1).to(device)
                    attention_mask = torch.cat([prompt_inputs["attention_mask"], response_inputs["attention_mask"]], dim=1).to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    batch_size, query_seq_len = prompt_inputs["input_ids"].shape
                    logits = outputs.logits[:, (query_seq_len - 1):-1, :]
                # Gather per label logits
                response_ids = response_inputs["input_ids"]
                response_mask = response_inputs["attention_mask"].to(device)
                # Logits are before softmax. So, we need to apply softmax to get the probabilities
                # logits is of the shape [batch_size, seq_len, vocab_size]
                # labels is of the shape [batch_size, seq_len]
                log_probs = F.log_softmax(logits, dim=-1)
                per_token_log_probs = -torch.gather(log_probs, 2, labels[:, :, None]).squeeze(2)
                if args.learning_algorithm in ["nll", "pg"]:
                    per_response_loss = torch.sum(per_token_log_probs * response_mask, dim=1)
                    if args.learning_algorithm == "pg":
                        # Multiply the loss with the rewards
                        rewards = torch.tensor(extra_data['rewards']).to(device)
                        current_epoch_loss_trajectories['rewards'].extend(rewards.tolist())
                        per_response_loss = per_response_loss * rewards
                elif args.learning_algorithm in ["offline_pg_seq", "offline_a2c"]:
                    with torch.no_grad():
                        # Compute the baseline model log probs
                        if args.model_type == "seq2seq":
                            baseline_outputs = baseline_model(**input_dict, output_hidden_states=True)
                            baseline_logits = baseline_outputs.logits
                        elif args.model_type == "causal":
                            # Merge the prompt and response inputs
                            baseline_outputs = baseline_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                            batch_size, query_seq_len = prompt_inputs["input_ids"].shape
                            baseline_logits = outputs.logits[:, (query_seq_len - 1):-1, :]
                        
                        if args.learning_algorithm == "offline_a2c":
                            if args.model_type == "seq2seq":
                                last_layer_encoder_hidden_state = baseline_outputs.encoder_last_hidden_state
                                # last_layer_encoder_hidden_state is of the shape (batch_size, seq_len, hidden_size)
                                # Get value function predictions for the generated responses
                            elif args.model_type == "causal":
                                last_layer_hidden_state = baseline_outputs.hidden_states[-1]
                                last_layer_encoder_hidden_state = last_layer_hidden_state[:, :query_seq_len, :]
                            # Calculate the value function estimates for current batch
                            # Get the last layer encoder hidden state
                            val_outputs = best_value_function_model(last_layer_encoder_hidden_state)
                        baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)
                        baseline_per_token_log_probs = -torch.gather(baseline_log_probs, 2, labels[:, :, None]).squeeze(2)
                        # Compute the seq importance sampling ratio over each word
                        # Both per_token_log_probs and baseline_per_token_log_probs are negative log probs
                        importance_sampling_ratio_per_token = torch.exp(baseline_per_token_log_probs - per_token_log_probs.detach()) * response_mask
                        if args.ppo_clip is not None:
                            # Clamp the importance sampling ratio
                            importance_sampling_ratio_per_token_clamped = torch.clamp(importance_sampling_ratio_per_token, 1 - args.ppo_clip, 1 + args.ppo_clip) * response_mask
                            # return_dict['importance_sampling_ratio_per_token_clamped'] = importance_sampling_ratio_per_token_clamped
                            importance_sampling_ratio_per_token = importance_sampling_ratio_per_token_clamped
                    # Multiply importance sampling with log probs
                    lm_loss = torch.mul(per_token_log_probs, importance_sampling_ratio_per_token)
                    per_response_loss_with_importance_sampling = reduce_mean(lm_loss, response_mask, axis=1)
                    current_epoch_loss_trajectories['loss w/ IS'].extend(per_response_loss_with_importance_sampling.tolist())
                    # Get the rewards from extra args
                    # extra_data is a dict with dict_keys(['texts', 'responses', 'batch', 'rewards'])
                    # Convert list of rewards from extra_data['rewards'] to tensor
                    rewards = torch.tensor(extra_data['rewards']).to(device)
                    if args.learning_algorithm == "offline_a2c":
                        # Multiply the loss with the advantage
                        advantage = rewards - val_outputs
                        advantage = torch.clamp(advantage, min=0.0)
                        current_epoch_loss_trajectories['advantage'].extend(advantage.tolist())
                        per_response_loss = per_response_loss_with_importance_sampling * advantage
                    else:
                        current_epoch_loss_trajectories['rewards'].extend(rewards.tolist())
                        per_response_loss = per_response_loss_with_importance_sampling * rewards
                else:
                    logging.error(f"Unknown learning algorithm: {args.learning_algorithm}")
                current_epoch_loss_trajectories['final loss'].extend(per_response_loss.tolist())
                loss = torch.mean(per_response_loss)

                # Backward: compute gradients
                loss.backward()
                
                total_train_loss += loss.item()
                if (step + 1) % accumulation_steps == 0:
                    
                    # Calculate elapsed time in minutes and print loss on the tqdm bar
                    elapsed = format_time(time() - start_time)
                    avg_train_loss = total_train_loss/(step+1)
                    # keep track of changing avg_train_loss
                    # Get avg and running avg loss for each key in current_epoch_loss_trajectories
                    avg_losses = {f"avg {key}": f"{np.mean(current_epoch_loss_trajectories[key]):.3f}" for key in current_epoch_loss_trajectories}
                    last_losses = {f"{key}": f"{np.mean(current_epoch_loss_trajectories[key][-args.batch_size:]):.3f}" for key in current_epoch_loss_trajectories}
                    pbar.set_postfix(**avg_losses, **last_losses)
                    train_loss_trajectory.append(last_losses)

                    # Clip the norm of the gradients to 1.0.
                    # This is to help prevent the "exploding gradients" problem.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Update parameters
                    optimizer.step()

                    # Clean the model's previous gradients
                    model.zero_grad()                           # Reset gradients tensors

                    # Update the learning rate.
                    scheduler.step()
                    pbar.update()
                if (step + 1) % val_steps == 0:
                    logging.info(f"############## Running Validation for epoch {epoch+1} step {step+1} ##############")
                    torch.cuda.empty_cache()
                    # Make predictions on val set
                    all_ids, all_gen_responses, all_val_responses, all_gen_rewards, all_gold_rewards, meteor_score = get_model_predictions(val_dataloader, model, tokenizer, device, reward_args, args)
                    # Get per reward average
                    gen_reward_avg = {k: sum([e[k] for e in all_gen_rewards])/len(all_gen_rewards) for k in all_gen_rewards[0].keys()}
                    gold_reward_avg = {k: sum([e[k] for e in all_gold_rewards])/len(all_gold_rewards) for k in all_gold_rewards[0].keys()}
                    logging.info(f"val_reward_avg: {gold_reward_avg}")
                    logging.info(f"gen_reward_avg: {gen_reward_avg}")
                    logging.info(f"METEOR: {meteor_score}")
                    if args.task_name == "IMDBForSeq2Seq":
                        current_model_success_measure = gen_reward_avg["sentiment"]
                    elif args.task_name == "COMET":
                        current_model_success_measure = gen_reward_avg["p_valid_model"]
                    elif args.task_name == "WOW":
                        current_model_success_measure = gen_reward_avg["faithdial"]
                    elif args.task_name in ["reddit_pos", "reddit_neg"]:
                        current_model_success_measure = gen_reward_avg["final_reward"]
                    elif args.task_name == "DailyDialog":
                        # Calculate model perplexity on val set
                        val_prompt_and_resps = [f"{e['prompt_or_input_text']} {e['references']}" for e in val_dataset]
                        all_val_gold_resp_loss = get_batched_dialog_loss(val_prompt_and_resps, all_val_responses, model, tokenizer, device, batch_size=args.val_batch_size)
                        val_perlexity = torch.exp(torch.tensor(all_val_gold_resp_loss).mean()).item()
                        current_model_success_measure = -val_perlexity
                        logging.info(f"Current model -ve perplexity on val set: {-val_perlexity}")
                    else:
                        current_model_success_measure = meteor_score
                    if best_success_measure < current_model_success_measure:
                        # Keep the copy of current model
                        logging.info(f"New best Val {args.success_measure_str} = {current_model_success_measure} achieved at epoch {epoch+1}. Compared to previous best Val {args.success_measure_str} = {best_success_measure} that was achieved at epoch {best_epoch}")
                        # Clear the GPU cache
                        torch.cuda.empty_cache()
                        model.to('cpu')
                        best_model = deepcopy(model)
                        model.to(device)
                        best_success_measure = current_model_success_measure
                        best_epoch = epoch+1
                        # Save the model and the Tokenizer now
                        logging.info(f"Saving the model and tokenizer in {args.save_dir}")
                        model.save_pretrained(args.save_dir)
                        tokenizer.save_pretrained(args.save_dir)
                        if args.baseline_update_threshold is not None:
                            # Check if best_success_measure score is greater than baseline_success_measure + args.baseline_update_threshold
                            if best_success_measure > baseline_success_measure + args.baseline_update_threshold: 
                                # Update the baseline model since new best model is found
                                baseline_model = deepcopy(model)
                                baseline_model.eval()
                                logging.info(f"Updated the baseline model to best checkpoint because best {args.success_measure_str} {best_success_measure} is greater than baseline {args.success_measure_str} {baseline_success_measure} + threshold {args.baseline_update_threshold}")
                                baseline_success_measure = best_success_measure
                                logging.info(f"Updated the baseline {args.success_measure_str} to {baseline_success_measure}")
                                # Also update the sampling weights
                                if args.train_sampling:
                                    if args.learning_algorithm == "offline_pg_seq":
                                        if args.task_name in ["Xsum", "CNNDailyMail"]:
                                            # NOTE: not using doc_nli reward right now. For CNN the doc_nli reward is mostly close to 0.
                                            train_rewards = [datum['reward_components']['fluency'] + datum['reward_components']['text_sim'] for datum in train_dataset]
                                            # rewards = [datum['reward_components']['fluency'] + datum['reward_components']['text_sim'] + datum['reward_components']['doc_nli_score'] for datum in batch]
                                        elif args.task_name in ["IWSLT2017EnDe", "IMDBForSeq2Seq", "DailyDialog", "COMET", "WOW", "reddit_pos", "reddit_neg"]:
                                            train_rewards = [datum['reward_components']['final_reward'] for datum in train_dataset]
                                        train_rewards = np.array(train_rewards)
                                        train_dataset.set_sample_weights(train_rewards)
                                        logging.info(f"Using train_rewards for sampling instead of random sampling")
                                    elif args.learning_algorithm == "offline_a2c":
                                        # Re-Estimate the value function on the dev predictions of baseline model
                                        value_function_estimator = ValueHeadMLP(config, max_value=args.max_value).to(device)
                                        torch.cuda.empty_cache()
                                        logging.info(f"Restimating the value function on the latest dev predictions of baseline model")
                                        best_value_function_model, best_value_mse, best_epoch = train_value_function_on_val_predictions(value_function_estimator, baseline_model, tokenizer, val_dataloader, all_ids, all_gen_responses, all_gen_rewards, args)
                                        best_value_function_model.eval()
                                        logging.info(f"Best value function model MSE: {best_value_mse} at val epoch {best_epoch}")
                                        torch.cuda.empty_cache()
                                        all_advantages = get_advantage_predictions_on_dataset(train_dataset, tokenize_collator, baseline_model, best_value_function_model, args)
                                        # Find ratio of instances with negative advantage
                                        num_instances_with_negative_advantage = len([e for e in all_advantages if e < 0])
                                        ratio = num_instances_with_negative_advantage / len(all_advantages)
                                        logging.info(f"Ratio of instances with negative advantage: {ratio * 100.0:.2f}%")
                                        # convert all negative advantages to 0
                                        np_all_advantages = np.array([e if e > 0 else 0.0 for e in all_advantages])
                                        train_dataset.set_sample_weights(np_all_advantages)
                                        logging.info(f"Using positive advantages for sampling instead of random sampling")
                                        torch.cuda.empty_cache()
                            else:
                                logging.info(f"Current best Val {args.success_measure_str} = {best_success_measure} that was achieved at epoch {best_epoch}")

                    current_validation_stats = {
                        'epoch': epoch + 1,
                        'step': step + 1,
                        'meteor': meteor_score,
                        'gen_reward_avg': gen_reward_avg,
                        'val_reward_avg': gold_reward_avg,
                    }
                    if args.task_name == "IMDBForSeq2Seq":
                        current_validation_stats["sentiment"] = current_model_success_measure
                    elif args.task_name == "COMET":
                        current_validation_stats["p_valid_model"] = current_model_success_measure
                    elif args.task_name == "WOW":
                        current_validation_stats["faithdial"] = current_model_success_measure
                    elif args.task_name in ["reddit_pos", "reddit_neg"]:
                        current_validation_stats["final_reward"] = current_model_success_measure
                    elif args.task_name == "DailyDialog":
                        current_validation_stats["-ve perplexity"] = -val_perlexity
                    validation_stats.append(current_validation_stats)
                    # Append to the validation stats jsonl file
                    validation_stats_file = os.path.join(args.output_dir, "validation_stats.jsonl")
                    logging.info(f"Appending the validation results in {validation_stats_file}")
                    save_in_jsonl([current_validation_stats], validation_stats_file, append=True)

                    # Put the model back in train setting
                    model.train()
                
                if step >= n_steps:
                    logging.info(f"Reached the end of epoch {epoch+1}. Breaking the training loop")
                    break

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)
            
            training_time = format_time(time() - start_time)

            # Record all statistics from this epoch.
            train_avg_losses = {f"Training avg {key}": f"{np.mean(current_epoch_loss_trajectories[key]):.3f}" for key in current_epoch_loss_trajectories}
            current_epoch_train_stats = {'epoch': epoch + 1,
                                         'Training Time': training_time,
                                         **train_avg_losses,}
            training_stats.append(current_epoch_train_stats)
            # Append the training stats in jsonl file
            training_stats_file = os.path.join(args.output_dir, "training_stats.jsonl")
            save_in_jsonl([current_epoch_train_stats], training_stats_file, append=True)

            # Save the loss trajectory
            epoch_train_loss.append(train_loss_trajectory)
        logging.info(f"Training complete with total Train time:{format_time(time()- total_start_time)}")
        log_list(training_stats)
        
        
        def plot_train_loss(loss_trajectory_per_epoch, trajectory_file):
            plt.cla()
            plt.clf()

            # loss_trajectory_per_epoch is a list of lists
            # each internal list contains dictionary of various loss component across the epoch train step
            # Example dictionary: {'final loss': '360.981'}
            x = [epoch * len(loss_trajectory) + j + 1 for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, per_step_avg_loss_dict in enumerate(loss_trajectory) ]
            # x_ticks = [ epoch + 1 for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, per_step_avg_loss_dict in enumerate(loss_trajectory) ]
            loss_keys = list(loss_trajectory_per_epoch[0][0].keys())
            per_loss_train_trajectory = defaultdict(list)
            for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch):
                for j, per_step_avg_loss_dict in enumerate(loss_trajectory):
                    for loss_key in loss_keys:
                        per_loss_train_trajectory["epoch_step"].append(epoch * len(loss_trajectory) + j + 1)
                        per_loss_train_trajectory["loss_type"].append(loss_key)
                        per_loss_train_trajectory["loss_value"].append(float(per_step_avg_loss_dict[loss_key]))
            
            # prepare pandas dataframe for plotting epoch vs all types of losses
            df = pd.DataFrame(per_loss_train_trajectory)
            loss_trajectory_plot = sns.lineplot(data=df, x="epoch_step", y="loss_value", hue="loss_type")
            loss_trajectory_plot.set_title("Train loss trajectory")
            loss_trajectory_plot.set_xlabel("Epoch Step")
            loss_trajectory_plot.set_ylabel("Loss Value")
            # Save the plot
            plt.tight_layout()
            loss_trajectory_plot.figure.savefig(trajectory_file, dpi=300)
            logging.info(f"Saved the train loss trajectory plot to {trajectory_file}")
            plt.clf()
            plt.cla()
        # Plot the train loss trajectory in a plot
        train_loss_trajectory_plot_file = os.path.join(args.output_dir, "train_loss_trajectory.png")
        plot_train_loss(epoch_train_loss, train_loss_trajectory_plot_file)

        # logging.info(f"Re Saving the training stats at {training_stats_file}")
        # save_in_jsonl(training_stats, training_stats_file, append=False)
        # logging.info(f"Re Saving the validation stats at {validation_stats_file}")
        # save_in_jsonl(validation_stats, validation_stats_file, append=False)
        
        # Log the best model stats and save it
        logging.info(f"Best Val {args.success_measure_str} = {best_success_measure} at epoch {best_epoch}.")
        model = best_model
        model.to(device)
        # Save the model and the Tokenizer here:
        logging.info(f"Saving the model and tokenizer in {args.save_dir}")
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)


        # TODO: Plot the validation performance
        # Save val_validation_statistics
    else:
        logging.info("No training needed. Directly going to evaluation!")



    def plot_gen_vs_gold_reward_distribution(segment_name, all_gen_rewards, all_gold_rewards,):
        components = list()
        rewards = list()
        sources = list()
        for source, reward_components_list in [("Generated", all_gen_rewards), ("Gold", all_gold_rewards)]:
            for reward_components in reward_components_list:
                for component, reward in reward_components.items():
                    components.append(component)
                    rewards.append(reward)
                    sources.append(source)
        df = pd.DataFrame({"components": components, "reward": rewards, "source": sources})
        # Plot violin plot
        # Set figure size
        plt.figure(figsize=(10, 9))
        violin_plot = sns.violinplot(data=df, x="components", y="reward", hue="source", split=True)
        violin_plot.set(xlabel="Reward Components", ylabel="Reward distribution")
        # Add total counts/percentage of instances for each threshold window
        xticklabels = list(reward_components_list[0].keys())
        # violin_plot.set_xticklabels(violin_plot.get_xticklabels(), rotation=90)
        violin_plot.set_xticklabels(xticklabels, rotation=70)
        violin_plot.set_title(f"Per component reward distribution for segment = {segment_name}, containing {len(reward_components_list)} instances")
        violin_plot_save_file = os.path.join(args.output_dir, f"final_generation_reward_distribution_plot_{segment_name}.png")
        # Tight layout
        plt.tight_layout()
        violin_plot.figure.savefig(violin_plot_save_file, dpi=300)
        logging.info(f"Saved violin plot to {violin_plot_save_file}")
        plt.clf()
        plt.cla()



    logging.info(f"#####################  Final evaluation")
    # Make predictions on val set
    all_ids, all_gen_responses, all_val_responses, all_gen_rewards, all_gold_rewards, meteor_score = get_model_predictions(val_dataloader, model, tokenizer, device, reward_args, args)
    # Get per reward average
    gen_reward_avg = {k: sum([e[k] for e in all_gen_rewards])/len(all_gen_rewards) for k in all_gen_rewards[0].keys()}
    gold_reward_avg = {k: sum([e[k] for e in all_gold_rewards])/len(all_gold_rewards) for k in all_gold_rewards[0].keys()}
    
    # Plot side-by-side reward distribution with violin plot
    plot_gen_vs_gold_reward_distribution("val", all_gen_rewards, all_gold_rewards)
    
    logging.info(f"val_reward_avg: {gold_reward_avg}")
    logging.info(f"val gen_reward_avg: {gen_reward_avg}")
    logging.info(f"val METEOR: {meteor_score}")
    final_val_stats = {
        'best_epoch': best_epoch,
        'meteor': meteor_score,
        'gen_reward_avg': gen_reward_avg,
        'val_reward_avg': gold_reward_avg,
    }
    if args.task_name == "DailyDialog":
        # Calculate model perplexity on val set
        val_prompt_and_resps = [f"{e['prompt_or_input_text']} {e['references']}" for e in val_dataset]
        all_val_gold_resp_loss = get_batched_dialog_loss(val_prompt_and_resps, all_val_responses, model, tokenizer, device, batch_size=args.val_batch_size)
        val_perlexity = torch.exp(torch.tensor(all_val_gold_resp_loss).mean()).item()
        logging.info(f"Final val Perplexity: {val_perlexity}")
        final_val_stats['-ve perplexity'] = -val_perlexity


    # Make predictions on test set
    all_ids, all_gen_responses, all_test_responses, all_gen_rewards, all_gold_rewards, meteor_score = get_model_predictions(test_dataloader, model, tokenizer, device, reward_args, args)
    # Get per reward average
    gen_reward_avg = {k: sum([e[k] for e in all_gen_rewards])/len(all_gen_rewards) for k in all_gen_rewards[0].keys()}
    gold_reward_avg = {k: sum([e[k] for e in all_gold_rewards])/len(all_gold_rewards) for k in all_gold_rewards[0].keys()}

    # Plot side-by-side reward distribution with violin plot
    plot_gen_vs_gold_reward_distribution("test", all_gen_rewards, all_gold_rewards)

    logging.info(f"test_reward_avg: {gold_reward_avg}")
    logging.info(f"test gen_reward_avg: {gen_reward_avg}")
    logging.info(f"test METEOR: {meteor_score}")
    final_test_stats = {
        'best_epoch': best_epoch,
        'meteor': meteor_score,
        'gen_reward_avg': gen_reward_avg,
        'test_reward_avg': gold_reward_avg,
    }
    if args.task_name == "DailyDialog":
        # Calculate model perplexity on test set
        test_prompt_and_resps = [f"{e['prompt_or_input_text']} {e['references']}" for e in test_dataset]
        all_test_gold_resp_loss = get_batched_dialog_loss(test_prompt_and_resps, all_test_responses, model, tokenizer, device, batch_size=args.val_batch_size)
        test_perlexity = torch.exp(torch.tensor(all_test_gold_resp_loss).mean()).item()
        logging.info(f"Final test Perplexity: {test_perlexity}")
        final_test_stats['-ve perplexity'] = -test_perlexity
    # Save the final val and test stats in jsonl file
    if args.train:
        final_eval_stats_file = os.path.join(args.output_dir, "train_final_eval_stats.jsonl")
    else:
        final_eval_stats_file = os.path.join(args.output_dir, "test_final_eval_stats.jsonl")
    logging.info(f"Saving the final evaluation stats at {final_eval_stats_file}")
    final_stats = {"val": final_val_stats, "test": final_test_stats}
    save_in_jsonl([final_stats], final_eval_stats_file)

if __name__ == '__main__':
    main()
    