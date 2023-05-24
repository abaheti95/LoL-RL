import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import collections
from copy import deepcopy

T = TypeVar('T')
NEGATIVE_INF = -100000.0


def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


def reduce_std(value, mask):
    return torch.sqrt(reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask)))


def logits_to_entropy(logits):
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy()


def mask_pad(value, mask):
    return value * mask + NEGATIVE_INF * (1 - mask)


def clamp(value, min_value, max_value):
    return torch.max(torch.min(value, max_value), min_value)


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def whiten(values, masks, shift_mean=True):
    mean, var = reduce_mean(values, masks), reduce_std(values, masks)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def flatten_dict(nested, sep='.'):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat


def distinctness(generations):
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])

    return len(unigrams) / total_words, len(bigrams) / total_words, len(trigrams) / total_words


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)


def disjunction(a, b):
    c = deepcopy(b)
    for k in a.keys():
        c[k] = a[k] or b[k]
    return c


#################################################################################
### My custom utils
#################################################################################

# Will save all the frequently used utils in this file

import os
import re
import string
import collections
import json
import pickle
import csv
import datetime
# TEMP: commented sklearn because creating random library issue
# ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.22' not found (required by /coc/flash5/abaheti3/miniconda3/envs/rlctg/lib/python3.9/site-packages/scipy/stats/_qmc_cy.cpython-39-x86_64-linux-gnu.so)
# from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
import random
RANDOM_SEED = 901
random.seed(RANDOM_SEED)


def remove_empty_str(row):
	return [x for x in row if x != ""]
	
def split_into_words(string):
	return [e.strip() for e in string.strip().split() if e.strip() != ""]

def remove_multiple_space(string):
	return re.sub(r'\s+', ' ', string).strip()

def longest_common_substring_from_left(s1, s2):
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            break
    return s1[:i]

URL_TOKEN = "<URL>"

url_regex = re.compile(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9@:;%_\\\+.~#?&//=]*)")
def replace_urls(post, replace_tok=URL_TOKEN):
	return url_regex.subn(replace_tok, post)

# Links in reddit are highlighted using markdown as follows [url text](URL)
# We will remove the URL in the brackets as it is not contributing new information
markdown_link_regex = re.compile(r"\[(.+)\]\(.+\)")
def remove_markdown_urls(reddit_post_or_comment):
	cleaned_post_or_comment, n_links = markdown_link_regex.subn(r"\1", reddit_post_or_comment)
	return cleaned_post_or_comment.strip(), n_links

gendered_pronouns = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "he's", "she's"}
stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "i'm", "that's", "you're", "they're", "it's", "he's", "i've", ".", "?", "”", "“", "‘", ",", "-", "—", "!", ":", ";", "(", ")", "[", "]", "…", "/", '"', "'"}

stopwords_without_gendered_pronouns = stopwords - gendered_pronouns

def print_list(l, K=None):
	# If K is given then only print first K
	for i, e in enumerate(l):
		if i == K:
			break
		print(e)
	print()

def log_list(l, K=None):
	# If K is given then only log first K
	for i, e in enumerate(l):
		if i == K:
			break
		logging.info(e)
	logging.info("")

def print_dict(d, K=None):
	# If K is given only log first K
	for i, key in enumerate(d.keys()):
		if i == K:
			break
		print(f"{key}:\t{d[key]}")
	print("")

def log_dict(d, K=None):
	# If K is given only log first K
	for i, key in enumerate(d.keys()):
		if i == K:
			break
		logging.info(f"{key}:\t{d[key]}")
	logging.info("")

def save_in_pickle(save_object, save_file):
	with open(save_file, "wb") as pickle_out:
		pickle.dump(save_object, pickle_out)

def load_from_pickle(pickle_file):
	with open(pickle_file, "rb") as pickle_in:
		return pickle.load(pickle_in)

# Ref: https://pynative.com/python-serialize-numpy-ndarray-into-json/
from json import JSONEncoder

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def save_in_json(save_dict, save_file):
	with open(save_file, 'w') as fp:
		json.dump(save_dict, fp, cls=NumpyArrayEncoder)

def load_from_json(json_file):
	with open(json_file, 'r') as fp:
		return json.load(fp)

def save_in_jsonl(list_of_dicts, save_file, append=False):
	if append:
		mode = "a"
	else:
		mode = "w"
	with open(save_file, mode, encoding="utf-8", errors="ignore") as writer:
		for save_dict in list_of_dicts:
			writer.write(f"{json.dumps(save_dict, cls=NumpyArrayEncoder)}\n")

def load_from_jsonl(jsonl_file, max_lines=None):
	with open(jsonl_file, "r") as reader:
		if max_lines is None:
			json_list = [json.loads(line) for line in reader]
		else:
			json_list = [json.loads(next(reader)) for _ in range(max_lines)]
	return json_list

def save_in_txt(list_of_strings, save_file, no_strip=False):
	with open(save_file, "w") as writer:
		for line in list_of_strings:
			if not no_strip:
				line = line.strip()
			writer.write(f"{line}\n")

def load_from_txt(txt_file):
	with open(txt_file, "r") as reader:
		all_lines = list()
		for line in reader:
			line = line.strip()
			all_lines.append(line)
		return all_lines

def save_list_of_tuples_to_tsv(list_of_tuples, save_file, header=None, delimiter='\t'):
	with open(save_file, "w") as tsv_out:
		writer = csv.writer(tsv_out, delimiter=delimiter)
		if header:
			writer.writerow(header)
		for row in list_of_tuples:
			writer.writerow(row)
		tsv_out.flush()
		tsv_out.close()

def load_from_tsv_to_list_of_list(tsv_file, delimiter="\t", header_present=False, encoding="utf-8", ignore_quotes=False):
	# Load the TSV into a list of list
	all_rows = list()
	with open(tsv_file, "r", encoding=encoding) as tsv_in:
		if ignore_quotes:
			reader = csv.reader(tsv_in, delimiter=delimiter, quoting=csv.QUOTE_NONE)
		else:
			reader = csv.reader(tsv_in, delimiter=delimiter)
		if header_present:
			header = next(reader)
		all_rows = [row for row in reader]
	
	if header_present:
		return all_rows, header
	return all_rows

def make_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		logging.info("Creating new directory: {}".format(directory))
		os.makedirs(directory)

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))

def plot_train_loss(loss_trajectory_per_epoch, trajectory_file):
	plt.cla()
	plt.clf()

	fig, ax = plt.subplots()
	x = [epoch * len(loss_trajectory) + j + 1 for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
	x_ticks = [ "(" + str(epoch + 1) + "," + str(j + 1) + ")" for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory) ]
	full_train_trajectory = [avg_loss for epoch, loss_trajectory in enumerate(loss_trajectory_per_epoch) for j, avg_loss in enumerate(loss_trajectory)]
	ax.plot(x, full_train_trajectory)

	ax.set(xlabel='Epoch, Step', ylabel='Loss',
			title='Train loss trajectory')
	step_size = 100
	ax.xaxis.set_ticks(range(0, len(x_ticks), step_size), x_ticks[::step_size])
	ax.grid()

	fig.savefig(trajectory_file)

# def draw_and_save_precision_recall_curve(scores, labels, title, label, save_file, pos_label=None):
# 	plt.cla()
# 	plt.clf()
# 	# Compute the average_precision_score
# 	average_precision = metrics.average_precision_score(labels, scores)
# 	precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=pos_label)
# 	plt.plot(recall, precision, marker='.', label=label)
# 	# axis labels
# 	plt.xlabel('Recall')
# 	plt.ylabel('Precision')
# 	plt.title(title)
# 	# show the legend
# 	plt.legend()
# 	# save the plot
# 	plt.savefig(save_file, dpi=300)
# 	plt.cla()
# 	plt.clf()
# 	return precision, recall, thresholds

def get_number_of_lines(file):
	# Ref: https://stackoverflow.com/a/1019572/4535284
	return sum(1 for line in open(file, "r"))

def write_list_to_file(l, file):
	with open(file, "w") as writer:
		for e in l:
			writer.write(f"{e}\n")

def log_TP_FP_FN_TN_from_binary_predictions(predictions, labels, instances, K=10):
	# Given binary predictions, gold labels and instances we will find instances that are TP, FP, FN and TN
	# Then we will log a sample of K instances from each category for verification
	categories = ["TP", "FP", "FN", "TN"]
	category_explanations = {"TP": "prediction = 1 and label = 1", "FP": "prediction = 1 and label = 0", "FN": "prediction = 0 and label = 1", "TN": "prediction = 0 and label = 0"}
	category_instances = {category:list() for category in categories}
	for prediction, label, instance in zip(predictions, labels, instances):
		if prediction == 1 and label == 1:
			# TP
			category_instances["TP"].append(instance)
		elif prediction == 1 and label == 0:
			# FP
			category_instances["FP"].append(instance)
		elif prediction == 0 and label == 1:
			# FN
			category_instances["FN"].append(instance)
		elif prediction == 0 and label == 0:
			# TN
			category_instances["TN"].append(instance)
		else:
			# Incorrect prediction or label
			logging.error(f"Incorrect prediction({prediction}) or label({label})")
			exit(1)
	# Log a sample form each category
	for category in categories:
		logging.info(f"{category}:{category_explanations[category]}:A sample of {K}/{len(category_instances[category])} instances:")
		category_sample = random.sample(category_instances[category], K)
		log_list(category_sample)

def get_ngrams_from_sentence(sent, n=1, lowercase=True):
	words = sent.strip().split()
	if lowercase:
		words = [word.lower() for word in words]
	ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
	return ngrams

def get_ngram_freq_from_corpus(sents, n=1, min_threshold=5, lowercase=True):
	# From list of sentences we will extract all ngrams along with their frequencies and return them in a dict
	ngram_freq = dict()
	for sent in sents:
		current_sent_ngrams = get_ngrams_from_sentence(sent, n, lowercase)
		for ngram in current_sent_ngrams:
			# Update freq of ngram
			ngram_freq.setdefault(ngram, 0)
			ngram_freq[ngram] += 1
	# Filter by min_threshold
	ngrams_to_remove = [ngram for ngram, count in ngram_freq.items() if count <= min_threshold]
	for ngram in ngrams_to_remove:
		del ngram_freq[ngram]
	return ngram_freq

def remove_stopwords_from_vocab(vocab):
	to_remove = list()
	for word in vocab:
		if word in stopwords:
			to_remove.append(word)
	for word in to_remove:
		del vocab[word]

def normalize_vocab(vocab, total=None, remove_stop_words=True):
	if not total:
		total = len(vocab)
	total = float(total)
	new_vocab = dict()
	for key in vocab:
		new_vocab[key] = float(vocab[key])/total
	if remove_stop_words:
		keys = list(new_vocab.keys())
		for key in keys:
			if type(key) == tuple and len(key) == 1:
				word_key = key[0]
			if word_key in stopwords_without_gendered_pronouns:
				del new_vocab[key]
	return new_vocab

def get_num_of_word_in_corpus(sents):
	return sum(len(sent.split()) for sent in sents)

def get_cuda_memory_usage():
	memory_allocated = torch.cuda.memory_allocated(0)/1024/1024/1024
	memory_reserved = torch.cuda.memory_reserved(0)/1024/1024/1024
	max_memory_reserved = torch.cuda.max_memory_reserved(0)/1024/1024/1024
	return memory_allocated, memory_reserved, max_memory_reserved

