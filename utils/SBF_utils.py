from .utils import RANDOM_SEED, log_list, print_list, make_dir_if_not_exists, save_in_pickle, load_from_pickle, save_in_json, load_from_json, format_time, plot_train_loss

from transformers import  GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config, BertTokenizer, BertPreTrainedModel, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch

import os
import re
import math
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import metrics

import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class SBF_GPT2_Dataset(Dataset):
	"""SBF_GPT2_Dataset stores the SBF instances. It takes list of strings as input"""
	def __init__(self, instances):
		super(SBF_GPT2_Dataset, self).__init__()
		self.instances = instances
		self.nsamples = len(self.instances)

	def __getitem__(self, index):
		return self.instances[index]

	def __len__(self):
		return self.nsamples

def count_unique_posts(df):
	n_unique_posts = df['post'].nunique()
	df_shape = df.shape
	return n_unique_posts, df_shape

def binarize_labels(l):
	avg = sum(l) / len(l)
	return 1 if avg >= 0.5 else 0

def convert_binarized_label_to_string(label, task):
	if task == "offend":
		return "[offY]" if label == 1 else "[offN]"
	if task == "intend":
		return "[intY]" if label == 1 else "[intN]"
	if task == "lewd":
		return "[lewdY]" if label == 1 else "[lewdN]"
	if task == "group":
		return "[grpY]" if label == 1 else "[grpN]"
	if task == "in_group":
		return "[ingY]" if label == 1 else "[ingN]"
	else:
		logging.error(f"Unknown task {task}")

def convert_string_label_to_binary(label):
	if label.startswith("[off"):
		return 1 if label == "[offY]" else 0
	if label.startswith("[int"):
		return 1 if label == "[intY]" else 0
	if label.startswith("[lewd"):
		return 1 if label == "[lewdY]" else 0
	if label.startswith("[grp"):
		return 1 if label == "[grpY]" else 0
	if label.startswith("[ing"):
		return 1 if label == "[ingY]" else 0
	else:
		logging.error(f"Unkown label = {label}")
		return 0

def relabel_with_binarized_votes_and_create_GPT2_instances(df):
	"""
	We will create LM instances for GPT-2 training with average binarized vote labels for 5 classification tasks instead of the original labels
	SBF dataframe's columns are
	- whoTarget: group vs. individual target
	- intentYN: was the intent behind the statement to offend
	- sexYN: is the post a sexual or lewd reference
	- sexReason: free text explanations of what is sexual
	- offensiveYN: could the post be offensive to anyone
	- annotatorGender: gender of the MTurk worker 
	- annotatorMinority: whether the MTurk worker identifies as a minority
	- sexPhrase: part of the post that references something sexual
	- speakerMinorityYN: whether the speaker was part of the same minority group that's being targeted
	- WorkerId: hashed version of the MTurk workerId
	- HITId: id that uniquely identifies each post
	- annotatorPolitics: political leaning of the MTurk worker
	- annotatorRace: race of the MTurk worker
	- annotatorAge: age of the MTurk worker
	- post: post that was annotated
	- targetMinority: demographic group targeted
	- targetCategory: high-level category of the demographic group(s) targeted
	- targetStereotype: implied statement

	We are most interested in post, sexYN, offensiveYN, intentYN, whoTarget, targetMinority, targetStereotype, speakerMinorityYN
	The authors add two task-specific vocabulary items for each of our five classification tasks (w[lewd] ,w[off] ,w[int] , w[grp] , w[ing]), 
	each representing the negative and positive values of the class (e.g., for offensiveYN, `[offY]` and `[offN]`)
	"""
	instances = set()
	all_unqiue_post_offend_labels = list()
	all_unqiue_post_intend_labels = list()
	all_unqiue_post_lewd_labels = list()
	all_unqiue_post_group_labels = list()
	all_unqiue_post_in_group_labels = list()
	for post, post_group_df in df.groupby("post"):
		offensiveYN = post_group_df['offensiveYN'].tolist()
		sexYN = post_group_df['sexYN'].tolist()
		intentYN = post_group_df['intentYN'].tolist()
		whoTarget = post_group_df['whoTarget'].tolist()
		targetMinority = post_group_df['targetMinority'].tolist()
		targetStereotype = post_group_df['targetStereotype'].tolist()
		speakerMinorityYN = post_group_df['speakerMinorityYN'].tolist()
		# print(f"post:{post}")
		# Verify majority label for every array
		# print(f"offensive:{offensiveYN}")
		offend_label = binarize_labels(offensiveYN)
		all_unqiue_post_offend_labels.append(offend_label)
		offend_label = convert_binarized_label_to_string(offend_label, "offend")
		# print(f"intent:{intentYN}")
		intend_label = binarize_labels(intentYN)
		all_unqiue_post_intend_labels.append(intend_label)
		intend_label = convert_binarized_label_to_string(intend_label, "intend")
		# print(f"lewd:{sexYN}")
		lewd_label = binarize_labels(sexYN)
		all_unqiue_post_lewd_labels.append(lewd_label)
		lewd_label = convert_binarized_label_to_string(lewd_label, "lewd")
		# print(f"group:{whoTarget}")
		group_label = binarize_labels(whoTarget)
		all_unqiue_post_group_labels.append(group_label)
		group_label = convert_binarized_label_to_string(group_label, "group")
		# print(f"in-group:{speakerMinorityYN}")
		in_group_label = binarize_labels(speakerMinorityYN)
		all_unqiue_post_in_group_labels.append(in_group_label)
		in_group_label = convert_binarized_label_to_string(in_group_label, "in_group")
		# print(f"group targeted:")
		# print_list(targetMinority)
		# print(f"implied statement:")
		# print_list(targetStereotype)

		# Create instances with the new labels
		for group_targeted, implied_statement in zip(targetMinority, targetStereotype):
			if type(group_targeted) != str and math.isnan(group_targeted):
				group_targeted = ""
			if type(implied_statement) != str and math.isnan(implied_statement):
				implied_statement = ""
			try:
				assert type(group_targeted) == str and type(implied_statement) == str
			except AssertionError:
				logging.error(f"Group targeted is not a string! {group_targeted}")
				logging.error(f"Implied Statement is not a string! {implied_statement}")
				exit()
			instance = f"[STR] {post} [SEP] {lewd_label} {offend_label} {intend_label} {group_label} [SEP] {group_targeted} [SEP] {implied_statement} [SEP] {in_group_label}"
			instances.add(instance)
	return list(instances), Counter(all_unqiue_post_offend_labels), Counter(all_unqiue_post_intend_labels), \
		Counter(all_unqiue_post_lewd_labels), Counter(all_unqiue_post_group_labels), Counter(all_unqiue_post_in_group_labels)

def get_labels_dict_from_string(s):
	try:
		lewd_off_int_grp, group_targeted, implied_statement, ing = s.strip().split("[SEP]")
	except ValueError:
		logging.error(f"Failed in [SEP] split. s = {s}")
		return {"offend": "[offN]", "intend": "[intN]", "lewd": "[lewdN]", "group": "[grpN]", "in-group": "[ingN]", "group-targeted": "", "implied-statement": ""}

	ing = ing.strip()
	try:
		lewd, off, intend, grp = lewd_off_int_grp.strip().split()
	except ValueError:
		logging.error(f"Failed in labels split. s = {s}")
		logging.error(f"lewd_off_int_grp = {lewd_off_int_grp}")
		return {"offend": "[offN]", "intend": "[intN]", "lewd": "[lewdN]", "group": "[grpN]", "in-group": "[ingN]", "group-targeted": "", "implied-statement": ""}
	return {"offend": off, "intend": intend, "lewd": lewd, "group": grp, "in-group": ing, "group-targeted": group_targeted, "implied-statement": implied_statement}

def get_labels_dict_from_list(l):
	# Given a list of labels in the order: offend_label, intend_label, lewd_label, group_label, in_group_label
	# Convert them into a dictionary recognizable by the code
	return {"offend": l[0], "intend": l[1], "lewd": l[2], "group": l[3], "in-group": l[4], "group-targeted": "N/A", "implied-statement": "N/A"}

# BERT model experiments

class SBF_BERT_Dataset(Dataset):
	"""SBF_BERT_Dataset stores the SBF instances. It takes list of tuples, where the first element of the tuple is the string and 
		second element is tuple of classification labels"""
	def __init__(self, instances):
		super(SBF_BERT_Dataset, self).__init__()
		self.instances = instances
		self.nsamples = len(self.instances)

	def __getitem__(self, index):
		return self.instances[index]

	def __len__(self):
		return self.nsamples

def relabel_with_binarized_votes_and_create_BERT_instances(df):
	"""
	We will create instances for BERT training with average binarized vote labels for 5 classification tasks instead of the original labels
	SBF dataframe's columns are
	- whoTarget: group vs. individual target
	- intentYN: was the intent behind the statement to offend
	- sexYN: is the post a sexual or lewd reference
	- sexReason: free text explanations of what is sexual
	- offensiveYN: could the post be offensive to anyone
	- annotatorGender: gender of the MTurk worker 
	- annotatorMinority: whether the MTurk worker identifies as a minority
	- sexPhrase: part of the post that references something sexual
	- speakerMinorityYN: whether the speaker was part of the same minority group that's being targeted
	- WorkerId: hashed version of the MTurk workerId
	- HITId: id that uniquely identifies each post
	- annotatorPolitics: political leaning of the MTurk worker
	- annotatorRace: race of the MTurk worker
	- annotatorAge: age of the MTurk worker
	- post: post that was annotated
	- targetMinority: demographic group targeted
	- targetCategory: high-level category of the demographic group(s) targeted
	- targetStereotype: implied statement

	We are most interested in post, sexYN, offensiveYN, intentYN, whoTarget, targetMinority, targetStereotype, speakerMinorityYN
	The authors add two task-specific vocabulary items for each of our five classification tasks (w[lewd] ,w[off] ,w[int] , w[grp] , w[ing]), 
	each representing the negative and positive values of the class (e.g., for offensiveYN, `[offY]` and `[offN]`)
	Instead for BERT model we directly classify them using a linear classifier. We skip predicting targetMinory and targetStereotype
	"""
	instances = set()
	all_unqiue_post_offend_labels = list()
	all_unqiue_post_intend_labels = list()
	all_unqiue_post_lewd_labels = list()
	all_unqiue_post_group_labels = list()
	all_unqiue_post_in_group_labels = list()
	for post, post_group_df in df.groupby("post"):
		offensiveYN = post_group_df['offensiveYN'].tolist()
		sexYN = post_group_df['sexYN'].tolist()
		intentYN = post_group_df['intentYN'].tolist()
		whoTarget = post_group_df['whoTarget'].tolist()
		targetMinority = post_group_df['targetMinority'].tolist()
		targetStereotype = post_group_df['targetStereotype'].tolist()
		speakerMinorityYN = post_group_df['speakerMinorityYN'].tolist()
		# Compute labels like the technique mentioned in the paper
		offend_label = binarize_labels(offensiveYN)
		all_unqiue_post_offend_labels.append(offend_label)
		intend_label = binarize_labels(intentYN)
		all_unqiue_post_intend_labels.append(intend_label)
		lewd_label = binarize_labels(sexYN)
		all_unqiue_post_lewd_labels.append(lewd_label)
		group_label = binarize_labels(whoTarget)
		all_unqiue_post_group_labels.append(group_label)
		in_group_label = binarize_labels(speakerMinorityYN)
		all_unqiue_post_in_group_labels.append(in_group_label)

		# Create instances with the new labels
		# Since we don't care about targeted_group or impiled_statement, we will simply create a new instance for each unique post
		instance = post
		labels = (offend_label, intend_label, lewd_label, group_label, in_group_label)
		instances.add((instance, labels))
	return list(instances), Counter(all_unqiue_post_offend_labels), Counter(all_unqiue_post_intend_labels), \
		Counter(all_unqiue_post_lewd_labels), Counter(all_unqiue_post_group_labels), Counter(all_unqiue_post_in_group_labels)

class BertForSBF(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		self.SBF_tasks = config.SBF_tasks
		# We will create a list of classifiers based on the number of SBF_tasks
		self.classifiers = [nn.Linear(config.hidden_size, config.num_labels) for t in self.SBF_tasks]

		self.init_weights()

	def forward(
		self,
		input_ids,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
			Labels for computing the sequence classification/regression loss.
			Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
			If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
			If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

	Returns:
		:obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
		loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
			Classification (or regression if config.num_labels==1) loss.
		logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
			Classification (or regression if config.num_labels==1) scores (before SoftMax).
		hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
			Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
			of shape :obj:`(batch_size, sequence_length, hidden_size)`.

			Hidden-states of the model at the output of each layer plus the initial embedding outputs.
		attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
			Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
			:obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

			Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
			heads.

	Examples::

		from transformers import BertTokenizer, BertForSequenceClassification
		import torch

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

		input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
		labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
		outputs = model(input_ids, labels=labels)

		loss, logits = outputs[:2]

		"""

		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		# DEBUG:
		# print("BERT model outputs shape", outputs[0].shape, outputs[1].shape)
		
		# OLD CODE:
		pooled_output = outputs[1]

		pooled_output = self.dropout(pooled_output)
		# Get logits for each subtask
		logits = [self.classifiers[i](pooled_output) for i in range(len(self.SBF_tasks))]
		
		# TODO: Debug from here!
		# pdb.set_trace()

		outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			
			# DEBUG:
			# print(f"Logits:{logits.view(-1, self.num_labels)}, \t, Labels:{labels.view(-1)}")
			for i, SBF_task in enumerate(self.SBF_tasks):
				if i == 0:
					loss = loss_fct(logits[i].view(-1, self.num_labels), labels[i].view(-1))
				else:
					loss += loss_fct(logits[i].view(-1, self.num_labels), labels[i].view(-1))
			outputs = (loss,) + outputs

		return outputs  # (loss), logits, (hidden_states), (attentions)


# Analysis Util Functions
def get_label_distributions_and_predictions_from_scores_and_thresholds(task_prediction_scores, task_thresholds):
	n_tasks = len(task_prediction_scores)
	task_distributions = [None] * n_tasks
	task_predictions = [None] * n_tasks
	for i in range(n_tasks):
		prediction_scores = task_prediction_scores[i]
		threshold = task_thresholds[i]
		predictions = [1 if e >= threshold else 0 for e in prediction_scores]
		task_predictions[i] = predictions
		task_distributions[i] = Counter(predictions)
	return task_distributions, task_predictions


def relabel_with_binarized_votes_and_create_tuple_instances(df):
	"""
	We will create instances for manual analysis of SBF train with average binarized vote labels for 5 classification tasks instead of the original labels
	SBF dataframe's columns are
	- whoTarget: group vs. individual target
	- intentYN: was the intent behind the statement to offend
	- sexYN: is the post a sexual or lewd reference
	- sexReason: free text explanations of what is sexual
	- offensiveYN: could the post be offensive to anyone
	- annotatorGender: gender of the MTurk worker 
	- annotatorMinority: whether the MTurk worker identifies as a minority
	- sexPhrase: part of the post that references something sexual
	- speakerMinorityYN: whether the speaker was part of the same minority group that's being targeted
	- WorkerId: hashed version of the MTurk workerId
	- HITId: id that uniquely identifies each post
	- annotatorPolitics: political leaning of the MTurk worker
	- annotatorRace: race of the MTurk worker
	- annotatorAge: age of the MTurk worker
	- post: post that was annotated
	- targetMinority: demographic group targeted
	- targetCategory: high-level category of the demographic group(s) targeted
	- targetStereotype: implied statement

	We are most interested in post, sexYN, offensiveYN, intentYN, whoTarget, targetMinority, targetStereotype, speakerMinorityYN
	The authors add two task-specific vocabulary items for each of our five classification tasks (w[lewd] ,w[off] ,w[int] , w[grp] , w[ing]), 
	each representing the negative and positive values of the class (e.g., for offensiveYN, `[offY]` and `[offN]`)
	Instead for BERT model we directly classify them using a linear classifier. We skip predicting targetMinory and targetStereotype
	"""
	instances = set()
	all_unqiue_post_offend_labels = list()
	all_unqiue_post_intend_labels = list()
	all_unqiue_post_lewd_labels = list()
	all_unqiue_post_group_labels = list()
	all_unqiue_post_in_group_labels = list()
	for post, post_group_df in df.groupby("post"):
		offensiveYN = post_group_df['offensiveYN'].tolist()
		sexYN = post_group_df['sexYN'].tolist()
		intentYN = post_group_df['intentYN'].tolist()
		whoTarget = post_group_df['whoTarget'].tolist()
		targetMinority = post_group_df['targetMinority'].tolist()
		targetStereotype = post_group_df['targetStereotype'].tolist()
		speakerMinorityYN = post_group_df['speakerMinorityYN'].tolist()
		# Compute labels like the technique mentioned in the paper
		offend_label = binarize_labels(offensiveYN)
		all_unqiue_post_offend_labels.append(offend_label)
		intend_label = binarize_labels(intentYN)
		all_unqiue_post_intend_labels.append(intend_label)
		lewd_label = binarize_labels(sexYN)
		all_unqiue_post_lewd_labels.append(lewd_label)
		group_label = binarize_labels(whoTarget)
		all_unqiue_post_group_labels.append(group_label)
		in_group_label = binarize_labels(speakerMinorityYN)
		all_unqiue_post_in_group_labels.append(in_group_label)

		# Create instances with the new labels
		# Since we don't care about targeted_group or impiled_statement, we will simply create a new instance for each unique post
		instance = post
		labels = (offend_label, intend_label, lewd_label, group_label, in_group_label)
		for minority in set(targetMinority):
			instances.add((instance, offend_label, intend_label, lewd_label, minority, group_label, in_group_label))
	return list(instances), Counter(all_unqiue_post_offend_labels), Counter(all_unqiue_post_intend_labels), \
		Counter(all_unqiue_post_lewd_labels), Counter(all_unqiue_post_group_labels), Counter(all_unqiue_post_in_group_labels)


def relabel_with_binarized_votes_and_create_tuple_instances_with_aggregate_stereotypes(df):
	"""
	We will create instances for manual analysis of SBF train steretypes data with average binarized vote labels for 5 classification tasks instead of the original labels
	SBF dataframe's columns are
	- whoTarget: group vs. individual target
	- intentYN: was the intent behind the statement to offend
	- sexYN: is the post a sexual or lewd reference
	- sexReason: free text explanations of what is sexual
	- offensiveYN: could the post be offensive to anyone
	- annotatorGender: gender of the MTurk worker 
	- annotatorMinority: whether the MTurk worker identifies as a minority
	- sexPhrase: part of the post that references something sexual
	- speakerMinorityYN: whether the speaker was part of the same minority group that's being targeted
	- WorkerId: hashed version of the MTurk workerId
	- HITId: id that uniquely identifies each post
	- annotatorPolitics: political leaning of the MTurk worker
	- annotatorRace: race of the MTurk worker
	- annotatorAge: age of the MTurk worker
	- post: post that was annotated
	- targetMinority: demographic group targeted
	- targetCategory: high-level category of the demographic group(s) targeted
	- targetStereotype: implied statement
	We are most interested in post, sexYN, offensiveYN, intentYN, whoTarget, targetMinority, targetStereotype, speakerMinorityYN
	The authors add two task-specific vocabulary items for each of our five classification tasks (w[lewd] ,w[off] ,w[int] , w[grp] , w[ing]), 
	each representing the negative and positive values of the class (e.g., for offensiveYN, `[offY]` and `[offN]`)
	Instead for BERT model we directly classify them using a linear classifier. We skip predicting targetMinory and targetStereotype
	"""
	instances = list()
	all_unqiue_post_offend_labels = list()
	all_unqiue_post_intend_labels = list()
	all_unqiue_post_lewd_labels = list()
	all_unqiue_post_group_labels = list()
	all_unqiue_post_in_group_labels = list()
	for post, post_group_df in df.groupby("post"):
		offensiveYN = post_group_df['offensiveYN'].tolist()
		sexYN = post_group_df['sexYN'].tolist()
		intentYN = post_group_df['intentYN'].tolist()
		whoTarget = post_group_df['whoTarget'].tolist()
		targetMinority = post_group_df['targetMinority'].tolist()
		targetStereotype = post_group_df['targetStereotype'].tolist()
		speakerMinorityYN = post_group_df['speakerMinorityYN'].tolist()
		dataSource = list(set(post_group_df['dataSource'].tolist()))
		assert len(dataSource) == 1
		dataSource = dataSource[0]
		# Compute labels like the technique mentioned in the paper
		offend_label = binarize_labels(offensiveYN)
		all_unqiue_post_offend_labels.append(offend_label)
		intend_label = binarize_labels(intentYN)
		all_unqiue_post_intend_labels.append(intend_label)
		lewd_label = binarize_labels(sexYN)
		all_unqiue_post_lewd_labels.append(lewd_label)
		group_label = binarize_labels(whoTarget)
		all_unqiue_post_group_labels.append(group_label)
		in_group_label = binarize_labels(speakerMinorityYN)
		all_unqiue_post_in_group_labels.append(in_group_label)

		# Create instances with the new labels
		# Since we don't care about targeted_group or impiled_statement, we will simply create a new instance for each unique post
		instance = re.sub('\s+', ' ', post.strip())
		labels = (offend_label, intend_label, lewd_label, group_label, in_group_label)
		for minority, stereotype in set(zip(targetMinority, targetStereotype)):
			instances.append((instance, minority, stereotype, offend_label, intend_label, lewd_label, group_label, in_group_label, dataSource))
	return instances, Counter(all_unqiue_post_offend_labels), Counter(all_unqiue_post_intend_labels), \
		Counter(all_unqiue_post_lewd_labels), Counter(all_unqiue_post_group_labels), Counter(all_unqiue_post_in_group_labels)

