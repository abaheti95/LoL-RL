'''Classification code for:

Symbolic Knowledge Distillation: from General Language Models to Commonsense Models
Peter West, Chandra Bhagavatula, Jack Hessel, Jena D. Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, Yejin Choi
https://arxiv.org/abs/2110.07178

This code contains the code for training the purification model described in Sec. 4 of the above paper.

The best hyperparameter settings are set as defaults for this script:
our search is described in more detail in the paper.

So, to get started, you could just run:

python classify.py purification_dataset.jsonl --model roberta-large-mnli

This script will save a model that can then be applied to unlabelled data with predict.py
'''
import numpy as np
import names #pip install names
import ftfy

_RELATIONS = {
    'HinderedBy':'can be hindered by',
    'xNeed':'but before, PersonX needed',
    'xWant':'as a result, PersonX wants',
    'xIntent':'because PersonX wanted',
    'xReact':'as a result, PersonX feels',
    'xAttr':'so, PersonX is seen as',
    'xEffect':'as a result, PersonX'
}



def _sep_pair_with_name(cur):
    p1 = names.get_first_name()
    p2 = p1
    while p2 == p1:
        p2 = names.get_first_name()
    cur = cur.replace('PersonX', p1)
    cur = cur.replace('PersonY', p2)
    cur = ftfy.fix_text(cur)
    cur = cur.split('**SEP**')
    return cur

def _to_string_main(x):
    ''' NLI flatten, but with names. '''
    if x['relation'] not in _RELATIONS:
        breakpoint()
    cur = '{}**SEP**{} {}'.format(x['head'], _RELATIONS[x['relation']], x['tail'])
    return _sep_pair_with_name(cur)

def get_comet_keras_input_and_labels(batch_dicts, tokenizer):
    labels = np.array([1.0 if b['valid'] > 0 else 0.0 for b in batch_dicts])
    texts = [_to_string_main(b) for b in batch_dicts]
    text_X = tokenizer(texts, return_tensors='pt', padding=True)
    return text_X, labels

import torch
from torch import nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, middle_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, middle_size)
        # self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(middle_size, num_labels)

    def forward(self, features):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        
        # We already get output from pooler so may not need to do this
        # features is of shape torch.Size([1, 1024])
        x = features
        # x = self.dropout(x)
        x = self.dense(x)
        # x.shape = torch.Size([1, 512])
        # Use Gelu activation function
        x = torch.nn.functional.gelu(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x.shape = torch.Size([1, 512])
        x = self.out_proj(x)
        # x.shape = torch.Size([1, 1])
        # Use sigmoid activation function
        x = torch.sigmoid(x)
        return x
    
from transformers import StoppingCriteria

class CustomStopTokenStoppingCritera(StoppingCriteria):
        def __init__(self, stop_token_id):
            self.stop_token_id = stop_token_id
            # self.tokenizer = tokenizer
        
        def __call__(self, input_ids, scores):
            # Check if each input_ids has stop_token_id in it
            breakpoint()
            batch_stop_token_id_present = [self.stop_token_id in input_id for input_id in input_ids]
            # batch_stop_token_id_present = [50264 in input_id for input_id in input_ids]
            return all(batch_stop_token_id_present)

def get_period_stopping_critera(tokenizer):
    #  13
    stop_token_id = tokenizer.convert_tokens_to_ids(".")
    # Define stopping criteria for edit model
    return CustomStopTokenStoppingCritera(stop_token_id)

