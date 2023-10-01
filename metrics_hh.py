import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import reward_model
import nltk

def get_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

def create_reward_fn_2(reward_batch_size=8):
    model_name = "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1"
    model_device = "cuda:{}".format(torch.cuda.device_count() - 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
    reward_model.eval()

    def get_score(prefixes, suffixes):
        texts = []
        
        for p, s in zip(prefixes, suffixes):
            assert p[-1] == "<|prompter|>" or p[-1] == "<|assistant|>", p[-1]
            temp_prefix = p[:-1] + [p[-1]+s]
            texts.append("".join([t + tokenizer.eos_token for t in temp_prefix]))
        
        input_content = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            rewards = reward_model(**input_content).logits
        torch.cuda.empty_cache()
        return rewards.view(-1)

    return get_score, reward_batch_size

def create_reward_fn_3(ds_plugin=None, reward_batch_size=32):
    model_name = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
    model_device = "cuda:{}".format(torch.cuda.device_count() - 1)
    print(f"Using reward model device = {model_device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    if ds_plugin is not None:
        with ds_plugin.zero3_init_context_manager(enable=False):
            reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
            reward_model.eval()
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
        reward_model.eval()

    def get_score(prefixes, suffixes):
        texts = []

        for p, s in zip(prefixes, suffixes):
            assert p[-1] == "<|prompter|>" or p[-1] == "<|assistant|>", p[-1]
            temp_prefix = p[:-1] + [p[-1]+s]
            texts.append("".join([t + tokenizer.eos_token for t in temp_prefix]))
        input_content = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            if ds_plugin is not None:
                with ds_plugin.zero3_init_context_manager(enable=False):
                    rewards = reward_model(**input_content).logits
            else:
                rewards = reward_model(**input_content).logits
        
        return rewards.view(-1)
    
    return get_score, reward_batch_size

create_reward_fn = create_reward_fn_3