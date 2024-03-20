# Following code has been copied directly from : https://raw.githubusercontent.com/lvwerra/trl/237eb9c6a5b1f90f6ccb674269bd7a33533e4bf7/trl/models/modeling_value_head.py

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AdamW

# from trl.models.modeling_base import PreTrainedModelWrapper
import torch.nn.functional as F
from .utils import logits_to_entropy, log_list
import pandas as pd
import numpy as np
import numba as nb
import logging
logging.basicConfig(level=logging.INFO)


# Data preprocessing related scripts
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_reward_components_distribution(reward_components_list, segment_name, args, figsize=(10, 9), rotation=70, color=None):
    components = list()
    rewards = list()
    for reward_components in reward_components_list:
        for component, reward in reward_components.items():
            components.append(component)
            rewards.append(reward)
    df = pd.DataFrame({"components": components, "reward": rewards})
    # Plot violin plot
    # Set figure size
    plt.figure(figsize=figsize)
    violin_plot = sns.violinplot(data=df, x="components", y="reward", color=color)
    violin_plot.set(xlabel="Reward Components", ylabel="Reward distribution")
    # Add total counts/percentage of instances for each threshold window
    xticklabels = list(reward_components_list[0].keys())
    # violin_plot.set_xticklabels(violin_plot.get_xticklabels(), rotation=90)
    violin_plot.set_xticklabels(xticklabels, rotation=rotation)
    violin_plot.set_title(f"Per component reward distribution for segment = {segment_name}, containing {len(reward_components_list)} instances")
    violin_plot_save_file = os.path.join(args.output_dir, f"per_component_reward_distribution_plot_{segment_name}.png")
    # Tight layout
    plt.tight_layout()
    violin_plot.figure.savefig(violin_plot_save_file, dpi=300)
    logging.info(f"Saved violin plot to {violin_plot_save_file}")
    plt.clf()
    plt.cla()

# Ref: https://stackoverflow.com/a/64136551/4535284
@nb.njit
def numba_choice(population, wc, k):
    # Get cumulative weights
    # wc = np.cumsum(weights)
    # Total of weights
    m = wc[-1]
    # Arrays of sample and sampled indices
    sample = np.empty(k, population.dtype)
    sample_idx = np.full(k, -1, np.int32)
    # Sampling loop
    i = 0
    while i < k:
        # Pick random weight value
        r = m * np.random.rand()
        # Get corresponding index
        idx = np.searchsorted(wc, r, side='right')
        # Check index was not selected before
        # If not using Numba you can just do `np.isin(idx, sample_idx)`
        for j in range(i):
            if sample_idx[j] == idx:
                continue
        # Save sampled value and index
        sample[i] = population[idx]
        sample_idx[i] = population[idx]
        i += 1
    return sample


class LanguageGenerationListofDict(Dataset):
    def __init__(self, list_of_dicts):
        self.list_of_dicts = list_of_dicts
        # each dict will at least contain keys = dict_keys(['id', 'prompt_or_input_text', 'references', 'meta_data', 'reward_components'])
        # Can keep more keys if needed
        self.sample_weights = None
        self.sample_probs = None

    def __len__(self):
        return len(self.list_of_dicts)
    
    def set_sample_weights(self, sample_weights):
        self.sample_weights = sample_weights
        self.sample_probs = self.sample_weights / np.sum(self.sample_weights)
        logging.info(f"Total instances = {len(self.sample_probs)}")
        # Keep track of indices and non-zero probs
        self.sample_indices = np.where(self.sample_probs != 0.0)[0]
        logging.info(f"Total instances with non-zero probs = {len(self.sample_indices)}")
        # Also update the sample_probs accordingly
        self.sample_probs = self.sample_probs[self.sample_indices]
        logging.info(f"Total instances with non-zero probs after filtering = {len(self.sample_probs)}")
        self.sample_probs_csum = np.cumsum(self.sample_probs)
        logging.info(f"Total instances with non-zero probs after cumsum = {len(self.sample_probs_csum)}")

    def __getitem__(self, idx):
        if self.sample_probs is not None:
            # sample_idx = np.random.choice(len(self.list_of_dicts), p=self.sample_probs)
            sample_idx = numba_choice(self.sample_indices, self.sample_probs_csum, 1)[0]
            return self.list_of_dicts[sample_idx]
        return self.list_of_dicts[idx]

class LanguageGenerationCollator(object):

    def __init__(self, params, tokenizer):
        self.params = params
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch is a list of dicts with dict_keys(['id', 'prompt_or_input_text', 'references', 'meta_data', 'reward_components'])
        prompts = [datum['prompt_or_input_text'] for datum in batch]
        responses = [datum['references'] for datum in batch]
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        response_inputs = self.tokenizer(text_target=responses, return_tensors="pt", padding=True, truncation=True, max_length=self.params.max_resp_length)
        # with self.tokenizer.as_target_tokenizer():
            # response_inputs = self.tokenizer(responses, return_tensors="pt", padding=True, truncation=True, max_length=self.params.max_resp_length)
            # response_inputs['input_ids'][0]
            
        ids = [datum['id'] for datum in batch]
        extra_data = {'prompts': prompts, 'responses': responses, 'ids': ids, 'batch': batch}
        if self.params.learning_algorithm != "nll":
            if self.params.task_name in ["Xsum", "CNNDailyMail"]:
                # NOTE: not using doc_nli reward right now. For CNN the doc_nli reward is mostly close to 0.
                rewards = [datum['reward_components']['fluency'] + datum['reward_components']['text_sim'] for datum in batch]
                # rewards = [datum['reward_components']['fluency'] + datum['reward_components']['text_sim'] + datum['reward_components']['doc_nli_score'] for datum in batch]
            elif self.params.task_name in ["IWSLT2017EnDe", "IMDBForSeq2Seq", "DailyDialog", "COMET", "WOW", "faithdial", "faithdial_wow", "reddit_pos", "reddit_neg"]:
                rewards = [datum['reward_components']['final_reward'] for datum in batch]
            else:
                breakpoint()
            extra_data["rewards"] = rewards
        return prompt_inputs, response_inputs, extra_data

from datasets import load_metric
from tqdm import tqdm
from copy import deepcopy

def get_model_predictions(dataloader, model, tokenizer, device, reward_args, args):
    metric = load_metric("meteor")
    assert 'task_name' in reward_args, breakpoint()
    task_name = reward_args['task_name']
    # Special stop_tokens for COMET
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(".")] if task_name == "COMET" else [tokenizer.eos_token_id]
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    all_gen_responses = []
    all_gold_responses = []
    all_gen_rewards = []
    all_gold_rewards = []
    all_ids = list()
    counter = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating responses for task {task_name}"):
            prompt_inputs, response_inputs, extra_data = batch
            prompt_inputs = prompt_inputs.to(device)
            response_inputs = response_inputs.to(device)
            
            # Generate the response
            generated_ids = model.generate(input_ids=prompt_inputs["input_ids"], attention_mask=prompt_inputs["attention_mask"], max_new_tokens=args.max_resp_length, num_beams=2, repetition_penalty=2.5, eos_token_id=stop_token_ids)
            # sampled_generated_ids = model.generate(input_ids=prompt_inputs["input_ids"], attention_mask=prompt_inputs["attention_mask"], max_new_tokens=args.max_resp_length, top_p=0.95, eos_token_id=stop_token_ids)
            # Get the generated response
            if args.model_type == "causal":
                generated_responses = tokenizer.batch_decode(generated_ids[:, prompt_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                # tokenizer.convert_ids_to_tokens(generated_ids[0, prompt_inputs["input_ids"].shape[1]:])
                # tokenizer.convert_ids_to_tokens(sampled_generated_ids[0, prompt_inputs["input_ids"].shape[1]:])
            elif args.model_type == "seq2seq":
                generated_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            else:
                logging.error(f"Unknown model type {args.model_type}")
                breakpoint()
            
            # extra_data id a dict with keys: dict_keys(['texts', 'responses', 'batch'])
            gold_responses = extra_data['responses']
            gold_responses = [resp.replace("<|endoftext|>", "") for resp in gold_responses]
            # Get text from prompts
            prompts = extra_data['prompts']
            # Extra data already has gold rewards. No need to calculate them again
            extra_data_batch = extra_data['batch']
            gold_rewards = [datum['reward_components'] for datum in extra_data_batch]
            
            # NOTE: gen_resp and gen_resp will be full text for IMDB task
            texts, gen_resp = get_text_and_responses_from_task_prompts_and_references(task_name, prompts, generated_responses)
            # Get gold and gen rewards
            reward_component_to_fn_map = task_to_reward_fn_map[task_name]
            gen_rewards = get_batch_rewards(texts, gen_resp, reward_args, reward_component_to_fn_map)
            # keep track of all ids
            all_ids.extend(extra_data['ids'])
            # Each element in gold_rewards and gen_rewards is a dict with keys: dict_keys(['fluency', 'text_sim', 'final_reward'])
            all_gen_responses.extend(generated_responses)
            all_gold_responses.extend(gold_responses)
            all_gen_rewards.extend(gen_rewards)
            all_gold_rewards.extend(gold_rewards)
            # logging.info(f"Prompts:")
            # log_list(prompts)
            # logging.info(f"Generated responses:")
            # gen_resp_and_rewards = list(zip(generated_responses, gen_rewards))
            # log_list(gen_resp_and_rewards)
            # breakpoint()
            # logging.info(f"Gold responses:")
            # gold_resp_and_rewards = list(zip(gold_responses, gold_rewards))
            # log_list(gold_resp_and_rewards)
            # counter += 1
            # # TEMP: debugging
            # if counter == 2:
            #     break
    # Compute the meteor metric
    score = metric.compute(predictions=generated_responses, references=gold_responses)
    meteor_score = score['meteor']
    return all_ids, all_gen_responses, all_gold_responses, all_gen_rewards, all_gold_rewards, meteor_score

def train_value_function_on_val_predictions(value_function_model, model, tokenizer, val_dataloader, all_ids, all_gen_responses, all_gen_rewards, args):
    # 1. Create a new optimizer for value head
    device = args.device
    value_fn_optimizer = AdamW([{'params': value_function_model.parameters()}], lr=args.learning_rate)
    n_value_head_epochs = args.a2c_n_value_head_epochs
    logging.info(f"Estimating baseline policy value function for {n_value_head_epochs} epochs on dev set...")
    best_value_mse = float("inf")
    best_value_function_model = None
    best_epoch = None
    value_function_model.train()
    id_to_gen_response_and_rewards = {id_: {"response": gen_resp, "rewards": gen_rewards} for id_, gen_resp, gen_rewards in zip(all_ids, all_gen_responses, all_gen_rewards)}
    model.eval()
    tqdm_flag = True
    for epoch in range(n_value_head_epochs):
        pbar = tqdm(val_dataloader) if tqdm_flag else val_dataloader
        total_loss = 0.0
        total_steps = 0
        model_correct = 0.0
        total_instances = 0.0
        for batch in pbar:
            value_function_model.zero_grad()
            prompt_inputs, response_inputs, extra_data = batch
            input_ids = prompt_inputs["input_ids"].to(device)
            attention_mask = prompt_inputs["attention_mask"].to(device)
            # Get generated responses for current batch
            # extra_data id a dict with keys: dict_keys(['prompts', 'responses', 'ids', 'batch', 'rewards'])
            batch_ids = extra_data['ids']
            batch_gold_responses = extra_data['responses']
            batch_gen_responses = [id_to_gen_response_and_rewards[id_]["response"] for id_ in batch_ids]
            batch_gen_rewards = [id_to_gen_response_and_rewards[id_]["rewards"] for id_ in batch_ids]
            # Tokenize the inputs in the batch and create input_ids and attention_mask for the model
            with tokenizer.as_target_tokenizer():
                tokenizer.padding_side = "right"
                tokenizer.truncation_side = "right"
                gen_response_inputs = tokenizer(batch_gen_responses, return_tensors="pt", padding=True, truncation=True, max_length=args.max_resp_length)
            labels = gen_response_inputs["input_ids"].to(device)

            # Get the last layer prompt side hidden layer for either seq2seqLM or causalLM models
            if args.model_type == "causal":
                # Merge the prompt and response inputs
                input_ids = torch.cat([prompt_inputs["input_ids"], gen_response_inputs["input_ids"]], dim=1).to(device)
                attention_mask = torch.cat([prompt_inputs["attention_mask"], gen_response_inputs["attention_mask"]], dim=1).to(device)
                with torch.no_grad(): outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                batch_size, query_seq_len = prompt_inputs["input_ids"].shape
                logits = outputs.logits[:, (query_seq_len - 1):-1, :]
                last_layer_hidden_state = outputs.hidden_states[-1]
                last_layer_encoder_hidden_state = last_layer_hidden_state[:, :query_seq_len, :]

            elif args.model_type == "seq2seq":
                # Tokenize the inputs in the batch and create input_ids and attention_mask for the model
                # Ref: https://github.com/huggingface/transformers/issues/3021
                input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                # Forward
                with torch.no_grad(): outputs = model(**input_dict, output_hidden_states=True)
                # loss = outputs.loss
                # Get the last layer encoder hidden state
                last_layer_encoder_hidden_state = outputs.encoder_last_hidden_state
            else:
                logging.error(f"Invalid model type: {args.model_type}")
                breakpoint()
            # last_layer_encoder_hidden_state is of the shape (batch_size, seq_len, hidden_size)
            # Get value function predictions for the query
            val_outputs = value_function_model(last_layer_encoder_hidden_state)
            # Get the query target rewards
            if args.task_name in ["Xsum", "CNNDailyMail"]:
                # NOTE: not using doc_nli reward right now. For CNN the doc_nli reward is mostly close to 0.
                final_val_rewards = [gen_reward['fluency'] + gen_reward['text_sim'] for gen_reward in batch_gen_rewards]
            else:
                final_val_rewards = [gen_reward['final_reward'] for gen_reward in batch_gen_rewards]
            val_targets = torch.FloatTensor(final_val_rewards).to(device)
            loss = F.mse_loss(val_outputs, val_targets)

            
            # 4. Backpropagate the loss
            loss.backward()
            value_fn_optimizer.step()
            # 5. Log the loss
            total_loss += loss.item()
            total_steps += 1
            avg_total_loss = total_loss / total_steps
            if tqdm_flag:
                pbar.set_postfix({"step_loss": loss.item(), "avg_total_loss": avg_total_loss})
        if avg_total_loss < best_value_mse:
            best_value_mse = avg_total_loss
            best_value_function_model = deepcopy(value_function_model)
            best_epoch = epoch+1
            logging.info(f"Found new best value head with MSE {best_value_mse} at epoch {best_epoch}")
        logging.info(f"Epoch {epoch+1} finished with MSE {avg_total_loss}")
    model.train()
    # 6. Return the trained best value head
    return best_value_function_model, best_value_mse, best_epoch

def get_advantage_predictions_on_dataset(train_dataset, tokenize_collator, model, best_value_function_model, args):
    device = args.device
    with torch.no_grad():
        all_advantages = list()
        train_dataloader = DataLoader(train_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=tokenize_collator)
        for batch in tqdm(train_dataloader, desc=f"Getting advantage estimates for task {args.task_name}"):
            prompt_inputs, response_inputs, extra_data = batch
            # Set response_inputs as labels for the model
            input_ids = prompt_inputs["input_ids"].to(device)
            attention_mask = prompt_inputs["attention_mask"].to(device)
            labels = response_inputs["input_ids"].to(device)

            # Tokenize the inputs in the batch and create input_ids and attention_mask for the model
            # Ref: https://github.com/huggingface/transformers/issues/3021
            if args.model_type == "seq2seq":
                input_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
                outputs = model(**input_dict, output_hidden_states=True)
                logits = outputs.logits
                last_layer_encoder_hidden_state = outputs.encoder_last_hidden_state
                # Forward
                outputs = model(**input_dict)
                # loss = outputs.loss
                logits = outputs.logits
            elif args.model_type == "causal":
                # Merge the prompt and response inputs
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                batch_size, query_seq_len = prompt_inputs["input_ids"].shape
                logits = outputs.logits[:, (query_seq_len - 1):-1, :]
                last_layer_hidden_state = outputs.hidden_states[-1]
                last_layer_encoder_hidden_state = last_layer_hidden_state[:, :query_seq_len, :]
                
            # Calculate the value function estimates for current batch
            # Get the last layer encoder hidden state
            val_outputs = best_value_function_model(last_layer_encoder_hidden_state)
            # Calculate the advantage from rewards and value function estimates
            rewards = torch.tensor(extra_data['rewards']).to(device)
            advantage = rewards - val_outputs
            all_advantages.extend(advantage.tolist())
        return all_advantages








class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # Value head must be provided with maximum possible value. All rewards are expected to be positive between 0 and 1.
        assert "max_value" in kwargs, breakpoint()
        self.max_value = kwargs["max_value"]

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        # avg sequence length
        output = torch.mean(output, dim=1)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output).squeeze()
        value = self.sigmoid(output) * self.max_value
        return value

class ValueHeadMLP(nn.Module):
    r"""
    The ValueHead class implements a avg emb head for Transformer hidden states that returns a scalar for full sequence.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "dropout"):
            dropout_prob = kwargs.pop("dropout", 0.1)
            logging.info(f"Manually initializing dropout to {dropout_prob}")
        else:
            dropout_prob = config.dropout

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
            logging.info(f"Initializing hidden size from word_embed_proj_dim: {hidden_size}")
        elif hasattr(config, "d_model"):
            hidden_size = config.d_model
            logging.info(f"Initializing hidden size from d_model: {hidden_size}")
        else:
            hidden_size = config.hidden_size
            logging.info(f"Initializing hidden size from hidden_size: {hidden_size}")

        self.summary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1)
        )
        self.sigmoid = nn.Sigmoid()
        # Value head must be provided with maximum possible value. All rewards are expected to be positive between 0 and 1.
        assert "max_value" in kwargs, breakpoint()
        self.max_value = kwargs["max_value"]

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        # output shape: (batch_size, seq_len, hidden_size)
        # avg sequence length
        output = torch.mean(output, dim=1)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary[0].weight.dtype:
            output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output).squeeze()
        value = self.sigmoid(output) * self.max_value
        return value

class ValueHeadAttention(nn.Module):
    r"""
    The ValueHeadAttention class implements a head for GPT2 model's source sequence.
    The head returns a scalar for the entire source sequence.
    It hidden states from source sequence are used as queries and values in MultiheadAttention.
    Attribute specific parameters are used as keys in MultiheadAttention.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        # Create MhA layer for source sequence
        num_heads = 8
        self.source_seq_multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        # Create query parameter vector for MhA layer
        self.seq_query_weights = nn.Parameter(torch.randn(hidden_size))
        # Create linear layer for value estimate after MhA
        self.summary = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # Value head must be provided with maximum possible value. All rewards are expected to be positive between 0 and 1.
        assert "max_value" in kwargs, breakpoint()
        self.max_value = kwargs["max_value"]

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        # output shape: (batch_size, seq_len, hidden_size)
        # key shape: (hidden_size)
        query_vector = self.seq_query_weights.unsqueeze(0).repeat(output.shape[0], 1).unsqueeze(1)
        mha_output, mha_weights = self.source_seq_multihead_attn(query=query_vector, key=output, value=output)
        # mha_output shape: (batch_size, 1, hidden_size)
        # Apply dropout
        mha_output = self.dropout(mha_output)
        # Apply linear layer
        output = self.summary(mha_output).squeeze(2).squeeze(1)
        value = self.sigmoid(output) * self.max_value
        return value


### Generation tasks utils
from .attributes_utils import get_cola_fluency_score

from sentence_transformers import SentenceTransformer, util

# To supress truncation warning. Ref: https://github.com/huggingface/transformers/issues/14285#issuecomment-968839045
import transformers
transformers.logging.set_verbosity_error()

def get_batch_cola_fluency_reward(batch_text, batch_responses, reward_args):
    # Get batch response fluency 
    # According to the Cola Model config: https://huggingface.co/textattack/roberta-base-CoLA/blob/main/config.json
    # "max_position_embeddings": 514
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    batch_resp_fluency = [get_cola_fluency_score(e) for e in reward_args["cola_pipeline"](batch_responses, **tokenizer_kwargs)]
    return batch_resp_fluency

def get_pos_sentiment_score(sentiment_classification_pipeline_result):
    if sentiment_classification_pipeline_result["label"] == "LABEL_1":
        return sentiment_classification_pipeline_result["score"]
    elif sentiment_classification_pipeline_result["label"] == "LABEL_0":
        return 1 - sentiment_classification_pipeline_result["score"]
    elif sentiment_classification_pipeline_result["label"] == "POSITIVE":
        return sentiment_classification_pipeline_result["score"]
    elif sentiment_classification_pipeline_result["label"] == "NEGATIVE":
        return 1 - sentiment_classification_pipeline_result["score"]
    else:
        logging.error(f"Invalid label: {sentiment_classification_pipeline_result['label']}. Terminating...")
        breakpoint()

def get_batch_continuation_sentiment_reward(batch_text, batch_responses, reward_args):
    # batch_responses are already text + response 
    # All roberta models have "max_position_embeddings": 514
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    batch_full_text_sentiment = [get_pos_sentiment_score(e) for e in reward_args["sentiment_classification_pipeline"](batch_responses, **tokenizer_kwargs)]
    return batch_full_text_sentiment

def get_batch_embedding_sim_reward(batch_text, batch_responses, reward_args):
    # Get title and text embedding
    batch_size = len(batch_text)
    batch_text_emb = reward_args['emb_sim_measure'].encode(batch_text, convert_to_tensor=True, show_progress_bar=False).detach().cpu()
    batch_resp_emb = reward_args['emb_sim_measure'].encode(batch_responses, convert_to_tensor=True, show_progress_bar=False).detach().cpu()
    batch_resp_text_sim = [util.pytorch_cos_sim(batch_text_emb[i], batch_resp_emb[i]).item() for i in range(batch_size)]
    return batch_resp_text_sim

def get_batch_doc_nli_entailment_reward(batch_text, batch_responses, reward_args):
    device = reward_args["device"]
    input = reward_args["doc_nli_tokenizer"](batch_text, batch_responses, padding=True, truncation=True, 
                                             truncation_strategy="only_first", return_tensors="pt").to(device)
    output = reward_args["doc_nli_model"](input["input_ids"])
    # Tested one example and looks good. Format: [CLS] trunc_text [SEP] response [SEP]
    prediction = torch.softmax(output["logits"], -1).tolist()
    label_names = ["entailment", "not_entailment"]
    # Get entailment prediction as reward
    doc_nli_score = [e[0] for e in prediction]
    return doc_nli_score

def get_batch_tfidf_reward(batch_text, batch_responses, reward_args):
    batch_size = len(batch_text)
    """ OLD Code for getting diversity distinct-1,2
    # We will use sum of ratio of unique unigrams and bigrams as the diversity reward
    
    # Get per response unique unigrams and bigrams
    batch_unigrams = [get_ngrams_from_sentence(response, n=1, lowercase=False) for response in batch_responses]
    batch_bigrams = [get_ngrams_from_sentence(response, n=2, lowercase=False) for response in batch_responses]
    batch_unique_unigrams = [set(e) for e in batch_unigrams]
    batch_unique_bigrams = [set(e) for e in batch_bigrams]
    per_response_unique_unigrams_ratio = [len(batch_unique_unigrams[i])/len(batch_unigrams[i]) for i in range(batch_size)]
    per_response_unique_bigrams_ratio = [len(batch_unique_bigrams[i])/len(batch_bigrams[i]) for i in range(batch_size)]
    batch_diversity_reward = [per_response_unique_unigrams_ratio[i] + per_response_unique_bigrams_ratio[i] for i in range(batch_size)]
    """
    # reward_args.keys() = dict_keys(['cv', 'tfidf_transformer', 'cola_classifier_model', 'cola_pipeline'])
    # Get tfidf vector for each response
    count_vector=reward_args['cv'].transform(batch_responses)
    tfidf_vector=reward_args['tfidf_transformer'].transform(count_vector)
    # Get mean of non-zero tfidf values as reward
    tfidf_means = [np.mean(tfidf_vector[i].data) if tfidf_vector[i].nnz > 0 else 0.0 for i in range(batch_size)]
    # One word responses are getting 1.0 tfidf. 
    # Adding a length penalty = #words/10 if < 10 words, else 1.0
    length_penalty = [tfidf_vector[i].nnz/10.0 if tfidf_vector[i].nnz < 5 else 1.0 for i in range(batch_size)]
    length_normalized_tfidf = [tfidf_means[i] * length_penalty[i] for i in range(batch_size)]
    # if any([e > 0.9 for e in length_normalized_tfidf]):
    #     breakpoint()
    return length_normalized_tfidf

def get_batch_rewards(batch_text, batch_responses, reward_args, reward_component_to_fn_map):
    component_to_batch_rewards = dict()
    for reward_component, reward_fn in reward_component_to_fn_map.items():
        batch_reward_components = reward_fn(batch_text, batch_responses, reward_args)
        component_to_batch_rewards[reward_component] = batch_reward_components
    # Get final rewards from components
    batch_size = len(batch_text)
    batch_final_rewards = list()
    reward_components = list(reward_component_to_fn_map.keys())
    for i in range(batch_size):
        batch_reward_components = [component_to_batch_rewards[reward_component][i] for reward_component in reward_components]
        batch_final_rewards.append(sum(batch_reward_components))
    component_to_batch_rewards["final_reward"] = batch_final_rewards
    # Prepare reward components from batch rewards
    reward_components += ["final_reward"]
    batch_reward_components = [{reward_component: component_to_batch_rewards[reward_component][i] for reward_component in reward_components} for i in range(batch_size)]

    return batch_reward_components

def summarization_batch_prompt_and_reward_generator(xsum_or_cnn_batch_samples, reward_args, extra_args):
    # Xsum: prompt_suffix: str = "TL;DR:"
    # xsum_or_cnn_batch_samples is a list of len batch_size containing Sample objects
    id_list = [sample.id for sample in xsum_or_cnn_batch_samples]
    prompt_or_input_text_list = [sample.prompt_or_input_text for sample in xsum_or_cnn_batch_samples]
    assert all([len(sample.references) == 1 for sample in xsum_or_cnn_batch_samples]), breakpoint()
    references_list = [sample.references[0] for sample in xsum_or_cnn_batch_samples]
    meta_data_list = [sample.meta_data for sample in xsum_or_cnn_batch_samples]

    text_list = [e[:-6] if e.endswith("TL;DR:") else e for e in prompt_or_input_text_list]
    breakpoint()
    
    reward_component_to_fn_map = {"fluency": get_batch_cola_fluency_reward, "text_sim": get_batch_embedding_sim_reward, "doc_nli_score": get_batch_doc_nli_entailment_reward}
    
    batch_reward_components = get_batch_rewards(text_list, references_list, reward_args, reward_component_to_fn_map)

    final_data_dicts = [{"id": id, "prompt_or_input_text": prompt_or_input_text, "references": references, "meta_data": meta_data, "reward_components": reward_components} for id, prompt_or_input_text, references, meta_data, reward_components in zip(id_list, prompt_or_input_text_list, references_list, meta_data_list, batch_reward_components)]
    return final_data_dicts


def style_transfer_batch_prompt_and_reward_generator(sentiment_transfer_batch_samples, reward_args, extra_args):
    # Primarily for IMDBForSeq2Seq
    # sentiment_transfer_batch_samples is a list of len batch_size containing Sample objects
    id_list = [sample.id for sample in sentiment_transfer_batch_samples]
    prompt_or_input_text_list = [sample.prompt_or_input_text for sample in sentiment_transfer_batch_samples]
    assert all([len(sample.references) == 1 for sample in sentiment_transfer_batch_samples]), breakpoint()
    references_list = [sample.references[0] for sample in sentiment_transfer_batch_samples]
    meta_data_list = [sample.meta_data for sample in sentiment_transfer_batch_samples]
    text_list = prompt_or_input_text_list
    
    reward_component_to_fn_map = {"fluency": get_batch_cola_fluency_reward, "sentiment": get_batch_continuation_sentiment_reward}
    
    batch_reward_components = get_batch_rewards(text_list, references_list, reward_args, reward_component_to_fn_map)

    final_data_dicts = [{"id": id, "prompt_or_input_text": prompt_or_input_text, "references": references, "meta_data": meta_data, "reward_components": reward_components} for id, prompt_or_input_text, references, meta_data, reward_components in zip(id_list, prompt_or_input_text_list, references_list, meta_data_list, batch_reward_components)]
    return final_data_dicts

def translation_batch_prompt_and_reward_generator(translation_batch_samples, reward_args, extra_args):
    # Primarily for IWSLT2017EnDe
    # translation_batch_samples is a list of len batch_size containing Sample objects
    id_list = [sample.id for sample in translation_batch_samples]
    prompt_or_input_text_list = [sample.prompt_or_input_text for sample in translation_batch_samples]
    assert all([len(sample.references) == 1 for sample in translation_batch_samples]), breakpoint()
    references_list = [sample.references[0] for sample in translation_batch_samples]
    meta_data_list = [sample.meta_data for sample in translation_batch_samples]
    
    text_list = [e[29:] if e.startswith("translate English to German: ") else e for e in prompt_or_input_text_list]
    
    reward_component_to_fn_map = {"text_sim": get_batch_embedding_sim_reward}
    
    batch_reward_components = get_batch_rewards(text_list, references_list, reward_args, reward_component_to_fn_map)

    final_data_dicts = [{"id": id, "prompt_or_input_text": prompt_or_input_text, "references": references, "meta_data": meta_data, "reward_components": reward_components} for id, prompt_or_input_text, references, meta_data, reward_components in zip(id_list, prompt_or_input_text_list, references_list, meta_data_list, batch_reward_components)]
    return final_data_dicts

def dialog_batch_prompt_and_reward_generator(dialog_batch_samples, reward_args, extra_args):
    # Primarily for DailyDialog
    # dialog_batch_samples is a list of len batch_size containing Sample objects
    id_list = [sample.id for sample in dialog_batch_samples]
    prompt_or_input_text_list = [sample.prompt_or_input_text for sample in dialog_batch_samples]
    assert all([len(sample.references) == 1 for sample in dialog_batch_samples]), breakpoint()
    references_list = [sample.references[0] for sample in dialog_batch_samples]
    meta_data_list = [sample.meta_data for sample in dialog_batch_samples]
    text_list = prompt_or_input_text_list
    batch_responses = [e.replace(" <EOU>", "").strip() for e in references_list]
    reward_component_to_fn_map = {"fluency": get_batch_cola_fluency_reward, "tfidf": get_batch_tfidf_reward}
    
    batch_reward_components = get_batch_rewards(text_list, batch_responses, reward_args, reward_component_to_fn_map)

    final_data_dicts = [{"id": id, "prompt_or_input_text": prompt_or_input_text, "references": references, "meta_data": meta_data, "reward_components": reward_components} for id, prompt_or_input_text, references, meta_data, reward_components in zip(id_list, prompt_or_input_text_list, references_list, meta_data_list, batch_reward_components)]
    return final_data_dicts

def get_batch_coverage_reward(batch_text, batch_responses, reward_args):
    # reward_args is a dict with dict_keys(['device', 'nlp', 'cola_classifier_model', 'cola_pipeline'])
    nlp = reward_args["nlp"]
    all_coverages = list()
    for concept, response in zip(batch_text, batch_responses):
        concept = batch_text[0]
        response = batch_responses[0]
        concept_lemmas_set = set([token.lemma_ for token in nlp(concept)])
        response_lemmas_set = set([token.lemma_ for token in nlp(response)])
        # Get the intersection of the two sets
        intersection_set = concept_lemmas_set.intersection(response_lemmas_set)
        all_coverages.append(len(intersection_set) / len(concept_lemmas_set))
    return all_coverages

from .comet_utils import get_comet_keras_input_and_labels
def get_batch_comet_reward(batch_text, batch_responses, reward_args):
    # reward_args is a dict with dict_keys(['task_name', 'device', 'comet_critic_model_name', 'comet_roberta_model', 'comet_tokenizer', 'comet_classification_head'])
    device = reward_args["device"]
    comet_roberta_model = reward_args["comet_roberta_model"]
    comet_tokenizer = reward_args["comet_tokenizer"]
    comet_classification_head = reward_args["comet_classification_head"]
    # Extract the head and relation from batch_text
    # each text is of the template: "<head> head </head> <relation> relation </relation> [GEN] "
    batch_heads = [e.split("</head>")[0].split("<head>")[1].strip() for e in batch_text]
    batch_relations = [e.split("</relation>")[0].split("<relation>")[1].strip() for e in batch_text]
    batch_tails = batch_responses
    # Get the comet scores
    comet_eval_dicts = [{"head": head, "relation": relation, "tail": tail, "valid": 1.0} for head, relation, tail in zip(batch_heads, batch_relations, batch_tails)]
    classifer_inputs, label = get_comet_keras_input_and_labels(comet_eval_dicts, comet_tokenizer)
    classifer_inputs.to(device)
    # comet_roberta_model.to(device)
    outputs = comet_roberta_model(**classifer_inputs)
    pooler_output = outputs.pooler_output
    # comet_classification_head.to(device)
    pytorch_critic_pred = comet_classification_head.forward(pooler_output)
    return pytorch_critic_pred.squeeze().tolist()

def get_batch_faithfulness_reward(batch_text, batch_responses, reward_args):
    # reward_args has dict_keys(['device', 'cola_classifier_model', 'cola_pipeline', 'faithdial_critic_model_name', 'faithdial_tokenizer', 'faithdial_critic_model', 'depth_dialogRPT_model_name', 'depth_dialogRPT_tokenizer', 'depth_dialogRPT_model'])
    device = reward_args["device"]
    faithdial_tokenizer = reward_args["faithdial_tokenizer"]
    faithdial_critic_model = reward_args["faithdial_critic_model"]
    eos_token = "<|endoftext|>"
    batch_knowledge = [e.split(eos_token, 1)[0][10:] for e in batch_text]
    with torch.no_grad():
        inputs = faithdial_tokenizer(batch_knowledge, batch_responses, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = faithdial_critic_model(**inputs)
        label_logits = outputs.logits
        probs = torch.softmax(label_logits, dim=1)
        # 0 is entailment
        critic_scores = probs[:, 0].tolist()
    return critic_scores

def get_batch_faithdial_depth_reward(batch_text, batch_responses, reward_args):
    # reward_args has dict_keys(['device', 'cola_classifier_model', 'cola_pipeline', 'faithdial_critic_model_name', 'faithdial_tokenizer', 'faithdial_critic_model', 'depth_dialogRPT_model_name', 'depth_dialogRPT_tokenizer', 'depth_dialogRPT_model'])
    device = reward_args["device"]
    eos_token = "<|endoftext|>"
    batch_history = [e.split(eos_token, 1)[1] for e in batch_text]
    # batch_history = [e.split("<|endoftext|>", 1)[1] for e in batch_text]
    depth_dialogRPT_tokenizer = reward_args["depth_dialogRPT_tokenizer"]
    depth_dialogRPT_model = reward_args["depth_dialogRPT_model"]
    with torch.no_grad():
        full_hypothesis = [f"{history}{response}" for history, response in zip(batch_history, batch_responses)]
        # full_hypothesis = [f"{history}<|endoftext|>{response}" for history, response in zip(batch_history, batch_responses)]
        model_inputs = depth_dialogRPT_tokenizer(full_hypothesis, return_tensors="pt", padding=True, truncation=True).to(device)
        # model_inputs["input_ids"].shape = torch.Size([32, 2251])
        try:
            result = depth_dialogRPT_model(**model_inputs)
        except RuntimeError:
            breakpoint()
        preds = torch.sigmoid(result.logits).squeeze().tolist()
    return preds


def get_batch_dialogrpt_depth_reward(batch_text, batch_responses, reward_args):
    # reward_args has dict_keys(['device', 'cola_classifier_model', 'cola_pipeline', 'depth_dialogRPT_model_name', 'depth_dialogRPT_tokenizer', 'depth_dialogRPT_model', 'updown_dialogRPT_model_name', 'updown_dialogRPT_tokenizer', 'updown_dialogRPT_model'])
    device = reward_args["device"]
    depth_dialogRPT_tokenizer = reward_args["depth_dialogRPT_tokenizer"]
    depth_dialogRPT_model = reward_args["depth_dialogRPT_model"]
    with torch.no_grad():
        full_hypothesis = [f"{parent_comment}{response}" for parent_comment, response in zip(batch_text, batch_responses)]
        model_inputs = depth_dialogRPT_tokenizer(full_hypothesis, return_tensors="pt", padding=True, truncation=True).to(device)
        # model_inputs["input_ids"].shape = torch.Size([32, 2251])
        try:
            result = depth_dialogRPT_model(**model_inputs)
        except RuntimeError:
            breakpoint()
        preds = torch.sigmoid(result.logits).squeeze().tolist()
    return preds

def get_batch_dialogrpt_updown_reward(batch_text, batch_responses, reward_args):
    device = reward_args["device"]
    updown_dialogRPT_tokenizer = reward_args["updown_dialogRPT_tokenizer"]
    updown_dialogRPT_model = reward_args["updown_dialogRPT_model"]
    with torch.no_grad():
        full_hypothesis = [f"{parent_comment}{response}" for parent_comment, response in zip(batch_text, batch_responses)]
        model_inputs = updown_dialogRPT_tokenizer(full_hypothesis, return_tensors="pt", padding=True, truncation=True).to(device)
        # model_inputs["input_ids"].shape = torch.Size([32, 2251])
        try:
            result = updown_dialogRPT_model(**model_inputs)
        except RuntimeError:
            breakpoint()
        preds = torch.sigmoid(result.logits).squeeze().tolist()
    return preds

from .attributes_utils import get_offensive_scores
def get_batch_toxichat_safe_reward(batch_text, batch_responses, reward_args):
    device = reward_args["device"]
    offensive_model = reward_args["offensive_model"]
    offensive_tokenizer = reward_args["offensive_tokenizer"]
    eos_token = "<|endoftext|>"
    full_threads = [f"{parent_comment}{response}{eos_token}" for parent_comment, response in zip(batch_text, batch_responses)]
    # full_threads = [f"{parent_comment}{response}<|endoftext|>" for parent_comment, response in zip(batch_text, batch_responses)]
    current_batch_post_threads_and_predictions = get_offensive_scores(full_threads, offensive_model, offensive_tokenizer, device, current_batch_post_threads_and_predictions=None, eos_seperator = "</s>")
    # Assert all offensive predictions are of length 2
    assert all([len(e[-1]['offensive']) == 2 for e in current_batch_post_threads_and_predictions]), breakpoint()
    # Get the offensive scores of last repsonse
    batch_safe_probs = [e[-1]['offensive'][-1][-1][0] for e in current_batch_post_threads_and_predictions]
    return batch_safe_probs

from .utils import get_ngrams_from_sentence
def get_batch_ngram_diversity_reward(batch_text, batch_responses, reward_args):
    unigrams = [get_ngrams_from_sentence(response, n=1, lowercase=False) for response in batch_responses]
    bigrams = [get_ngrams_from_sentence(response, n=2, lowercase=False) for response in batch_responses]
    trigrams = [get_ngrams_from_sentence(response, n=3, lowercase=False) for response in batch_responses]
    unique_unigrams_ratio = [len(set(e))/len(e) for e in unigrams]
    unique_bigrams_ratio = [len(set(e))/len(e) if len(e) > 0 else 0.0 for e in bigrams]
    unique_trigrams_ratio = [len(set(e))/len(e) if len(e) > 0 else 0.0 for e in trigrams]
    batch_ngram_diversity_reward = [unique_unigrams_ratio[i] + unique_bigrams_ratio[i] + unique_trigrams_ratio[i] for i in range(len(batch_text))]
    return batch_ngram_diversity_reward

# Task to reward mapping
task_to_reward_fn_map = {
    "Xsum": {"fluency": get_batch_cola_fluency_reward, "text_sim": get_batch_embedding_sim_reward, "doc_nli_score": get_batch_doc_nli_entailment_reward},
    "CNNDailyMail": {"fluency": get_batch_cola_fluency_reward, "text_sim": get_batch_embedding_sim_reward, "doc_nli_score": get_batch_doc_nli_entailment_reward},
    "IMDBForSeq2Seq": {"fluency": get_batch_cola_fluency_reward, "sentiment": get_batch_continuation_sentiment_reward},
    "IWSLT2017EnDe": {"text_sim": get_batch_embedding_sim_reward},
    "DailyDialog": {"fluency": get_batch_cola_fluency_reward, "tfidf": get_batch_tfidf_reward},
    "CommonGen": {"fluency": get_batch_cola_fluency_reward, "coverage": get_batch_coverage_reward},
    "COMET": {"p_valid_model": get_batch_comet_reward},
    "WOW": {"fluency": get_batch_cola_fluency_reward, "faithdial": get_batch_faithfulness_reward, "depth": get_batch_faithdial_depth_reward, "tfidf": get_batch_tfidf_reward},
    "faithdial": {"fluency": get_batch_cola_fluency_reward, "faithdial": get_batch_faithfulness_reward, "depth": get_batch_faithdial_depth_reward, "tfidf": get_batch_tfidf_reward},
    "faithdial_wow": {"fluency": get_batch_cola_fluency_reward, "faithdial": get_batch_faithfulness_reward, "depth": get_batch_faithdial_depth_reward, "tfidf": get_batch_tfidf_reward},
    "reddit_pos": {"fluency": get_batch_cola_fluency_reward, "depth": get_batch_dialogrpt_depth_reward, "updown": get_batch_dialogrpt_updown_reward, "safe": get_batch_toxichat_safe_reward, "tfidf": get_batch_tfidf_reward},
    "reddit_neg": {"fluency": get_batch_cola_fluency_reward, "depth": get_batch_dialogrpt_depth_reward, "updown": get_batch_dialogrpt_updown_reward, "safe": get_batch_toxichat_safe_reward, "tfidf": get_batch_tfidf_reward},
}

def get_text_and_responses_from_task_prompts_and_references(task_name, prompt_or_input_text_list, references_list):
    if task_name == "Xsum":
        batch_texts = [e[:-6] if e.endswith("TL;DR:") else e for e in prompt_or_input_text_list]
        batch_responses = references_list
    elif task_name == "CNNDailyMail":
        batch_texts = [e[11:] if e.startswith("Summarize: ") else e for e in prompt_or_input_text_list]
        batch_responses = references_list
    elif task_name == "IMDBForSeq2Seq":
        batch_texts = prompt_or_input_text_list
        # For imdb we will set responses as full text
        batch_responses = [f"{e} {r}" for e, r in zip(prompt_or_input_text_list, references_list)]
    elif task_name == "IWSLT2017EnDe":
        batch_texts = [e[29:] if e.startswith("translate English to German: ") else e for e in prompt_or_input_text_list]
        batch_responses = references_list
    elif task_name in ["DailyDialog", "COMET"]:
        batch_texts = prompt_or_input_text_list
        batch_responses = references_list
    elif task_name in ["WOW", "faithdial", "faithdial_wow", "reddit_pos", "reddit_neg"]:
        batch_texts = prompt_or_input_text_list
        # Remove eos_token from end of responses
        eos_token = "<|endoftext|>"
        batch_responses = [e[:-len(eos_token)] if e.endswith(eos_token) else e for e in references_list]
    elif task_name == "CommonGen":
        concepts = [e.replace("generate a sentence with: ", "").strip()[:-1] for e in prompt_or_input_text_list]
        batch_texts = concepts
        batch_responses = references_list
    else:
        print(f"Error: task_name {task_name} not supported")
        breakpoint()
    return batch_texts, batch_responses

def general_batch_prompt_and_reward_generator(task_batch_samples, reward_args, extra_args):
    # task_batch_samples is a list of len batch_size containing Sample objects
    id_list = [sample.id for sample in task_batch_samples]
    prompt_or_input_text_list = [sample.prompt_or_input_text for sample in task_batch_samples]
    assert all([len(sample.references) == 1 for sample in task_batch_samples]), breakpoint()
    references_list = [sample.references[0] for sample in task_batch_samples]
    meta_data_list = [sample.meta_data for sample in task_batch_samples]
    assert "task_name" in extra_args, breakpoint()
    task_name = extra_args["task_name"]
    if task_name == "DailyDialog":
        # Replace the <EOU> in the prompt to gpt2 eos_token (i.e. <|endoftext|>)
        prompt_or_input_text_list = [e.replace("<EOU>", "<|endoftext|>").strip() for e in prompt_or_input_text_list]
        # Remove the <EOU> in the references
        references_list = [e.replace(" <EOU>", "").strip() for e in references_list]
    
    reward_component_to_fn_map = task_to_reward_fn_map[task_name]
    batch_texts, batch_responses = get_text_and_responses_from_task_prompts_and_references(task_name, prompt_or_input_text_list, references_list)
    batch_reward_components = get_batch_rewards(batch_texts, batch_responses, reward_args, reward_component_to_fn_map)

    final_data_dicts = [{"id": id, "prompt_or_input_text": prompt_or_input_text, "references": references, "meta_data": meta_data, "reward_components": reward_components} for id, prompt_or_input_text, references, meta_data, reward_components in zip(id_list, prompt_or_input_text_list, references_list, meta_data_list, batch_reward_components)]
    return final_data_dicts

from utils.rl4lms_data_utils import Xsum, CNNDailyMail, IMDBForSeq2Seq, IWSLT2017EnDe, DailyDialog, CommonGen


task_to_class_mapping = {
                        # prompt_suffix: "TL;DR:"
                        "Xsum": {"class": Xsum, 
                                "prepare_args":{"prompt_suffix": "TL;DR:"}}, 
                        # CNNDailyMail: prompt_prefix: "Summarize: " in RL4LMs config. But we are not using that here.
                        "CNNDailyMail": {"class": CNNDailyMail, 
                                        "prepare_args":{"prompt_prefix": "Summarize: "}}, 
                        # "ToTTo": ToTTo, 
                        # "IMDB": IMDB, 
                        # positive_ratio: 1.0
                        "IMDBForSeq2Seq": {"class": IMDBForSeq2Seq, 
                                        "prepare_args": {"positive_ratio": 1.0}},
                        # prompt_prefix: "translate English to German: "
                        "IWSLT2017EnDe": {"class": IWSLT2017EnDe, 
                                        "prepare_args": {"prompt_prefix": "translate English to German: "}},
                        # context_size: 5
                        "DailyDialog": {"class": DailyDialog, 
                                        "prepare_args": {"context_size": 5}},
                        # concept_end_token: '.'
                        # concept_separator_token: ' '
                        # prefix: "generate a sentence with: "
                        "CommonGen": {"class": CommonGen,
                                      "prepare_args": {"concept_end_token": ".", "concept_separator_token": " ", "prefix": "generate a sentence with: "}},
                        }

