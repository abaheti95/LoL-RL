#### Classes and Functions for ToxiChat Classifiers #########################
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
import numpy as np
from time import time
from math import comb

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    n_is = np.array(cls_num_list)
    per_cls_weights = (1 - beta) / (1 - np.power(beta, n_is))
    per_cls_weights = torch.from_numpy(per_cls_weights)
    # per_cls_weights = per_cls_weights / per_cls_weights.sum() * 10
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        zi = -input
        batch_size = input.size(0)
        zi[torch.arange(batch_size), target] *= -1
        pis = F.sigmoid(zi)
        first_term = (1-pis) ** self.gamma
        second_term = torch.log(pis)
        multipled = torch.einsum("bj,bj->b", (first_term, second_term))
        class_weights = self.weight[target].float().to(device)
        loss = -class_weights.dot(multipled)
        return loss


class GPT2ForOC_S_stance(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_off_labels = 2
        self.num_stance_labels = 3
        # Stane labels: neutral, agree, disagree
        # print(f"Number of off labels for GPT2ForOC_S_stance classifier = {self.num_off_labels}")
        # print(f"Number of target labels for GPT2ForOC_S_stance classifier = {len(TARGET_GROUPS)}")
        print(f"Number of stance labels for GPT2ForOC_S_stance classifier = {self.num_stance_labels}")
        self.dropout = nn.Dropout(config.embd_pdrop)
        # self.off_classifier = nn.Linear(config.hidden_size, self.num_off_labels)
        # self.target_classifier = nn.Linear(config.hidden_size, len(TARGET_GROUPS))
        self.stance_classifier = nn.Linear(config.hidden_size*4, self.num_stance_labels)
        # self.init_weights()
        config.focal_loss = True
        if config.focal_loss:
            # Instantiate using Focal loss
            weight = reweight(config.cls_num_list)
            self.stance_loss_fct = FocalLoss(weight=weight, gamma=1.0)
            print(f"Using Class balanced focal loss with beta = 0.9999 and gamma = 1.0")
        else:
            # self.stance_loss_fct = nn.CrossEntropyLoss()
            # print(f"Using Cross Entropy loss with no weights")
            # self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0]))
            # print(f"Using Cross Entropy loss with weights [1.0, 10.0, 10.0]")
            self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 100.0]))
            print(f"Using Cross Entropy loss with weights [1.0, 100.0, 100.0]")
        # self.target_loss_fct = nn.BCEWithLogitsLoss()
        # self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 100.0]))
        # self.stance_loss_multiplier = 2.0
    
    def forward(
        self,
        input_ids,
        eos_toward_token_ids=None,
        eos_response_token_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        stance_labels=None,
        # off_labels=None,
        # target_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
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
        """
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # Type of outputs = BaseModelOutputWithPastAndCrossAttentions
        # ref: https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#BaseModelOutputWithPastAndCrossAttentions
        GPT2_last_layer_output = outputs.last_hidden_state

        # Get the hidden representations for the EOS token ids
        eos_toward_token_representation = GPT2_last_layer_output[eos_toward_token_ids[0], eos_toward_token_ids[1], :]
        eos_response_token_representation = GPT2_last_layer_output[eos_response_token_ids[0], eos_response_token_ids[1], :]
        difference1 = eos_toward_token_representation - eos_response_token_representation
        hadamard = eos_toward_token_representation * eos_response_token_representation
        stance_classifier_input = torch.cat([eos_toward_token_representation, eos_response_token_representation, difference1, hadamard], axis=1)
        # Apply dropout
        stance_classifier_input = self.dropout(stance_classifier_input)
        # Compute stance logits from concatenated eos representations
        stance_logits = self.stance_classifier(stance_classifier_input)


        outputs = (stance_logits,) + outputs[2:]
        # If stance_labels given, compute loss from stance_logits
        
        loss = 0.0
        if stance_labels is not None:
            loss = self.stance_loss_fct(stance_logits.view(-1, self.num_stance_labels), stance_labels.view(-1))
            # print(f"input ids = {input_ids}, DGPT outputs shape = {GPT2_last_layer_output.size()} vs nan count = {torch.isnan(GPT2_last_layer_output).sum()}")
            # print(f"Off logits = {stance_logits} vs Off labels = {off_labels}")
            # if target_labels is not None:
            # 	# Some of the target_labels can still be None. We have to ignore loss for these target labels
            # 	for i, target_label in enumerate(target_labels):
            # 		if target_label is not None:
            # 			loss += self.target_loss_fct(target_logits[i], target_label.to(device))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

def get_stance_model(stance_model_dir, device):
    start_time = time()
    print(f"Loading pretrained Stance model and tokenizer from {stance_model_dir}...")
    stance_model = GPT2ForOC_S_stance.from_pretrained(stance_model_dir).to(device)
    stance_tokenizer = GPT2Tokenizer.from_pretrained(stance_model_dir)
    print(f"Loaded Stance model in device = {device} in {time() - start_time:.2f} seconds")
    return stance_model, stance_tokenizer

### Similar code for Offensive prediction

class ValueHeadMLP(nn.Module):
    r"""
    The ValueHead class implements a head for Offensive Classifier giving scalar value every prediction.
    """

    def __init__(self, hidden_size, dropout_prob, max_value, **kwargs):
        super().__init__()

        self.dropout = nn.Dropout(dropout_prob)

        self.summary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1)
        )
        self.sigmoid = nn.Sigmoid()
        # Value head must be provided with maximum possible value. All rewards are expected to be positive between 0 and 1.
        self.max_value = max_value

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        # output shape: (off_pred, hidden_size)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary[0].weight.dtype:
            output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output).squeeze()
        value = self.sigmoid(output) * self.max_value
        return value

class GPT2ForOC_S_offensive(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_off_labels = 2
        # Offense labels: safe, offensive
        self.num_stance_labels = 3
        print(f"Number of off labels for GPT2ForOC_S_offensive classifier = {self.num_off_labels}")
        # print(f"Number of target labels for GPT2ForOC_S_offensive classifier = {len(TARGET_GROUPS)}")
        # print(f"Number of stance labels for GPT2ForOC_S_offensive classifier = {self.num_stance_labels}")
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.off_classifier = nn.Linear(config.hidden_size, self.num_off_labels)
        self.value_head = None
        if hasattr(config, "value_head") and config.value_head:
            print(f"Adding value head to GPT2ForOC_S_offensive classifier")
            self.value_head = config.value_head
            self.critic = ValueHeadMLP(config.hidden_size, config.embd_pdrop, max_value=1.0)
        # self.target_classifier = nn.Linear(config.hidden_size, len(TARGET_GROUPS))
        # self.stance_classifier = nn.Linear(config.hidden_size*4, self.num_stance_labels)
        # self.init_weights()
        self.loss_fct = nn.CrossEntropyLoss()
        # self.target_loss_fct = nn.BCEWithLogitsLoss()
        # self.stance_loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0, 100.0]))
        # self.stance_loss_multiplier = 2.0
    
    def forward(
        self,
        input_ids,
        utterance_eos_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        off_labels=None,
        # target_labels=None,
        # stance_labels=None,
        # eos_toward_token_ids=None,
        # eos_response_token_ids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
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
        """
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        # Type of outputs = BaseModelOutputWithPastAndCrossAttentions
        # ref: https://huggingface.co/transformers/_modules/transformers/modeling_outputs.html#BaseModelOutputWithPastAndCrossAttentions
        GPT2_last_layer_output = outputs.last_hidden_state

        # Extract all EOS token representations from GPT2's last layer representations
        eos_token_representation = GPT2_last_layer_output[utterance_eos_ids[0], utterance_eos_ids[1], :]
        # Apply dropout on representations
        eos_token_representation = self.dropout(eos_token_representation)
        # Compute logits from cls representations
        off_logits = self.off_classifier(eos_token_representation)
        
        # target_logits = self.target_classifier(eos_token_representation)

        outputs['logits'] = off_logits
        # If off_labels given, compute loss from off_logits
        
        loss = 0.0
        if off_labels is not None:
            loss = self.loss_fct(off_logits.view(-1, self.num_off_labels), off_labels.view(-1))
            # print(f"input ids = {input_ids}, DGPT outputs shape = {GPT2_last_layer_output.size()} vs nan count = {torch.isnan(GPT2_last_layer_output).sum()}")
            # print(f"Off logits = {off_logits} vs Off labels = {off_labels}")
            # if target_labels is not None:
            # 	# Some of the target_labels can still be None. We have to ignore loss for these target labels
            # 	for i, target_label in enumerate(target_labels):
            # 		if target_label is not None:
            # 			loss += self.target_loss_fct(target_logits[i], target_label.to(device))
            outputs['loss'] = loss
        if self.value_head:
            off_pred_value = self.critic(eos_token_representation)
            # append at the end of outputs
            outputs['values'] = off_pred_value

        return outputs  # (loss), logits, (value)


def get_off_model(off_model_dir, device):
    start_time = time()
    print(f"Loading pretrained Offensive model and tokenizer from {off_model_dir}...")
    offensive_model = GPT2ForOC_S_offensive.from_pretrained(off_model_dir).to(device)
    offensive_tokenizer = GPT2Tokenizer.from_pretrained(off_model_dir)
    print(f"Loaded Offensive model in device {device} in {time() - start_time:.2f} seconds")
    return offensive_model, offensive_tokenizer

def prepare_threads_for_stance_model_predictions(current_threads, tokenizer, eos_seperator = "</s>"):
    all_GPT2_model_input_texts = list()
    gold_stance_u_id_pairs = list()
    per_instance_n_utterances = list()
    for i, post_thread in enumerate(current_threads):
        # GPT2_string = post_thread.replace(" EOS ", tokenizer.eos_token)
        # GPT2_string = post_thread.replace("</s>", tokenizer.eos_token)
        GPT2_string = post_thread.replace(eos_seperator, tokenizer.eos_token)
        all_GPT2_model_input_texts.append(GPT2_string)
        # n_utterances = len([u for u in post_thread.split(" EOS ") if u])
        # n_utterances = len([u for u in post_thread.split("</s>") if u])
        n_utterances = len([u for u in post_thread.split(eos_seperator) if u])
        per_instance_n_utterances.append(n_utterances)
        # Create stance u_id_pairs
        for u_from in range(2, n_utterances+1):
            for u_to in range(1, u_from):
                gold_stance_u_id_pairs.append((i, u_to, u_from))

    # Tokenize
    all_GPT2_model_inputs_tokenized = tokenizer.batch_encode_plus(all_GPT2_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
    input_ids, attention_mask = all_GPT2_model_inputs_tokenized['input_ids'], all_GPT2_model_inputs_tokenized['attention_mask']
    # Extract the word_ids of CLS tokens i.e. the beginning of all the utterances
    eos_token_ids = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)

    assert len(per_instance_n_utterances) == len(current_threads)
    # Convert the pad_token_ids to eos_token_ids as there is no pad token in DGPT model
    input_ids[input_ids==tokenizer.pad_token_id] = tokenizer.eos_token_id
    try:
        assert input_ids.size(1) < 512
    except AssertionError:
        print(f"Error: One of the instance has length longer than 512 tokens: {input_ids.shape}")
        print(f"Error: Skipping this batch!")
        return None

    # For stance labels create specific eos_token_ids for stance u_id pairs
    # Compute the per instance per utterance EOS ids
    per_instance_per_utterance_eos_ids = [list() for i in range(len(current_threads))]
    instance_ids = eos_token_ids[0].tolist()
    utterance_eos_ids = eos_token_ids[1].tolist()
    for instance_id, utterance_eos_id in zip(instance_ids, utterance_eos_ids):
        per_instance_per_utterance_eos_ids[instance_id].append(utterance_eos_id)
    # Using the creating list compute the eos_ids for stance u_id pairs
    stance_specific_instance_ids = list()
    eos_toward_token_ids = list()
    eos_response_token_ids = list()
    try:
        for instance_id, toward_u_id, response_u_id in gold_stance_u_id_pairs:
            stance_specific_instance_ids.append(instance_id)
            eos_toward_token_ids.append(per_instance_per_utterance_eos_ids[instance_id][toward_u_id-1])
            eos_response_token_ids.append(per_instance_per_utterance_eos_ids[instance_id][response_u_id-1])
    except IndexError:
        print(f"Error: Index error at {instance_id}, with {toward_u_id} and {response_u_id}")
        print(f"Per instance per utterance eos ids = {per_instance_per_utterance_eos_ids}")
        print(f"Current thread = {current_threads[instance_id]}$$")
        # print(f"Response = {resp}$$")
        print(f"Gold stance u_id pairs = {gold_stance_u_id_pairs}")

        return None
    # Convert generated lists into tensors
    stance_specific_instance_ids = torch.LongTensor(stance_specific_instance_ids)
    eos_toward_token_ids = torch.LongTensor(eos_toward_token_ids)
    eos_response_token_ids = torch.LongTensor(eos_response_token_ids)
    # Convert token_ids into tuples for future processing
    eos_toward_token_ids = (stance_specific_instance_ids, eos_toward_token_ids)
    eos_response_token_ids = (stance_specific_instance_ids, eos_response_token_ids)
    return {"input_ids": input_ids, "eos_token_ids": eos_token_ids, "gold_stance_u_id_pairs": gold_stance_u_id_pairs, "eos_toward_token_ids": eos_toward_token_ids, "eos_response_token_ids": eos_response_token_ids, "input_str": all_GPT2_model_input_texts, "n_utterances": per_instance_n_utterances, "batch_threads": current_threads}

def get_stance_scores(current_batch_post_threads, stance_model, stance_tokenizer, device, current_batch_post_threads_and_predictions=None, eos_seperator = "</s>"):
    if current_batch_post_threads_and_predictions is None:
        current_batch_post_threads_and_predictions = [[post_thread, {"stance":list(), "offensive":list()}] for post_thread in current_batch_post_threads]

    # Get stance predictions for current threads
    batch_data = prepare_threads_for_stance_model_predictions(current_batch_post_threads, stance_tokenizer, eos_seperator)
    if batch_data is None:
        return None
    input_dict = {"input_ids": batch_data["input_ids"].to(device)}
    input_dict["eos_toward_token_ids"] = batch_data["eos_toward_token_ids"]
    input_dict["eos_response_token_ids"] = batch_data["eos_response_token_ids"]
    # Forward
    stance_logits = stance_model(**input_dict)[0]
    # Apply softmax on the stance_logits
    softmax_func = nn.Softmax(dim=1)
    softmax_stance_logits = softmax_func(stance_logits).tolist()
    per_instance_n_utterances = batch_data["n_utterances"]
    # convert scores and id_pairs to per_instance scores and labels
    gold_stance_u_id_pairs = batch_data["gold_stance_u_id_pairs"]
    # print(batch_data)
    # print(f"Sofmax stance logits = {softmax_stance_logits}")
    # print(f"Gold stance u_id pairs = {gold_stance_u_id_pairs}")
    for index, (instance_id, to_u_id, from_u_id) in enumerate(gold_stance_u_id_pairs):
        current_batch_post_threads_and_predictions[instance_id][1]["stance"].append((to_u_id, from_u_id, softmax_stance_logits[index]))
    return current_batch_post_threads_and_predictions

def prepare_threads_for_offensive_model_predictions(current_threads, tokenizer, eos_seperator = "</s>"):
    all_GPT2_model_input_texts = list()
    per_instance_n_utterances = list()
    for i, post_thread in enumerate(current_threads):
        # GPT2_string = post_thread.replace(" EOS ", tokenizer.eos_token)
        # GPT2_string = post_thread.replace("</s>", tokenizer.eos_token)
        GPT2_string = post_thread.replace(eos_seperator, tokenizer.eos_token)
        all_GPT2_model_input_texts.append(GPT2_string)
        # n_utterances = len([u for u in post_thread.split(" EOS ") if u])
        # n_utterances = len([u for u in post_thread.split("</s>") if u])
        n_utterances = len([u for u in post_thread.split(eos_seperator) if u])
        per_instance_n_utterances.append(n_utterances)

    # Tokenize
    all_GPT2_model_inputs_tokenized = tokenizer.batch_encode_plus(all_GPT2_model_input_texts, padding=True, add_special_tokens=False, return_tensors="pt")
    input_ids, attention_mask = all_GPT2_model_inputs_tokenized['input_ids'], all_GPT2_model_inputs_tokenized['attention_mask']
    # Extract the word_ids of CLS tokens i.e. the beginning of all the utterances
    eos_token_ids = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)

    assert len(per_instance_n_utterances) == len(current_threads)
    # Convert the pad_token_ids to eos_token_ids as there is no pad token in DGPT model
    input_ids[input_ids==tokenizer.pad_token_id] = tokenizer.eos_token_id
    try:
        assert input_ids.size(1) < 512
    except AssertionError:
        print(f"Error: One of the instance has length longer than 512 tokens: {input_ids.shape}")
        print(f"Error: Skipping this batch!")
        return None

    return {"input_ids": input_ids, "eos_token_ids": eos_token_ids, "input_str": all_GPT2_model_input_texts, "n_utterances": per_instance_n_utterances, "batch_threads": current_threads}

def get_offensive_scores(current_batch_post_threads, off_model, off_tokenizer, device, current_batch_post_threads_and_predictions=None, eos_seperator = "</s>"):
    if current_batch_post_threads_and_predictions is None:
        current_batch_post_threads_and_predictions = [[post_thread, {"stance":list(), "offensive":list()}] for post_thread in current_batch_post_threads]
    
    # Get offensive predictions for the current threads
    batch_data = prepare_threads_for_offensive_model_predictions(current_batch_post_threads, off_tokenizer, eos_seperator)
    if batch_data is None:
        return None
    eos_token_ids = batch_data["eos_token_ids"]
    input_dict = {"input_ids": batch_data["input_ids"].to(device), "utterance_eos_ids": batch_data["eos_token_ids"]}
    # Forward
    outputs = off_model(**input_dict)
    off_logits = outputs['logits']
    softmax_func = nn.Softmax(dim=1)
    softmax_off_logits = softmax_func(off_logits)

    assert softmax_off_logits.size(0) == eos_token_ids[0].size(0), breakpoint()
    softmax_off_logits = softmax_off_logits.tolist()
    # Convert eos_token_ids from tensor to list and zip
    eos_token_ids = (eos_token_ids[0].tolist(), eos_token_ids[1].tolist())
    prev_instance_id = -1
    for index, (instance_id, eos_token_id) in enumerate(zip(eos_token_ids[0], eos_token_ids[1])):
        if instance_id != prev_instance_id:
            prev_instance_id = instance_id
            u_id = 0
        else:
            u_id += 1
        current_batch_post_threads_and_predictions[instance_id][1]["offensive"].append((u_id, softmax_off_logits[index]))
    return current_batch_post_threads_and_predictions

# TODO: Create batched versions of each function for future use
def get_offensive_scores_batched(current_batch_post_threads, off_model, off_tokenizer, device, current_batch_post_threads_and_predictions=None, eos_seperator = "</s>", batch_size = 16):
    all_post_threads_and_predictions = list()
    for i in range(0, len(current_batch_post_threads), batch_size):
        sub_batch_post_threads = current_batch_post_threads[i:i+batch_size]
        sub_batch_post_threads_and_predictions = current_batch_post_threads_and_predictions[i:i+batch_size] if current_batch_post_threads_and_predictions is not None else None
        return_value = get_offensive_scores(sub_batch_post_threads, off_model, off_tokenizer, device, sub_batch_post_threads_and_predictions, eos_seperator)
        if return_value is not None:
            sub_batch_post_threads_and_predictions = return_value
            all_post_threads_and_predictions.extend(sub_batch_post_threads_and_predictions)
    return all_post_threads_and_predictions

def get_stance_scores_batched(current_batch_post_threads, stance_model, stance_tokenizer, device, current_batch_post_threads_and_predictions=None, eos_seperator = "</s>", batch_size = 16):
    all_post_threads_and_predictions = list()
    for i in range(0, len(current_batch_post_threads), batch_size):
        sub_batch_post_threads = current_batch_post_threads[i:i+batch_size]
        sub_batch_post_threads_and_predictions = current_batch_post_threads_and_predictions[i:i+batch_size] if current_batch_post_threads_and_predictions is not None else None
        return_value = get_stance_scores(sub_batch_post_threads, stance_model, stance_tokenizer, device, sub_batch_post_threads_and_predictions, eos_seperator)
        if return_value is not None:
            sub_batch_post_threads_and_predictions = return_value
            all_post_threads_and_predictions.extend(sub_batch_post_threads_and_predictions)
    return all_post_threads_and_predictions

############################################################################################################

#### Adding more attribute functions and dictionaries along with toxichat ####
import math

def get_GPT2_loss(batch_responses, gpt2_model, gpt2_tokenizer, device):
    with torch.no_grad():
        encoded_responses = gpt2_tokenizer(batch_responses, padding=True, return_tensors="pt").to(device)
        outputs = gpt2_model(**encoded_responses)
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = encoded_responses['input_ids'][..., 1:].contiguous()
        shift_mask = encoded_responses['attention_mask'][..., :-1].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=gpt2_tokenizer.eos_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_logits.size(0), shift_logits.size(1))
        per_response_loss = torch.sum(loss * shift_mask, dim=1) / torch.sum(shift_mask, dim=1)
        # NOTE: we will directly used loss with a scaling factor to get the reward
        # Convert nans to 100.0
        per_response_loss[torch.isnan(per_response_loss)] = 100.0
        return per_response_loss.cpu().tolist()

def get_batched_gpt2_metrics(responses, gpt2_model, gpt2_tokenizer, device, batch_size=32):
    all_responses_gpt2_loss_scores = list()
    for i in range(0, len(responses), batch_size):
        batch_responses = responses[i:i+batch_size]
        batch_responses_gpt2_loss_scores = get_GPT2_loss(batch_responses, gpt2_model, gpt2_tokenizer, device)
        all_responses_gpt2_loss_scores.extend(batch_responses_gpt2_loss_scores)
    return all_responses_gpt2_loss_scores

def get_dailog_resp_loss(batch_threads_and_resps, batch_resps, dialog_model, dialog_tokenizer, device):
    with torch.no_grad():
        encoded_resps = dialog_tokenizer(batch_resps, padding=True, return_tensors="pt").to(device)
        encoded_threads_and_resps = dialog_tokenizer(batch_threads_and_resps, padding=True, return_tensors="pt").to(device)
        outputs = dialog_model(**encoded_threads_and_resps)
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = encoded_threads_and_resps['input_ids'][..., 1:].contiguous()
        shift_mask = encoded_threads_and_resps['attention_mask'][..., :-1].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(shift_logits.size(0), shift_logits.size(1))
        # loss.shape = (batch_size, max_thread_and_resp_len)
        loss_list = loss.cpu().tolist()
        # Remove the padding tokens from the loss_list
        loss_list = [l[:shift_mask[i].sum().item()] for i, l in enumerate(loss_list)]
        resp_toks = encoded_resps['input_ids'].cpu().tolist()
        resp_mask = encoded_resps['attention_mask']
        resp_toks_list = [r[:resp_mask[i].sum().item()] for i, r in enumerate(resp_toks)]
        # Compute the loss for each response
        per_response_loss_list = [l[-len(r):] for l, r in zip(loss_list, resp_toks_list)]
        per_response_loss = torch.tensor([np.mean(l) for l in per_response_loss_list])
        # Convert nans to 100.0
        per_response_loss[torch.isnan(per_response_loss)] = 100.0
        return per_response_loss.cpu().tolist()

def get_batched_dialog_loss(threads_and_resps, last_resps, dialog_model, dialog_tokenizer, device, batch_size=32):
    all_resps_dialog_loss = list()
    for i in range(0, len(threads_and_resps), batch_size):
        batch_threads_and_resps = threads_and_resps[i:i+batch_size]
        batch_resps = last_resps[i:i+batch_size]
        # Add eos token to the end of the response
        # batch_resps = [resp + dialog_tokenizer.eos_token for resp in batch_resps]
        batch_resps_dialog_loss_scores = get_dailog_resp_loss(batch_threads_and_resps, batch_resps, dialog_model, dialog_tokenizer, device)
        all_resps_dialog_loss.extend(batch_resps_dialog_loss_scores)
    return all_resps_dialog_loss

def get_sentiment_score(batch_responses, sentiment_pipeline, current_batch_post_threads_and_predictions=None):
    sentiment_scores = sentiment_pipeline(batch_responses)
    # print(f"Sentiment scores: {sentiment_scores}")
    for i, sentiment_score in enumerate(sentiment_scores):
        if sentiment_score['label'] == 'NEGATIVE':
            sentiment_neg_score = sentiment_score['score']
            sentiment_pos_score = 1.0 - sentiment_neg_score
        else:
            sentiment_pos_score = sentiment_score['score']
            sentiment_neg_score = 1.0 - sentiment_pos_score
        current_batch_post_threads_and_predictions[i][1]["sentiment"] = [sentiment_pos_score, sentiment_neg_score]
    return current_batch_post_threads_and_predictions

def get_cola_fluency_score(cola_pipeline_result):
    if cola_pipeline_result["label"] == "LABEL_1":
        return cola_pipeline_result["score"]
    elif cola_pipeline_result["label"] == "LABEL_0":
        return 1 - cola_pipeline_result["score"]
    else:
        log.error(f"Invalid label: {cola_pipeline_result['label']}. Terminating...")
        breakpoint()

def get_regressor_predictions(all_responses, regressor_model, regressor_tokenizer, device, clamp_range=None):
    with torch.no_grad():
        # Prepare input_dict for the responses
        all_model_inputs_tokenized = regressor_tokenizer.batch_encode_plus(all_responses, padding=True, add_special_tokens=False, return_tensors="pt")
        input_ids, attention_mask = all_model_inputs_tokenized['input_ids'], all_model_inputs_tokenized['attention_mask']
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, :512]
            attention_mask = attention_mask[:, :512]
        # Create testing instance for model
        input_dict = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
        # Forward
        outputs = regressor_model(**input_dict)
        logits = outputs.logits
        if clamp_range is not None:
            logits = torch.clamp(logits, *clamp_range)
        # logits are the predictions
        predictions = logits.squeeze().tolist()
        if type(predictions) == float:
            predictions = [predictions]
        assert len(predictions) == len(all_responses), breakpoint()
    return predictions

def get_batched_regressor_predictions(all_responses, regressor_model, regressor_tokenizer, device, batch_size=16, clamp_range=None):
    regressor_model.eval()
    all_responses_yelp_sentiment_scores = list()
    for i in range(0, len(all_responses), batch_size):
        batch_responses = all_responses[i:i+batch_size]
        batch_sentiment_scores = get_regressor_predictions(batch_responses, regressor_model, regressor_tokenizer, device, clamp_range=clamp_range)
        all_responses_yelp_sentiment_scores.extend(batch_sentiment_scores)
    return all_responses_yelp_sentiment_scores

# TODO: Update this to include all metrics
def get_batched_metrics_for_threads(current_batch_post_threads, extra_args, batch_size = 16, last_responses = None, thresholds_dict = None):
    # NOTE: current_batch_post_threads should be a list of full threads (i.e. each thread should contain eos_token at the end)
    # 		they should use tokenizer.eos_token as the eos_token
    # logging.info(f"Extra args: {extra_args.keys()}")
    if thresholds_dict is None:
        # Initialize the thresholds from extra_args
        thresholds_dict = extra_args["thresholds_dict"]
    if thresholds_dict is None:
        # Initialize from attributes dict
        thresholds_dict = extra_args["attributes_dict"]
    # tokenizer = extra_args["tokenizer"]
    device = extra_args["device"]

    # Full thread should contain eos_token at the end
    last_response_metric_results = dict()
    current_batch_post_threads_and_predictions = None
    for metric_name, thresholds in thresholds_dict.items():
        if metric_name == "toxichat_off":
            offensive_model = extra_args["offensive_model"]
            offensive_tokenizer = extra_args["offensive_tokenizer"]
            # current_batch_post_threads_and_predictions = get_offensive_scores(current_batch_post_threads, offensive_model, offensive_tokenizer, device, eos_seperator=tokenizer.eos_token)
            current_batch_post_threads_and_predictions = get_offensive_scores_batched(current_batch_post_threads, offensive_model, offensive_tokenizer, device, 
                                                            current_batch_post_threads_and_predictions=current_batch_post_threads_and_predictions, 
                                                            eos_seperator=offensive_tokenizer.eos_token, batch_size=batch_size)
            if len(current_batch_post_threads_and_predictions) != len(current_batch_post_threads):
                # Problem in one of the sub-batches. Abort!
                return None
            # Check if each element in offensive predictions contain n_utterances + 1 predictions
            # Compute n_utterances for each thread and verify if we have the right number of offensive predictions
            # NOTE: Only exactly empty response will be ignored. Otherwise, even a response with all spaces will be counted
            n_utterances = [len([e for e in thread.split(offensive_tokenizer.eos_token) if e != ""]) for thread in current_batch_post_threads]
            # [len(e[1]['offensive']) for e in current_batch_post_threads_and_predictions]
            assert all(len(e[1]['offensive']) == n_utterance for e, n_utterance in zip(current_batch_post_threads_and_predictions, n_utterances)), breakpoint()
            if type(thresholds) == int:
                # Varied threshold version. Only return the values.
                metric_values = [e[1]['offensive'][-1][-1][thresholds] for e in current_batch_post_threads_and_predictions]
                last_response_metric_results[metric_name] = {"values": metric_values}
            else:
                metric_values = [e[1]['offensive'][-1][-1][thresholds[0]] for e in current_batch_post_threads_and_predictions]
                threshold_avg = (thresholds[1] + thresholds[2])/2.0
                # print(f"Metric threshold avg = {threshold_avg}")
                # Find difference between metric value and threshold
                metric_diffs = np.array([abs(metric_value - threshold_avg) for metric_value in metric_values])
                # print(f"Metric diffs = {metric_diffs}")
                # Compute Threshold satisfaction
                threshold_satisfied = [thresholds[1] <= metric_value <= thresholds[2] for metric_value in metric_values]
                # print(f"Threshold satisfied = {threshold_satisfied}")
                last_response_metric_results[metric_name] = {"values": metric_values, "threshold_satisfied": threshold_satisfied, "diffs": metric_diffs}
        elif metric_name == "toxichat_stance":
            stance_model = extra_args["stance_model"]
            stance_tokenizer = extra_args["stance_tokenizer"]
            # current_batch_post_threads_and_predictions = get_stance_scores(current_batch_post_threads, stance_model, stance_tokenizer, device, eos_seperator=tokenizer.eos_token)
            current_batch_post_threads_and_predictions = get_stance_scores_batched(current_batch_post_threads, stance_model, stance_tokenizer, device, 
                                                            current_batch_post_threads_and_predictions=current_batch_post_threads_and_predictions, 
                                                            eos_seperator=stance_tokenizer.eos_token, batch_size=batch_size)
            if len(current_batch_post_threads_and_predictions) != len(current_batch_post_threads):
                # Problem in one of the sub-batches. Abort!
                return None
            # Check if each element in stance predictions contain n_utterances + 1 predictions
            # Compute n_utterances for each thread and verify if we have the right number of stance predictions
            # NOTE: Only exactly empty response will be ignored. Otherwise, even a response with all spaces will be counted
            n_utterances = [len([e for e in thread.split(stance_tokenizer.eos_token) if e != ""]) for thread in current_batch_post_threads]
            n_stance_pairs = [comb(e,2) for e in n_utterances]
            # [len(e[1]['stance']) for e in current_batch_post_threads_and_predictions]
            assert all(len(e[1]['stance']) == n_stance_pair for e, n_stance_pair in zip(current_batch_post_threads_and_predictions, n_stance_pairs)), breakpoint()
            if type(thresholds) == int:
                # Varied threshold version. Only return the values.
                try:
                    metric_values = [e[1]['stance'][-1][-1][thresholds] for e in current_batch_post_threads_and_predictions]
                    metric_values = [e[1]['stance'][-1][-1][0] for e in current_batch_post_threads_and_predictions]
                except IndexError:
                    breakpoint()
                last_response_metric_results[metric_name] = {"values": metric_values}
            else:
                metric_values = [e[1]['stance'][-1][-1][thresholds[0]] for e in current_batch_post_threads_and_predictions]
                threshold_avg = (thresholds[1] + thresholds[2])/2.0
                # print(f"Metric threshold avg = {threshold_avg}")
                # Find difference between metric value and threshold
                metric_diffs = np.array([abs(metric_value - threshold_avg) for metric_value in metric_values])
                # print(f"Metric diffs = {metric_diffs}")
                # Compute Threshold satisfaction
                threshold_satisfied = [thresholds[1] <= metric_value <= thresholds[2] for metric_value in metric_values]
                # print(f"Threshold satisfied = {threshold_satisfied}")
                last_response_metric_results[metric_name] = {"values": metric_values, "threshold_satisfied": threshold_satisfied, "diffs": metric_diffs}
        elif metric_name == "yelp_sentiment":
            # Use yelp_sentiment_regression model to make predictions on the last_responses
            assert last_responses is not None, breakpoint()
            yelp_sentiment_model = extra_args["yelp_sentiment_model"]
            yelp_sentiment_tokenizer = extra_args["yelp_sentiment_tokenizer"]
            last_response_metric_results[metric_name] = {"values": get_batched_regressor_predictions(last_responses, yelp_sentiment_model, yelp_sentiment_tokenizer, device, batch_size=batch_size, clamp_range=[1.0, 5.0])}
        elif metric_name == "formality":
            # Use formality_regression model to make predictions on the last_responses
            assert last_responses is not None, breakpoint()
            formality_model = extra_args["formality_model"]
            formality_tokenizer = extra_args["formality_tokenizer"]
            last_response_metric_results[metric_name] = {"values": get_batched_regressor_predictions(last_responses, formality_model, formality_tokenizer, device, batch_size=batch_size, clamp_range=[-3.0, 3.0])}
    if "gpt2_model" in extra_args and extra_args["gpt2_model"] is not None:
        # Also extra the gpt2 loss and save it in the last_response_metric_results
        assert last_responses is not None, "last_responses should not be None if gpt2_model is provided"
        last_response_gpt_loss = get_batched_gpt2_metrics(last_responses, extra_args["gpt2_model"], extra_args["gpt2_tokenizer"], device, batch_size=batch_size)
        last_response_metric_results["gpt_loss"] = {"values": last_response_gpt_loss}
    if "cola_pipeline" in extra_args and extra_args["cola_pipeline"] is not None:
        assert last_responses is not None and len(last_responses) > 0, "last_responses should not be None if cola_pipeline is provided"
        last_response_cola_fluency_scores = [get_cola_fluency_score(e) for e in extra_args["cola_pipeline"](last_responses)]
        last_response_metric_results["cola_fluency"] = {"values": last_response_cola_fluency_scores}
    if "dialog_model" in extra_args and extra_args["dialog_model"] is not None:
        assert last_responses is not None and len(last_responses) > 0, "last_responses should not be None if dialog_model is provided"
        last_response_dialog_loss = get_batched_dialog_loss(current_batch_post_threads, last_responses, extra_args["dialog_model"], extra_args["dialog_tokenizer"], device, batch_size=batch_size)
        last_response_metric_results["dialog_loss"] = {"values": last_response_dialog_loss}
    return last_response_metric_results


metric_to_key_mapping = {
    "toxichat_off": ["safe", "offensive"],
    "toxichat_stance": ["neutral", "agree", "disagree"],
    "sentiment": ["negative", "positive"],
    "yelp_sentiment": ["positive"],
    "formality": ["formal"],
}

def get_per_response_metrics_from_results(metric_results, thresholds_dict, num_responses):
    new_response_metrics = [dict() for _ in range(num_responses)]
    for metric_name, thresholds in thresholds_dict.items():
        if metric_name in metric_results:
            if type(thresholds) == int:
                metric_key = metric_to_key_mapping[metric_name][thresholds]
            elif type(thresholds) == tuple:
                metric_key = metric_to_key_mapping[metric_name][thresholds[0]]
            else:
                print(f"Error! Unknown type for thresholds = {type(thresholds)}, thresholds = {thresholds}")
                breakpoint()
            for k in range(len(new_response_metrics)):
                new_response_metrics[k][metric_key] = metric_results[metric_name]['values'][k]
    if "gpt_loss" in metric_results:
        for k in range(len(new_response_metrics)):
            new_response_metrics[k]["gpt_loss"] = metric_results["gpt_loss"]['values'][k]
    if "cola_fluency" in metric_results:
        for k in range(len(new_response_metrics)):
            new_response_metrics[k]["cola_fluency"] = metric_results["cola_fluency"]['values'][k]
    if "dialog_loss" in metric_results:
        for k in range(len(new_response_metrics)):
            new_response_metrics[k]["dialog_loss"] = metric_results["dialog_loss"]['values'][k]
    return new_response_metrics


def get_metric_threshold_satisfaction(metrics, thresholds_dict):
    metric_threshold_satisfaction = dict()
    for metric_name, thresholds in thresholds_dict.items():
        metric_key = metric_to_key_mapping[metric_name][thresholds[0]]
        metric_value = metrics[metric_key]
        metric_threshold_satisfaction[metric_key] = thresholds[1] <= metric_value <= thresholds[2]
    return metric_threshold_satisfaction
