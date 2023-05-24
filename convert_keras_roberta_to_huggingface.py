
from transformers import AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup,  AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, RobertaPreTrainedModel, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import autograd
import torch
import keras
import numpy as np

import os
import logging
from utils.utils import make_dir_if_not_exists, save_in_txt, load_from_jsonl
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from time import time


from transformers.utils import ExplicitEnum
from transformers import AutoModelForSequenceClassification, RobertaModel, convert_tf_weight_name_to_pt_weight_name
# class TransposeType(ExplicitEnum):
#     """
#     Possible ...
#     """

#     NO = "no"
#     SIMPLE = "simple"
#     CONV1D = "conv1d"
#     CONV2D = "conv2d"
from utils.comet_utils import RobertaClassificationHead

def convert_keras_weights_and_save_to_pytorch(comet_critic_model, save_dir):
    names = [weight.name for layer in comet_critic_model.layers for weight in layer.weights]
    temp_file = os.path.join("")
    weights = comet_critic_model.get_weights()
    names_to_weights = {name: weight for name, weight in zip(names, weights)}
    # We need to convert this into RobertaForSequenceClassification model
    # 2. Load the RobertaForSequenceClassification model
    start_time = time()
    logging.info(f"Loading the RobertaForSequenceClassification model from roberta-large-mnli")

    config = AutoConfig.from_pretrained("roberta-large-mnli")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    config.num_labels = 1
    # roberta_model_pytorch = AutoModelForSequenceClassification.from_pretrained("roberta-large", config=config)
    roberta_model_pytorch = RobertaModel(config, add_pooling_layer=True)
    logging.info(f"Loaded the RobertaForSequenceClassification model and tokenizer in {time() - start_time:.2f} seconds")
    # We will also need to transpose some of the weights
    # Ref: https://github.com/huggingface/transformers/blob/dfeb5aa6a9d0cb95c008854c4e67ceecfeff6ccc/src/transformers/modeling_tf_pytorch_utils.py#L487
    # last 4 pt_names = pt_names[-4:] = ['weight', 'bias', 'weight', 'bias']
    # We need to adjust a few names in this list
    # roberta_model_pytorch_state_dict['classifier.out_proj.weight'].shape = torch.Size([1, 1024])
    # roberta_model_pytorch_state_dict['classifier.out_proj.bias'].shape = torch.Size([3])
    # roberta_model_pytorch_state_dict['classifier.dense.weight'].shape = torch.Size([1024, 1024])
    # roberta_model_pytorch_state_dict['classifier.dense.bias'].shape = torch.Size([1024])
    # names_to_weights['dense/kernel:0'].shape = (1024, 512)
    # names_to_weights['dense/bias:0'].shape = (512,)
    # names_to_weights['dense_1/kernel:0'].shape = (512, 1)
    # names_to_weights['dense_1/bias:0'].shape = (1,)
    # last 4 names = names[-4:] = ['dense/kernel:0', 'dense/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0']
    # We need to convert them to ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    pt_names = [convert_tf_weight_name_to_pt_weight_name(name)[0] for name in names]
    fixed_pt_names = pt_names[:-4] + ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    keras_weight_transpose = [convert_tf_weight_name_to_pt_weight_name(name)[1] for name in names]
    fixed_pt_names_to_weights = {name: weight for name, weight in zip(fixed_pt_names, weights)}
    fixed_pt_name_to_transpose = {name: transpose.name for name, transpose in zip(fixed_pt_names, keras_weight_transpose)}

    # Get state dict from roberta_model_pytorch
    roberta_model_pytorch_state_dict = roberta_model_pytorch.state_dict()
    pytorch_model_weight_names = list(roberta_model_pytorch_state_dict.keys())
    robert_model_weight_names_file = os.path.join(save_dir, "roberta_model_weight_names.txt")
    save_in_txt(pytorch_model_weight_names, robert_model_weight_names_file)
    # len(pytorch_model_weight_names) = 392

    fixed_roberta_pt_names = [e.replace("roberta.", "") for e in fixed_pt_names[:-4]]
    fixed_roberta_pt_names_to_weights = {name: weight for name, weight in zip(fixed_roberta_pt_names, weights[:-4])}
    fixed_roberta_pt_name_to_transpose = {name: transpose.name for name, transpose in zip(fixed_roberta_pt_names, keras_weight_transpose)}
    # Check the intersection of pytorch_model_weight_names and pt_names
    common_names = set(pytorch_model_weight_names).intersection(set(fixed_roberta_pt_names))
    logging.info(f"Len common_names = {len(common_names)} with pytorch names {len(pytorch_model_weight_names)} and keras names {len(fixed_roberta_pt_names)}")
    #  len(common_names) = 393
    # What are the names that are not in common_names
    uncommon_names = set(pytorch_model_weight_names).difference(set(fixed_roberta_pt_names))
    #  uncommon_names = {'embeddings.position_ids'}
    # We can keep the embeddings.position_ids as it is
    uncommon_og_names = set(fixed_roberta_pt_names).difference(set(pytorch_model_weight_names))
    # This is empty now
    # {'roberta.pooler.dense.bias', 'roberta.pooler.dense.weight'}
    # fixed_pt_names_to_weights['roberta.pooler.dense.bias'].shape = (1024,)
    # fixed_pt_names_to_weights['roberta.pooler.dense.weight'].shape = (1024, 1024)

    # Assert that state_dict size and weights shape are the same for all common_names
    
    size_check = [roberta_model_pytorch_state_dict[name].shape == fixed_roberta_pt_names_to_weights[name].shape for name in common_names]
    incorrect_sizes = [f"{name} pytorch shape {roberta_model_pytorch_state_dict[name].shape} vs {fixed_roberta_pt_names_to_weights[name].shape} keras shape" for name in common_names if roberta_model_pytorch_state_dict[name].shape != fixed_roberta_pt_names_to_weights[name].shape]
    incorrect_sizes_names = [name for name in common_names if roberta_model_pytorch_state_dict[name].shape != fixed_roberta_pt_names_to_weights[name].shape]
    # (Pdb) incorrect_sizes_names
    # ['encoder.layer.10.intermediate.dense.weight', 'encoder.layer.11.output.dense.weight', 'encoder.layer.2.intermediate.dense.weight', 'encoder.layer.17.output.dense.weight', 'encoder.layer.16.output.dense.weight', 'encoder.layer.15.intermediate.dense.weight', 'encoder.layer.20.output.dense.weight', 'encoder.layer.12.intermediate.dense.weight', 'encoder.layer.23.intermediate.dense.weight', 'encoder.layer.23.output.dense.weight', 'encoder.layer.19.output.dense.weight', 'encoder.layer.9.intermediate.dense.weight', 'encoder.layer.6.output.dense.weight', 'encoder.layer.18.intermediate.dense.weight', 'encoder.layer.21.output.dense.weight', 'encoder.layer.4.intermediate.dense.weight', 'encoder.layer.5.intermediate.dense.weight', 'encoder.layer.17.intermediate.dense.weight', 'encoder.layer.15.output.dense.weight', 'encoder.layer.12.output.dense.weight', 'encoder.layer.7.output.dense.weight', 'encoder.layer.3.output.dense.weight', 'encoder.layer.5.output.dense.weight', 'encoder.layer.9.output.dense.weight', 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.13.intermediate.dense.weight', 'encoder.layer.3.intermediate.dense.weight', 'encoder.layer.0.output.dense.weight', 'encoder.layer.22.intermediate.dense.weight', 'encoder.layer.4.output.dense.weight', 'encoder.layer.8.intermediate.dense.weight', 'encoder.layer.14.output.dense.weight', 'encoder.layer.6.intermediate.dense.weight', 'encoder.layer.1.output.dense.weight', 'encoder.layer.8.output.dense.weight', 'encoder.layer.1.intermediate.dense.weight', 'encoder.layer.18.output.dense.weight', 'encoder.layer.21.intermediate.dense.weight', 'encoder.layer.16.intermediate.dense.weight', 'encoder.layer.20.intermediate.dense.weight', 'encoder.layer.2.output.dense.weight', 'encoder.layer.22.output.dense.weight', 'encoder.layer.13.output.dense.weight', 'encoder.layer.14.intermediate.dense.weight', 'encoder.layer.7.intermediate.dense.weight', 'encoder.layer.11.intermediate.dense.weight', 'encoder.layer.19.intermediate.dense.weight', 'encoder.layer.10.output.dense.weight']
    # (Pdb) len(incorrect_sizes_names)
    # 48
    # Counter(fixed_roberta_pt_name_to_transpose.values()) = Counter({'NO': 246, 'SIMPLE': 145})
    simple_transpose_names = [name for name, transpose in fixed_roberta_pt_name_to_transpose.items() if transpose == "SIMPLE"]
    # len(simple_transpose_names) = 145
    assert set(incorrect_sizes_names).issubset(set(simple_transpose_names))

    # Convert weights in fixed_roberta_pt_names_to_weights to pytorch tensors
    def transpose_keras_numpy_weight_to_pytorch_tensor(pt_weight, array, transpose):
        if transpose == "SIMPLE":
            array = np.transpose(array)

        if len(pt_weight.shape) < len(array.shape):
            array = np.squeeze(array)
        elif len(pt_weight.shape) > len(array.shape):
            array = np.expand_dims(array, axis=0)

        if list(pt_weight.shape) != list(array.shape):
            try:
                array = np.reshape(array, pt_weight.shape)
            except AssertionError as e:
                e.args += (pt_weight.shape, array.shape)
                raise e

        try:
            assert list(pt_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += (pt_weight.shape, array.shape)
            raise e

        # logger.warning(f"Initialize PyTorch weight {pt_weight_name}")
        # Make sure we have a proper numpy array
        if np.isscalar(array):
            array = np.array(array)
        return torch.from_numpy(array)
    logging.info(f"Transposing and converting {len(fixed_roberta_pt_names)} keras weights from numpy to pytorch")
    fixed_roberta_pt_names_to_pytorch_tensors = {name: transpose_keras_numpy_weight_to_pytorch_tensor(roberta_model_pytorch_state_dict[name], fixed_roberta_pt_names_to_weights[name], fixed_roberta_pt_name_to_transpose[name]) for name in fixed_roberta_pt_names}

    # Now check if the weight shapes are same
    sanity_check = [fixed_roberta_pt_names_to_pytorch_tensors[name].shape == roberta_model_pytorch_state_dict[name].shape for name in fixed_roberta_pt_names]
    name = fixed_roberta_pt_names[0]
    logging.info(f"Sanity check = {all(sanity_check)}")
    
    # Create a new state dict with the fixed weights
    fixed_roberta_model_pytorch_state_dict = roberta_model_pytorch_state_dict.copy()
    for name in fixed_roberta_pt_names: fixed_roberta_model_pytorch_state_dict[name] = fixed_roberta_pt_names_to_pytorch_tensors[name]
    fixed_roberta_model_pytorch_state_dict[name]
    # Update roberta_model_pytorch with new state dict
    roberta_model_pytorch.load_state_dict(fixed_roberta_model_pytorch_state_dict)
    logging.info(f"Updated the state dict of roberta_model_pytorch with transposed keras weights")
    # Save the model
    roberta_model_pytorch.save_pretrained(save_dir)
    logging.info(f"Saved the model to {save_dir}")
    # Save the tokenizer too
    tokenizer.save_pretrained(save_dir)
    logging.info(f"Saved the tokenizer to {save_dir}")
    # Create a sequential from final 4 classification layers of fixed_pt_names: ['classifier.dense.weight', 'classifier.dense.bias', 'classifier.out_proj.weight', 'classifier.out_proj.bias']
    # names_to_weights['dense/kernel:0'].shape = (1024, 512)
    # names_to_weights['dense/bias:0'].shape = (512,)
    # names_to_weights['dense_1/kernel:0'].shape = (512, 1)
    # names_to_weights['dense_1/bias:0'].shape = (1,)

    # Keras model dropout layers
    post_pooler_dropout = comet_critic_model.layers[2].rate
    classifier_dropout = comet_critic_model.layers[4].rate
    # Both are = 0.1. so we will just use one of them
    # fixed_pt_names_to_weights['classifier.dense.weight'].shape = (1024, 512)
    # fixed_pt_names_to_weights['classifier.dense.bias'].shape = (512,)
    # fixed_pt_names_to_weights['classifier.out_proj.weight'].shape = (512, 1)
    # fixed_pt_names_to_weights['classifier.out_proj.bias'].shape = (1,)
    
    custom_roberta_classification_head = RobertaClassificationHead(1024, 512, 1, post_pooler_dropout)
    logging.info(f"Created a custom_roberta_classification_head with params hidden_size = 1024, intermediate_size = 512, num_labels = 1, dropout = {post_pooler_dropout}")
    custom_roberta_classification_head_state_dict = custom_roberta_classification_head.state_dict()
    # custom_roberta_classification_head_state_dict.keys() = odict_keys(['dense.weight', 'dense.bias', 'out_proj.weight', 'out_proj.bias'])
    fixed_pt_names[-4:]
    # We need to transpose the weights
    keras_classifier_to_pytorch_shapes = {name.replace("classifier.", ""): fixed_pt_names_to_weights[name] if ".weight" not in name else np.transpose(fixed_pt_names_to_weights[name]) for name in fixed_pt_names[-4:]}
    keras_classifier_state_dict = {name.replace("classifier.", ""): torch.from_numpy(keras_classifier_to_pytorch_shapes[name.replace("classifier.", "")]) for name in fixed_pt_names[-4:]}
    name = "dense.weight"
    # Assert that the shapes are same
    sanity_check = [custom_roberta_classification_head_state_dict[name].shape == keras_classifier_state_dict[name].shape for name in custom_roberta_classification_head_state_dict.keys()]
    logging.info(f"Classifier Sanity check = {sanity_check}")

    # Update the state dict of custom_roberta_classification_head with keras_classifier_state_dict
    custom_roberta_classification_head_state_dict.update(keras_classifier_state_dict)
    logging.info(f"Updated the state dict of custom_roberta_classification_head with keras_classifier_state_dict")
    # Save the custom_roberta_classification_head with torch
    classification_torch_save_path = os.path.join(save_dir, "custom_roberta_classification_head.pt")
    torch.save(custom_roberta_classification_head_state_dict, classification_torch_save_path)
    logging.info(f"Saved the custom_roberta_classification_head to {classification_torch_save_path}")
    # Additional OLD Debug: Save the list of names
    names_save_path = os.path.join(save_dir, "keras_layer_names.txt")
    save_in_txt(names, names_save_path)
    fixed_names_save_path = os.path.join(save_dir, "fixed_keras_layer_names.txt")
    save_in_txt(fixed_pt_names, fixed_names_save_path)
    

def main():
    save_dir = "saved_models/comet_critic_keras_to_pytorch"
    make_dir_if_not_exists(save_dir)
    # 1. Load the Comet Critic model
    start_time = time()
    comet_critic_model = "data/symbolic_knowledge_distillation/original_from_paper/model~model=roberta-large-mnli~lr=5e-06~bs=128~dropout=0.10~original_from_paper"
    logging.info(f"Loading the COMET critic model from {comet_critic_model} and tokenizer from roberta-large-mnli")
    from huggingface_hub import from_pretrained_keras
    comet_critic_model = from_pretrained_keras(comet_critic_model)
    comet_critic_model.summary()
    logging.info(f"Loaded the COMET Critic model and tokenizer in {time() - start_time:.2f} seconds")
    
    # Model summary
    #     _________________________________________________________________
    #  Layer (type)                Output Shape              Param #
    # =================================================================
    #  input_1 (InputLayer)        [(None, None)]            0

    #  roberta (Custom>TFRobertaMa  {'pooler_output': (None,  355359744
    #  inLayer)                     1024),
    #                               'last_hidden_state': (N
    #                              one, None, 1024)}

    #  dropout_73 (Dropout)        (None, 1024)              0

    #  dense (Dense)               (None, 512)               524800

    #  dropout_74 (Dropout)        (None, 512)               0

    #  dense_1 (Dense)             (None, 1)                 513

    # =================================================================
    # Total params: 355,885,057
    # Trainable params: 355,885,057
    # Non-trainable params: 0
    # _________________________________________________________________
    # Check: https://huggingface.co/transformers/v4.3.3/_modules/transformers/models/roberta/modeling_tf_roberta.html
    # Within that search for TFRobertaMainLayer
    # It has both last_hidden and pooler output. I am not sure which one the keras model is using
    
    # Checking the activation function in comet_critic_model
    # (Pdb) comet_critic_model.layers[3].__dict__.keys()
    # dict_keys(['_self_setattr_tracking', '_obj_reference_counts_dict', '_auto_get_config', '_auto_config', '_instrumented_keras_api', '_instrumented_keras_layer_class', '_instrumented_keras_model_class', '_input_spec', '_build_input_shape', '_saved_model_inputs_spec', '_saved_model_arg_spec', '_activity_regularizer', '_trainable_weights', '_non_trainable_weights', '_updates', '_thread_local', '_callable_losses', '_losses', '_metrics', '_metrics_lock', '_autocast', '_self_tracked_trackables', '_inbound_nodes_value', '_outbound_nodes_value', '_call_spec', '_dynamic', '_initial_weights', '_auto_track_sub_layers', '_preserve_input_structure_in_config', '_name_scope_on_declaration', '_captured_weight_regularizer', 'units', 'activation', 'use_bias', 'kernel_initializer', 'bias_initializer', 'kernel_regularizer', 'bias_regularizer', 'kernel_constraint', 'bias_constraint', '_supports_masking', '_name', '_trainable', '_dtype_policy', '_compute_dtype_object', '_stateful', '_self_unconditional_checkpoint_dependencies', '_self_unconditional_dependency_names', '_self_unconditional_deferred_dependencies', '_self_name_based_restores', 'kernel', 'bias', 'built', '_serialized_attributes', '_self_saveable_object_factories', '_self_update_uid'])
    # comet_critic_model.layers[3].activation <function gelu at 0x7f0c9d1655a0>
    # comet_critic_model.layers[4].activation <function sigmoid at 0x7f0c9d165900>
    # NOTE: Created a new model and saved it
    # convert_keras_weights_and_save_to_pytorch(comet_critic_model, save_dir)


    # Now load the pytorch model and check if its predictions are same as keras model

    # 1.1 Load the state dict from 
    start_time = time()
    roberta_model_pytorch = RobertaModel.from_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    logging.info(f"Loaded the Roberta model and tokenizer from {save_dir} in {time() - start_time:.2f} seconds")

    custom_roberta_classification_head = RobertaClassificationHead(1024, 512, 1)
    classification_torch_save_path = os.path.join(save_dir, "custom_roberta_classification_head.pt")
    custom_roberta_classification_head.load_state_dict(torch.load(classification_torch_save_path))
    logging.info(f"Loaded the custom_roberta_classification_head from {classification_torch_save_path}")
    
    from utils.comet_utils import get_comet_keras_input_and_labels
    # Test example from ATOMIC
    atomic_test_dict = {"head": "PersonX decides to see a therapist", "relation": "xEffect", "tail": "feels better", "split": "train", "rec_0.6": True, "rec_0.9": True, "rec_0.5": True, "rec_0.7": True, "rec_0.8": True, "p_valid_model": 0.9934503436088562}
    atomic_test_dict["valid"] = 1.0
    # Load 1000 examples from ATOMIC
    atomic_jsonl_file = os.path.join("data/symbolic_knowledge_distillation/downloaded", "ATOMIC10X.jsonl")
    N = 100
    N = 50
    atomic_dataset = load_from_jsonl(atomic_jsonl_file, max_lines=N)
    # Add valid field to the atomic_dataset
    valid_atomic_dicts = [d | {"valid": 1.0} for d in atomic_dataset]
    test_x, label = get_comet_keras_input_and_labels(valid_atomic_dicts, tokenizer)
    comet_critic_pred = comet_critic_model.predict(test_x)
    logging.info(f"Comet Critic prediction: {comet_critic_pred}")
    # comet_critic_pred = array([[0.9930738]], dtype=float32)
    # Process the text_x through roberta_model_pytorch
    roberta_model_pytorch.eval()
    with torch.no_grad():
        # text_x is numpy ndarray of shape (1, 17)
        input_ids = torch.tensor(test_x)
        outputs = roberta_model_pytorch(input_ids)
        last_hidden_state = outputs.last_hidden_state
        # last_hidden_state.shape = torch.Size([1, 17, 1024])
        pooler_output = outputs.pooler_output
        # pooler_output.shape = torch.Size([1, 1024])
        # Now pass the pooler_output through the custom_roberta_classification_head
        pytorch_critic_pred = custom_roberta_classification_head.forward(pooler_output)
        logging.info(f"Pytorch Critic prediction from pooler: {pytorch_critic_pred}")
        pytorch_critic_pred_last_head = custom_roberta_classification_head.forward(last_hidden_state[:, 0, :])
        logging.info(f"Pytorch Critic prediction from last hidden: {pytorch_critic_pred}")
        # INFO:root:Comet Critic prediction: [[0.9930738]]
        # INFO:root:Pytorch Critic prediction from pooler: tensor([[0.9901]])
        # INFO:root:Pytorch Critic prediction from last hidden: tensor([[0.8654]])
        # Conclusion: Use pooler output for classification
        diff = np.abs(comet_critic_pred - pytorch_critic_pred.numpy())
        logging.info(f"Total Diff between comet_critic_pred and pytorch_critic_pred for {N} instances: {diff.sum()}")
        breakpoint()

    # Total Diff between comet_critic_pred and pytorch_critic_pred for 100 instances: 1.5174801349639893
    # Without Dropout: INFO:root:Total Diff between comet_critic_pred and pytorch_critic_pred for 50 instances: 1.8537044525146484e-05
if __name__ == "__main__":
    main()