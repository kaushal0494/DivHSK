
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """


import copy
import math
import os
import warnings
import sys 
import re 
from scipy.spatial.distance import directed_hausdorff
import numpy as np 
from torch.nn.utils.rnn import pad_sequence
import gensim.downloader as api

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from itertools import chain 
import gensim 

from ...activations import ACT2FN
from ...file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_t5 import T5Config
from .tokenization_t5 import T5Tokenizer
from ..bart import BartForConditionalGeneration
from difflib import get_close_matches

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words.extend(['</s>','The','made','.','','I','said','also','des','would','A','u','sh','<pad>','<unk>','nji','kul','mur','...','yr','s','b'])

import spacy.cli
nlp = spacy.cli.download('en_core_web_lg')
nlp_l = spacy.load('en_core_web_lg')
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (:obj:`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:
                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24
    Example::
            # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
            model = T5ForConditionalGeneration.from_pretrained('t5-3b')
            device_map = {0: [0, 1, 2],
                         1: [3, 4, 5, 6, 7, 8, 9],
                         2: [10, 11, 12, 13, 14, 15, 16],
                         3: [17, 18, 19, 20, 21, 22, 23]}
            model.parallelize(device_map)
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example::
        # On a 4 GPU machine with t5-3b:
        model = T5ForConditionalGeneration.from_pretrained('t5-3b')
        device_map = {0: [0, 1, 2],
                     1: [3, 4, 5, 6, 7, 8, 9],
                     2: [10, 11, 12, 13, 14, 15, 16],
                     3: [17, 18, 19, 20, 21, 22, 23]}
        model.parallelize(device_map) # Splits the model across several devices
        model.deparallelize() # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
"""


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""
    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a text-to-text
    denoising generative setting.
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.
            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.
            `What are input IDs? <../glossary.html#input-ids>`__
            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__
            T5 uses the :obj:`pad_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).
            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at `T5 Training
            <./t5.html#training>`__.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in ``[0,
            1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        decoder_head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in ``[0,
            1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`:
            `attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a
            sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.
            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.
            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.
            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5Model
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> # forward pass
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        #print("encoder config : ",encoder_config)
        #print("type : ",type(encoder_config))
        #print("encoder layers : ",encoder_config.num_layers)
        #wordencoder_config = copy.deepcopy(config)
        #ordencoder_config.num_layers = 1 
        #print("w_encoder config : ",wordencoder_config)
        #print("type : ",type(encoder_config))
        #print("w_encoder layers : ",wordencoder_config.num_layers) 
        self.encoder = T5Stack(encoder_config, self.shared)
        # PHG Addition
        self.keyencoder = T5Stack(encoder_config, self.shared)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        #self.wordencoder = T5Stack(wordencoder_config, self.shared)
        #self.bartmodel = BartForConditionalGeneration.from_pretrained('facebook/bart-base', encoder_layers=1)
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/raid/ai20mtech14005/venkatesh/Venkatesh_code/GoogleNews-vectors-negative300.bin', binary=True)
        self.my_dictionary = {'<pad>':torch.tensor(0),'[SEP]':torch.tensor(2),'</s>':torch.tensor(1)}

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.keyencoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        #self.wordencoder.parallelize(self.device_map)
        #self.bartmodel.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.keyencoder.deparallelize()
        self.decoder.deparallelize()
        #self.wordencoder.deparallelize()
        #self.bartmodel.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.keyencoder = self.keyencoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        #self.wordencoder = self.wordencoder.to("cpu")
        #self.bartmodel = self.bartmodel.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.keyencoder.set_input_embeddings(new_embeddings)
        self.encoder.set_input_embeddings(new_embeddings)
        #self.wordencoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
        #self.bartmodel.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_keyencoder(self):
      return self.keyencoder

    def get_encoder(self):
        return self.encoder

    #def get_wordencoder(self):
    #   return self.wordencoder

    def get_decoder(self):
        return self.decoder
    
    """
    Additional Functions - to derive keywords, keyids, subkey ids from input ids
    and attention ---- used to modify input ids as keyword_ids + input_ids to the
    final model during training & testing 
    """
    # 1 
    def get_subkey_representations(self,multihead_attentions,input_ids):
        """
        Inputs : multihead_attentions (32,12,128,128)
                 input_ids (32,128) # batch size is 32 
        Returns: Subkey IDs for given input ids 
        """
        #print("\n\nFunction Name : get_subkey_representations")
        n = multihead_attentions.shape[0]
        subkeys1 = [None]*n
        subkeys2 = [None]*n
        subkeys3 = [None]*n

        for i in range(n):
            mh_attention = multihead_attentions[i] # (12,128,128)
            #print("mh_attention:",mh_attention.shape)
            input_id = input_ids[i] # (128)
            #print("inpuut shape : ",input_id.shape)
            input_sentence = self.tokenizer.decode(input_id,skip_special_tokens=True)
            group1_atts, group2_atts, group3_atts = self.get_subkey_helper(mh_attention,input_id) # 2
            subkeys1[i], subkeys2[i], subkeys3[i] = self.get_subkeyword_ids(group1_atts,group2_atts,group3_atts,input_sentence) # 3
        #print("\n\nFunction Returns [subkeys1, subkeys2, subkeys3] : \n",[subkeys1,subkeys2,subkeys3])
        # [[32,10],[32,10],[32,10]] -> (3,32,10)
        return [subkeys1, subkeys2, subkeys3]

    # 2 
    def get_subkey_helper(self,mh_attention, input_id):
        """
        Input : mh_attention (12 attention heads) (12,128,128)
                input_id (128)
        Returns: group_attentions(using 12 attention heads)
        """
        #print("\n\n Function Name : get_subkey_helper")
        num_heads = mh_attention.shape[0]
        ind_sks_attentions = [None]*num_heads
        # (12,128,128) -> (128,128)
        for i in range(num_heads): 
            ind_sks_attentions[i] = self.get_individual_attention_sks(mh_attention[i],input_id) # 2-1

        group_sk1_attentions, group_sk2_attentions, group_sk3_attentions = self.get_group_attention_sks(ind_sks_attentions) # 2-2
        #print("\n\n Function Returns : group_sk1_attentions, group_sk2_attentions, group_sk3_attentions",group_sk1_attentions, group_sk2_attentions, group_sk3_attentions)
        return group_sk1_attentions, group_sk2_attentions, group_sk3_attentions 

    # 2 - 1
    def get_individual_attention_sks(self,attention,input_id,top_k = 10):
        """
        Input: attention - individual attention (128,128)
               input_id - (128)
               top_k = 10(by default) # takes top 10 keyword into account 
        Returns : top_keywords set along with attentions 
        """
        #print("\n\nFunction Name : get_individual_attention_sks")
        keywords = set()
        attention_wts = torch.sum(attention, dim=0) # size = [nrow, 1]
        vk, ik = attention_wts.topk(top_k)
        attention_store = {}

        for i in ik.reshape(1,top_k):
            curr_key = self.tokenizer.batch_decode(input_id[i],skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #print("\n\ninput keys : \n\n",curr_key)
            for j in range(len(curr_key)):
                if curr_key[j] not in self.my_dictionary:
                    self.my_dictionary[curr_key[j]] = input_id[i][j]
            curr_key = [word for word in curr_key if (not word in stop_words) and len(word)>2]

            for j in range(len(curr_key)):
                if curr_key[j] not in attention_store:
                    attention_store[curr_key[j]] = [1,float(vk[j])]
                else:
                    val = attention_store[curr_key[j]][1]
                    c = attention_store[curr_key[j]][0] + 1
                    attention_store[curr_key[j]] = [c,(val + float(vk[j]))/c]

            keywords = keywords.union(set(curr_key))
        #print("\n\n Function Returns [{x for x in keywords if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())}, attention_store]  : ",[{x for x in keywords if not (x.isdigit() or x[0] == '-' and x[1:].isdigit())}, attention_store])
        return [{x for x in keywords if not (x.isdigit() 
                                            or x[0] == '-' and x[1:].isdigit())}, attention_store] 
    
    # 2 - 2                                         
    def get_group_attention_sks(self,ind_sks_attentions):
        #print("\n\nFunction Name : get_group_attention_sks")
        g1_keys, g2_keys, g3_keys = set(), set(), set()
        for i,j,k in zip(range(4),range(5,8),range(8,12)):
            g1_keys = g1_keys.union(ind_sks_attentions[i][0])
            g2_keys = g2_keys.union(ind_sks_attentions[j][0])
            g3_keys = g3_keys.union(ind_sks_attentions[k][0])
        
        g1_attentions = self.GetAttention(g1_keys,ind_sks_attentions[:4]) # 2 - 2 - 1
        g2_attentions = self.GetAttention(g2_keys,ind_sks_attentions[4:8])
        g3_attentions = self.GetAttention(g3_keys,ind_sks_attentions[8:])

        #print("\n\n Function Returns [[g1_keys,g1_attentions],[g2_keys,g2_attentions],[g3_keys,g3_attentions]] : ",[[g1_keys,g1_attentions],[g2_keys,g2_attentions],[g3_keys,g3_attentions]])
        return [[g1_keys,g1_attentions],[g2_keys,g2_attentions],[g3_keys,g3_attentions]]
    
    # 2 - 2 - 1
    def GetAttention(self,keys,attentions):
        """
        Input : keys - keywords in group 
                attentions - individual attention for 4 single head attention
        Returns: Group Attention 
        """
        #print("\n\nFunction Name : GetAttention")
        d1 = attentions[0][1]
        d2 = attentions[1][1]
        d3 = attentions[2][1]
        d4 = attentions[3][1]
        group_attention = {}
        for k in keys:
            c = 0 
            wts = 0 
            if k in d1:
                c+=1 
                wts+=d1[k][1]
            if k in d2:
                c+=1 
                wts+=d2[k][1]
            if k in d3:
                c+=1 
                wts+=d3[k][1]
            if k in d4:
                c+=1 
                wts+=d4[k][1]
            group_attention[k] = wts/c 
        #print("\n\n Function Returns group_attention : ",group_attention)
        return group_attention 
    
    # 3 
    def get_subkeyword_ids(self,g1,g2,g3,input_sentence = None,n = 5, r = 2, related_keywords = False):
        """
        Inputs : g1, g2, g3 -three group of keywords 
        Returns: sk_ids1, sk_ids2, sk_ids3 - three group of subkey ids 
        """
        #print("\n\nFunction Name : get_subkeyword_ids")
        g_keys1, g_keys2, g_keys3 = self.SelectTopKeywords([g1[0],g2[0],g3[0]],[g1[1],g2[1],g3[1]],n,r) # 3 - 1
        #sk_ids1 = self.tokenizer.convert_tokens_to_ids(g_keys1)
        #sk_ids2 = self.tokenizer.convert_tokens_to_ids(g_keys2)
        #sk_ids3 = self.tokenizer.convert_tokens_to_ids(g_keys3) 
        # tokenizer.encode 
        #sk_ids1 = self.tokenizer.encode(g_keys1)
        #sk_ids2 = self.tokenizer.encode(g_keys2)
        #sk_ids3 = self.tokenizer.encode(g_keys3)

        sk_ids1 = self.customtokens_to_ids(g_keys1) # 3 - 2 
        sk_ids2 = self.customtokens_to_ids(g_keys2)
        sk_ids3 = self.customtokens_to_ids(g_keys3)
        
        if input_sentence is not None:
            keys1 = self.subkeytoKeywordConverter(input_sentence, g_keys1,n,related_keywords) # 3 - 3
            keys2 = self.subkeytoKeywordConverter(input_sentence, g_keys2,n,related_keywords)
            keys3 = self.subkeytoKeywordConverter(input_sentence, g_keys3,n,related_keywords)
            
            #print("Keys 1 : \n",g_keys1)
            #print('Keys 2 : \n',g_keys2)
            #print("Keys 3 : \n",g_keys3)
            
            key_ids1 = self.EncodeKeyword(keys1) # 3 - 4 
            key_ids2 = self.EncodeKeyword(keys2)
            key_ids3 = self.EncodeKeyword(keys3)

            #print("final list1 : ",self.tokenizer.decode(key_ids1))
            #print("final list2 : ",self.tokenizer.decode(key_ids2))
            #print("final list3 : ",self.tokenizer.decode(key_ids3))

            return key_ids1, key_ids2, key_ids3 
 
        #print("\n\nG Keys 1 : ",g_keys1)
        #print("G IDs 1 : ",sk_ids1)
        #print("G Keys 2 : ",g_keys2)
        #print("G IDs 2 : ",sk_ids2)
        #print("G Keys 3 : ",g_keys3)
        #print("G IDs 3 : ",sk_ids3)
        
        #print("\n\nFunction Returns sk_ids1, sk_ids2, sk_ids3 : \n",sk_ids1,sk_ids2,sk_ids3)
        return sk_ids1, sk_ids2, sk_ids3

    # 3 - 1
    def SelectTopKeywords(self,keywords_group, group_attention,n = 5,r = 2):
        """
        Input : keywords_group - three group of keywords got from single 
                attention heads such that G1(H1-4),G2(H5-8),G3(H9-12)
                group_attention - G1 Attentions, G2 Attentions, G3 Attetnions
                n - 5 total keywords 
                r - 2 keywords from non-overlapping subsets 
                n-r - 3 keywords from overlapping subsets 
        Returns:three group of keywords given these n and r 
        """
        #print("\n\n Function Name : SelectTopKeywords")
        g1_set = keywords_group[0]
        g2_set = keywords_group[1]  
        g3_set = keywords_group[2]

        onlyg1 = (g1_set-g2_set)-g3_set
        onlyg2 = (g2_set-g1_set)-g3_set
        onlyg3 = (g3_set-g1_set)-g2_set

        int_g1g2g3 = g1_set.intersection(g2_set,g3_set)
        uni_g1g2g3 = g1_set.union(g2_set,g3_set)

        g1g2 = g1_set.intersection(g2_set)-int_g1g2g3
        g2g3 = g2_set.intersection(g3_set)-int_g1g2g3
        g1g3 = g1_set.intersection(g3_set)-int_g1g2g3
        """
        print("\n\nKeywords - Only G1 : \n\n",onlyg1)
        print("\n\nKeywords - Only G2 : \n\n",onlyg2)
        print("\n\nKeywords - Only G3 : \n\n",onlyg3)
        print("\n\nKeywords - Only in G1 & G2 : \n\n",g1g2)
        print("\n\nKeywords - Only in G2 & G3 : \n\n",g2g3)
        print("\n\nKeywords - Only in G1 & G3 : \n\n",g1g3)
        print("\n\nKeywords - G1 & G2 & G3 : \n\n",int_g1g2g3)
        """
        #print("\n\n Function Returns : self.SelectionProcedure(n,r,onlyg1,onlyg2,onlyg3,g1g2,g2g3,g1g3,int_g1g2g3,group_attention)",)
        return self.SelectionProcedure(n,r,onlyg1,onlyg2,onlyg3,g1g2,g2g3,g1g3,int_g1g2g3,group_attention) # 3 - 1 - 1 

    # 3 - 1 - 1 
    def SelectionProcedure(self,n,r,g1,g2,g3,g1g2,g2g3,g1g3,g1g2g3,g_attention):
        """
        Inputs : n, r - 5, 2 - keyword selection numbers 
                 g1 - only group 1 keywords 
                 g2 - only group 2 keywords 
                 g3 - only group 3 keywords
                 g1g2, g2g3, g1g3 - overlapping keywords in the respective groups
                 but not in g1g2g3
                 g1g2g3 - overlapping keywords in all three groups 
                 g_attention - group attention of each groups 
        Returns : three sets of group of keywords 
        """
        #print("\n\n Function Name : SelectionProcedure")
        no_g1 = self.GroupSelection(g1,g_attention[0],r) # 3- 1 - 1 - 1
        o_g1 = self.GroupSelection([g1g2,g1g3,g1g2g3],g_attention,n-len(no_g1),1,'o')

        no_g2 = self.GroupSelection(g2,g_attention[1],r)
        o_g2 = self.GroupSelection([g2g3,g1g2,g1g2g3],g_attention,n-len(no_g2),2,'o')

        no_g3 = self.GroupSelection(g3,g_attention[2],r)
        o_g3 = self.GroupSelection([g1g3,g2g3,g1g2g3],g_attention,n-len(no_g3),3,'o')

        group1 = []
        group2 = []
        group3 = []

        for key_val in no_g1:
            group1.append(key_val[0])
        for key_val in o_g1:
            group1.append(key_val[0])

        for key_val in no_g2:
            group2.append(key_val[0])
        for key_val in o_g2:
            group2.append(key_val[0])

        for key_val in no_g3:
            group3.append(key_val[0])
        for key_val in o_g3:
            group3.append(key_val[0])
        
        # https://stackoverflow.com/questions/31040525/insert-element-in-python-list-after-every-nth-element
        n = 1
        ele = self.tokenizer.eos_token
        g1 = list(chain(*[group1[i:i+n] + [ele] if len(group1[i:i+n]) == n else group1[i:i+n] for i in range(0, len(group1), n)]))
        g2 = list(chain(*[group2[i:i+n] + [ele] if len(group2[i:i+n]) == n else group2[i:i+n] for i in range(0, len(group2), n)]))
        g3 = list(chain(*[group3[i:i+n] + [ele] if len(group3[i:i+n]) == n else group3[i:i+n] for i in range(0, len(group3), n)]))
       
        if len(g1)<(2*n):
            g1 = self.pad_sequence(g1,2*n-len(g1)) # 3- 1 - 1 - 2
        if len(g2)<(2*n):
            g2 = self.pad_sequence(g2,2*n-len(g2))
        if len(g3)<(2*n):
            g3 = self.pad_sequence(g3,2*n-len(g3))
        #print("\n\n Function Returns : g1,g2,g3\n",g1,g2,g3)
        
        return g1,g2,g3 
    
    # 3 - 1 - 1 - 1
    def GroupSelection(self,groups,attentions,k,group_id = 0, group_type = 'no'):
        """
        Input : groups 
                attentions 
                k - number of keywords to be selected 
                group_id - 0,1,2 
                group_type = "no" # default indicates non-overlapping 
                           = "o" for overlapping group 
        Returns: group of keywords 
        """
        #print("\n\n Function Name : GroupSelection")
        if group_type=='no':
            group_sel = []
            for key in groups:
                group_sel.append([key,attentions[key]])
            group_sel = sorted(group_sel, key = lambda x: x[1],reverse = True)
            final_sel = []
            for curr in group_sel:
                final_sel.append(curr)
                if len(final_sel) == k:
                    #print("Function Output : final_sel",final_sel)
                    return final_sel 
            #print("Function Output : final_sel",final_sel)
            return final_sel 
        
        elif group_type=='o':
            group_int1 = groups[0]
            group_int2 = groups[1]
            group_int3 = groups[2]

            id = group_id - 1 
            id2 = [(id+1)%3,(id+2)%3]
            group_sel = []
            for key in group_int1: 
                group_sel.append([key,(attentions[id][key]+attentions[id2[0]][key])/2])
            for key in group_int2:
                group_sel.append([key,(attentions[id][key]+attentions[id2[1]][key])/2])
            for key in group_int3:
                group_sel.append([key,(attentions[id][key]+attentions[id2[0]][key]+attentions[id2[1]][key])/3])

            group_sel = sorted(group_sel, key = lambda x: x[1],reverse = True)
            final_sel = []
            for curr in group_sel:
                final_sel.append(curr)
                if len(final_sel) == k:
                    #print("Function Output : final_sel",final_sel)
                    return final_sel 
            #print("Function Output : final_sel",final_sel)
            return final_sel 
    
    # 3 - 1 - 1 - 2
    def pad_sequence(self,g,n):
        """
        Input : g - group 
                n - len(group)
        Returns : group with padded sequence to fulfill the total length 
        """
        #print("\n\n Function Name : pad_sequence ")
        for _ in range(n//2):
            g.append(self.tokenizer.pad_token)
            g.append(self.tokenizer.eos_token) # or </s> or </SEP>
        #print("\n\nFunction Returns : g",g)
        return g 

    # 3 - 2 
    def customtokens_to_ids(self,g):
        """
        Inputs : g - group of tokens 
        Returns: ids from dictionary for the corresponding keyword tokens passed
        """
        ids = []
        for kw in g:
            id = self.my_dictionary[kw]
            ids.append(id.item())
        return ids 

    # 3 - 3 
    def subkeytoKeywordConverter(self,input_sentence,g,n,related_keywords=False):
        """
        Input : input_sentence - input context 
                             g - group 
        Returns: given group of subkeys it returns group of keywords 
        """
        keywords = set()
        input_list = input_sentence.split(' ')
        input_list_ws = [word for word in input_list if not word in stop_words]
        
        for k in g:
            k = self.RemovePunctuation(k) # 3 - 3 - 1
            close_k = get_close_matches(k,input_list_ws)
            if len(close_k)>0:
                keywords.add(close_k[0])

        if related_keywords:
            related_keys = self.GetRelatedKeywords(keywords)
            for keys in related_keys:
                keywords.add(keys)
        
        final_keys = []
        for keys in keywords:
            final_keys.append(keys)
            final_keys.append(self.tokenizer.eos_token) # modify to eos_token 

        if len(final_keys)<10:
            final_keys = self.pad_sequence(final_keys,2*n-len(final_keys))

        return final_keys

    # 3 - 3 - 1
    def RemovePunctuation(self,word,space=False):
        if space:
            return re.sub(r'[0-9]', '', re.sub(r'[^\w\s]', ' ', word))
        return re.sub(r'[^\w\s]', '', word)
    
    def GetRelatedKeywords(self,keywords):
        rk = []
        for keys in keywords:
            r_keys = self.most_similar(self.RemovePunctuation(keys))
            rk.append(r_keys[0][0])
        return rk
        
    def most_similar(word, topn=1):
        word = nlp_l.vocab[str(word)]
        queries = [
            w for w in word.vocab 
            if w.is_lower == word.is_lower and w.prob >= -12 and np.count_nonzero(w.vector)
        ]
        print(len(queries))
        by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
        return [(w.lower_,w.similarity(word)) for w in by_similarity[:topn+1] if w.lower_ != word.lower_]

    # 3 - 4 
    def EncodeKeyword(self, keys):
        id_list = []
        for i in range(len(keys)):
            ids = self.tokenizer.encode(keys[i],add_special_tokens=False)
            id_list.extend(ids)
        return id_list 

    # 4 
    def GetValidInputs(self,input_ids, attention_mask, subkey_representations):
        valid_inputs,valid_attentions = [], []
        for i in range(len(subkey_representations)):
            l = len(subkey_representations[i])
            curr_att = [1]*l
            id = torch.cat((torch.Tensor(subkey_representations[i]).to('cuda:0').long(),input_ids[i][l:]))
            atts = torch.cat((torch.Tensor(curr_att).to('cuda:0').long(),attention_mask[i][l:]))
            valid_inputs.append(id)
            valid_attentions.append(atts)
        return torch.stack(valid_inputs), torch.stack(valid_attentions)

    # 5 
    def HeadlineLoss(self,label_list,subkey_representations):
        loss = 0 
        for label, subkey_rep in zip(label_list,subkey_representations):
            loss+=self.HLossHelper(label,subkey_rep) # 5 - 1 
        return loss     
    # 5 - 1 
    def HLossHelper(self,label,subkey_rep):
        loss = 0 
        for i in range(label.shape[0]):
            curr_headline = label[i]
            curr_subkeyrep = self.tokenizer.decode(subkey_rep[i],skip_special_tokens = True)
            if len(curr_subkeyrep)>2:
                headlinerep,keyrep = self.GetHeadlineAndKeywordRepresentation(curr_headline,curr_subkeyrep)
                if headlinerep.shape[0]!=0 and keyrep.shape[0]!=0:
                    loss+=self.CalculateLoss(headlinerep, keyrep)
            else:
                print("lab : ",subkey_rep[i])
        return loss 
    
    # 5 - 1 - 1 
    def GetHeadlineAndKeywordRepresentation(self,headline_id,keywords):
        headline = self.tokenizer.decode(headline_id,skip_special_tokens = True)
        clean_headline = self.RemovePunctuation(headline,True)
        clean_keywords = self.RemovePunctuation(keywords,True)
        modified_headline = [word for word in clean_headline.split() if word not in stop_words and len(word)>2]
        key_list = [word for word in clean_keywords.split() if word not in stop_words and len(word)>2]
        key_rep = []
        headline_rep = []

        for word in modified_headline:
            if word in self.word2vec_model:
                curr_rep = torch.Tensor(self.word2vec_model[word])
                headline_rep.append(curr_rep)

        for key in key_list:
            if key in self.word2vec_model:
                curr_keyrep = torch.Tensor(self.word2vec_model[key])
                key_rep.append(curr_keyrep)


        """
        for word in modified_headline:
            curr_hids = self.tokenizer(word,return_tensors='pt')
            #self.bartmodel = self.bartmodel.to('cuda:0')
            encoded_headline = self.wordencoder(input_ids = curr_hids['input_ids'].cuda(),
                                                attention_mask = curr_hids['attention_mask'].cuda(),use_cache = False)
            print(encoded_headline.last_hidden_state.squeeze(0))
            print(encoded_headline.last_hidden_state.squeeze(0).shape)
            print(torch.stack(encoded_headline.last_hidden_state.squeeze(0)).sum())
            sys.exit()
            headline_rep.append(nn.AvgPool2D(encoded_headline.last_hidden_state.squeeze(0)))
        print(torch.Tensor(headline_rep).shape)
        """
        """
        enc = self.tokenizer(modified_headline, return_tensors="pt")
        self.t5model = self.t5model.to("cuda:0")
        output = self.t5model.encoder(
            input_ids=enc["input_ids"], 
            attention_mask=enc["attention_mask"], 
            return_dict=True
        )
        emb = output.last_hidden_state
        print(emb.shape)
        """
        if len(key_rep)<1 or len(headline_rep)<1:
            print("hl : ",headline)
            print("hr : ",len(headline_rep))
            print("kw : ",keywords)
            print("kr : ",len(key_rep))
            return torch.Tensor(0), torch.Tensor(0)
        return torch.stack(headline_rep), torch.stack(key_rep)
    
    # 5 - 1 - 2
    def CalculateLoss(self,headlinerep, subkeyrep):
        return directed_hausdorff(np.array(headlinerep).reshape(-1,1),np.array(subkeyrep).reshape(-1,1))[0]
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # modifications in this forward 
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_attention_mask1 = None, 
        decoder_attention_mask2 = None, 
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        labels1 = None, 
        labels2 = None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        beam_flag = False, 
        generate_flag = False, 
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
                
        if beam_flag:
            #print("encoder ouput in modelling t5 :\n",encoder_outputs)
            # code is exactly same from encoder-decoder model given in transformer library
            if encoder_outputs is None: #changed to not none for verification
            # modify - 8 
                attention_mask = attention_mask[0].reshape(1,128)
                encoder_outputs = self.encoder(
                    input_ids = input_ids, #strip last 6 
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )

            hidden_states = encoder_outputs[0]

            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)

            if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids = self._shift_right(labels)

            # If decoding with past key value states, only the last tokens
            # should be given as an input
            if past_key_values is not None:
                assert labels is None, "Decoder should not use cached key value states when training."
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids[:, -1:]
                if decoder_inputs_embeds is not None:
                    decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states = hidden_states.to(self.decoder.first_device)
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.decoder.first_device)
                if decoder_attention_mask is not None:
                    decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

            # Decode
            #print("decoder_input_ids at decoder : ",decoder_input_ids)
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim ** -0.5)

            lm_logits = self.lm_head(sequence_output)

            return Seq2SeqLMOutput(
                loss=None, # modify loss to total loss 
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            ) 
        # generate flag is set true to get subkey ids in testing phase 
        if generate_flag: # same encoder during training & generation 
            input_ids = decoder_input_ids 
            keyencoder_outputs = self.keyencoder(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              inputs_embeds=inputs_embeds,
                                              head_mask=head_mask,
                                              output_attentions=True, # doubt: should we give output_attentions = True? (PREVIOUS : output_attentions=output_attentions)
                                              output_hidden_states=output_hidden_states,
                                              return_dict=return_dict,
                                            )
            multihead_attentions = keyencoder_outputs.attentions[0]
            subkey_representations = self.get_subkey_representations(multihead_attentions,input_ids)
            return subkey_representations  
            
        # training phase - when beam flag and gen flag is false - it reaches here 
        keyencoder_outputs = self.keyencoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=True, # doubt: should we give output_attentions = True? (PREVIOUS : output_attentions=output_attentions)
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        multihead_attentions = keyencoder_outputs.attentions[0]
        subkey_representations = self.get_subkey_representations(multihead_attentions,input_ids)

        label_list = [labels, labels1, labels2]
        dec_att_list = [decoder_attention_mask,decoder_attention_mask1,decoder_attention_mask2]

        total_loss = torch.Tensor(0).to("cuda:0") # initialize total loss
        loss_list = [None]*3 # initialize loss list 

        for i,(lbs,dec_att_mask) in enumerate(zip(label_list,dec_att_list)):
        # Encode if needed (training, first prediction pass)
            if encoder_outputs is None:
                #print("Input IDs : \n",input_ids)
                inp_ids, att_mask = self.GetValidInputs(input_ids, attention_mask, subkey_representations[i])
                #print("Inp IDs : \n",inp_ids)
                assert inp_ids.shape==input_ids.shape 
                # Convert encoder inputs in embeddings if needed
                encoder_outputs = self.encoder(
                    input_ids=inp_ids,
                    attention_mask=att_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                )

            hidden_states = encoder_outputs[0]

            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)

            if lbs is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids = self._shift_right(lbs)

            # If decoding with past key value states, only the last tokens
            # should be given as an input
            if past_key_values is not None:
                assert lbs is None, "Decoder should not use cached key value states when training."
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids[:, -1:]
                if decoder_inputs_embeds is not None:
                    decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.decoder.first_device)
                hidden_states = hidden_states.to(self.decoder.first_device)
                if decoder_input_ids is not None:
                    decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.decoder.first_device)
                if dec_att_mask is not None:
                    dec_att_mask = dec_att_mask.to(self.decoder.first_device)

            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=dec_att_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim ** -0.5)

            lm_logits = self.lm_head(sequence_output)

            loss = None

            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # pass 3 labels 
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss_list[i] = loss # added loss to loss lit 

            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        sum_loss = (loss_list[0]+loss_list[1]+loss_list[2])
        headlines_loss = self.HeadlineLoss(label_list,subkey_representations)
        total_loss = (0.8 * sum_loss + 0.2 * headlines_loss)/3 

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5EncoderModel(T5PreTrainedModel):
    authorized_missing_keys = [
        r"encoder\.embed_tokens\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Example::
            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs