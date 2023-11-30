# torchvision: 0.16.2, transformers: 4.37.2, Python: 3.8.18
from collections.abc import Mapping
import math
from typing import Union
import torch
from torch import nn
import logging
import numpy as np
from transformers import OPTConfig, OPTModel
from collections.abc import Mapping
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
import os, sys
from .. import ModuleShard, ModuleShardConfig
from . import TransformerShardData

class OptLayerShard(ModuleShard):
    """Module shard based on `OPTDecoderLayer`."""

    def __init__(self, config: OPTConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.k_proj = None
        self.v_proj = None
        self.q_proj = None
        self.out_proj = None
        self.self_attn_layer_norm = None
        self.activation_fn = None
        self.fc1 = None
        self.fc2 = None
        self.final_layer_norm = None
        self.residue = None
        self.data_shape = None
        self.hidden_states = None
        self.attn_weights = None
        self.present_key_value = None
        self.head_dim = None
        self._build_shard()

    def k_v_q_proj(self, data, casual_attention_mask):
        bsz, tgt_len, _ = data.size()
        scaling = self.head_dim**-0.5
        query_states = self.q_proj(data) * scaling
        key_states = self._shape(self.k_proj(data), -1, bsz)
        value_states = self._shape(self.v_proj(data), -1, bsz)

        self.present_key_value = (key_states, value_states)

        proj_shape = (bsz * self.config.num_attention_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = attn_weights.view(bsz, self.config.num_attention_heads, tgt_len, src_len) + casual_attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        )
        attn_weights = attn_weights.view(bsz * self.config.num_attention_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.config.num_attention_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.config.hidden_size)
        return attn_output

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2).contiguous()

    def _build_shard(self):
        if self.has_layer(0):
            self.k_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=self.config.enable_bias)
            self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=self.config.enable_bias)
            self.q_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=self.config.enable_bias)
            self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        if self.has_layer(1):
            self.out_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=self.config.enable_bias)
            self.self_attn_layer_norm = nn.LayerNorm(self.config.hidden_size, elementwise_affine=self.config.layer_norm_elementwise_affine)
        if self.has_layer(2):
            self.activation_fn = ACT2FN[self.config.activation_function]
            self.fc1 = nn.Linear(self.config.hidden_size, self.config.ffn_dim, bias=self.config.enable_bias)
        if self.has_layer(3):
            self.fc2 = nn.Linear(self.config.ffn_dim, self.config.hidden_size, bias=self.config.enable_bias)
            self.final_layer_norm = nn.LayerNorm(self.config.hidden_size, elementwise_affine=self.config.layer_norm_elementwise_affine)

    @torch.no_grad()
    def forward(self, datapack: TransformerShardData) -> TransformerShardData:
        """Compute layer shard."""
        casual_attention_mask, data = datapack[0], datapack[1]
        if self.has_layer(0):
            self.residual = data
            data = self.k_v_q_proj(data, casual_attention_mask)
        if self.has_layer(1):
            data = self.out_proj(data)
            data = self.residual + data
            data = self.self_attn_layer_norm(data)
        if self.has_layer(2):
            self.data_shape = data.shape
            data = data.reshape(-1, data.size(-1))
            self.residual = data
            data = self.fc1(data)
            data = self.activation_fn(data)
            # data = self.fc2(data)
            # data = (self.residual + data).view(data_shape)
        if self.has_layer(3):
            data = self.fc2(data)
            data = (self.residual + data).view(self.data_shape)
            data = self.final_layer_norm(data)
            data = (data,)
            data += (self.present_key_value,)
        return [casual_attention_mask, data]

logger = logging.getLogger(__name__)

class OPTModelShard(ModuleShard):
    """Module shard based on `OPTModel_135m`."""

    def __init__(self, config: OPTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)

        self.embed_tokens = None
        self.embed_positions = None
        self.project_out = None
        self.project_in = None
        self.layers = nn.ModuleList()
        self.casual_attention_mask = None
        self.next_decoder_cache = ()

        logger.debug(">>>> Model name: %s", self.config.model_type)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)
        
    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.embed_tokens.weight.copy_(torch.from_numpy((weights["decoder.embed_tokens.weight"])))
        self.embed_positions.weight.copy_(torch.from_numpy((weights["decoder.embed_positions.weight"])))
        self.project_in.weight.copy_(torch.from_numpy((weights["decoder.project_in.weight"])))
    
    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.project_out.weight.copy_(torch.from_numpy((weights["decoder.project_out.weight"])))
        

    def _build_shard(self, weights):
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.word_embed_proj_dim, self.config.pad_token_id)
            self.embed_positions = OPTLearnedPositionalEmbedding(self.config.max_position_embeddings, self.config.hidden_size)
            self.project_in = nn.Linear(self.config.word_embed_proj_dim, self.config.hidden_size, bias=False)
            self.embed_tokens.eval()
            self.embed_positions.eval()
            self.project_in.eval()
            self._load_weights_first(weights)

        layer_curr = self.shard_config.layer_start
        while layer_curr <= self.shard_config.layer_end:
            layer_id = math.ceil(layer_curr / 4) - 1
            sublayer_start = (layer_curr - 1) % 4
            if layer_id == math.ceil(self.shard_config.layer_end / 4) - 1:
                sublayer_end = (self.shard_config.layer_end - 1) % 4
            else:
                sublayer_end = 3
            logger.debug(">>>> Load layer %d, sublayers %d-%d",
                         layer_id, sublayer_start, sublayer_end)
            layer_config = ModuleShardConfig(layer_start=sublayer_start, layer_end=sublayer_end)
            layer = OptLayerShard(self.config, layer_config)
            self._load_weights_layer(weights, layer_id, layer)
            self.layers.append(layer)
            layer_curr += sublayer_end - sublayer_start + 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load Linear model for the last shard")
            self.project_out = nn.Linear(self.config.hidden_size, self.config.word_embed_proj_dim, bias=False)
            self.project_out.eval()
            self._load_weights_last(weights)
    
    @torch.no_grad()
    # def forward(self, input_ids, attention_mask):
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        if self.shard_config.is_first:
            input_shape = data.size()
            input_ids = data.view(-1, input_shape[-1])
            data = self.embed_tokens(data)
            batch_size, seq_length = input_shape

            past_key_values_length = 0
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=data.device)

            self.casual_attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, data, past_key_values_length)
            
            pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
            data = self.project_in(data)
            data = data + pos_embeds
        if isinstance(data, tuple):
            data, self.casual_attention_mask = data[1], data[0]
        for layer in self.layers:
            layer_outputs = layer([self.casual_attention_mask, data])[1]
            data = layer_outputs[0]
            self.next_decoder_cache += (layer_outputs[1],)
        if self.shard_config.is_last:
            data = self.project_out(data)
            next_cache = self.next_decoder_cache
            data = BaseModelOutputWithPast(last_hidden_state=data, past_key_values=next_cache, hidden_states=None, attentions=None)
            data = data['last_hidden_state']
            return data
        return (self.casual_attention_mask, data)

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer_id, layer):
        root = f"decoder.layers.{layer_id}."
        if layer.has_layer(0):
            layer.k_proj.weight.copy_(torch.from_numpy(weights[root + "self_attn.k_proj.weight"]))
            layer.v_proj.weight.copy_(torch.from_numpy(weights[root + "self_attn.v_proj.weight"]))
            layer.q_proj.weight.copy_(torch.from_numpy(weights[root + "self_attn.q_proj.weight"]))
            layer.out_proj.weight.copy_(torch.from_numpy(weights[root + "self_attn.out_proj.weight"]))
            layer.k_proj.bias.copy_(torch.from_numpy(weights[root + "self_attn.k_proj.bias"]))
            layer.v_proj.bias.copy_(torch.from_numpy(weights[root + "self_attn.v_proj.bias"]))
            layer.q_proj.bias.copy_(torch.from_numpy(weights[root + "self_attn.q_proj.bias"]))
            layer.out_proj.bias.copy_(torch.from_numpy(weights[root + "self_attn.out_proj.bias"]))
        if layer.has_layer(1):
            layer.self_attn_layer_norm.weight.copy_(torch.from_numpy(weights[root + "self_attn_layer_norm.weight"]))
            layer.self_attn_layer_norm.bias.copy_(torch.from_numpy(weights[root + "self_attn_layer_norm.bias"]))
        if layer.has_layer(2):
            layer.fc1.weight.copy_(torch.from_numpy(weights[root + "fc1.weight"]))
            layer.fc2.weight.copy_(torch.from_numpy(weights[root + "fc2.weight"]))
            layer.fc1.bias.copy_(torch.from_numpy(weights[root + "fc1.bias"]))
            layer.fc2.bias.copy_(torch.from_numpy(weights[root + "fc2.bias"]))
        if layer.has_layer(3):
            layer.final_layer_norm.weight.copy_(torch.from_numpy(weights[root + "final_layer_norm.weight"]))
            layer.final_layer_norm.bias.copy_(torch.from_numpy(weights[root + "final_layer_norm.bias"]))

    @staticmethod
    def save_weights(model_name: str, model_file: str) -> None:
        """Save the model weights file."""
        model = OPTModel.from_pretrained(model_name)
        state_dict = model.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        np.savez(model_file, **weights)