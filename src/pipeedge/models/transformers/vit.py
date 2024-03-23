"""ViT Transformers."""
from collections.abc import Mapping
import logging
import math
import os
from typing import Optional, Union
import numpy as np
import requests
import torch
from torch import nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings, ViTIntermediate, ViTOutput, ViTSelfAttention, ViTSelfOutput
)
from .. import ModuleShard, ModuleShardConfig
from . import TransformerShardData
import torch.nn.functional as F
import types
import copy


logger = logging.getLogger(__name__)

_WEIGHTS_URLS = {
    'google/vit-base-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz',
    'google/vit-large-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz',
    'google/vit-huge-patch14-224-in21k': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
}


class ViTLayerShard(ModuleShard):
    """Module shard based on `ViTLayer`."""

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.layernorm_before = None
        self.self_attention = None
        self.self_output = None
        self.layernorm_after = None
        self.intermediate = None
        self.output = None
        self._build_shard()

    def _build_shard(self):
        if self.has_layer(0):
            self.layernorm_before = nn.LayerNorm(self.config.hidden_size,
                                                 eps=self.config.layer_norm_eps)
            self.self_attention = ViTSelfAttention(self.config)
        if self.has_layer(1):
            self.self_output = ViTSelfOutput(self.config)
        if self.has_layer(2):
            self.layernorm_after = nn.LayerNorm(self.config.hidden_size,
                                                eps=self.config.layer_norm_eps)
            self.intermediate = ViTIntermediate(self.config)
        if self.has_layer(3):
            self.output = ViTOutput(self.config)

    
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute layer shard."""
        if self.has_layer(0):
            data_norm = self.layernorm_before(data)
            data = (self.self_attention(data_norm)[0], data)
        if self.has_layer(1):
            skip = data[1]
            data = self.self_output(data[0], skip)
            data += skip
        if self.has_layer(2):
            data_norm = self.layernorm_after(data)
            data = (self.intermediate(data_norm), data)
        if self.has_layer(3):
            data = self.output(data[0], data[1])
        return data


class ViTModelShard(ModuleShard):
    """Module shard based on `ViTModel` (no pooling layer)."""

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping], prune=False):
        super().__init__(config, shard_config)
        self.embeddings = None
        # ViTModel uses an encoder here, but we'll just add the layers here instead.
        # Since we just do inference, a ViTEncoderShard class wouldn't provide real benefit.
        self.layers = nn.ModuleList()
        self.layernorm = None
        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights, prune)
        else:
            self._build_shard(model_weights, prune)

    def _build_shard(self, weights, prune=False):
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.embeddings = ViTEmbeddings(self.config)
            self._load_weights_first(weights, prune)

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
            layer = ViTLayerShard(self.config, layer_config)
            self._load_weights_layer(weights, layer_id, layer, prune)
            self.layers.append(layer)
            layer_curr += sublayer_end - sublayer_start + 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self._load_weights_last(weights, prune)

    @torch.no_grad()
    def _load_weights_first(self, weights, prune=False):
            if(prune):
                self.embeddings.cls_token.copy_(torch.from_numpy(weights["vit.embeddings.cls_token"]))
                self.embeddings.position_embeddings.copy_(torch.from_numpy(weights["vit.embeddings.position_embeddings"]))
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(weights["vit.embeddings.patch_embeddings.projection.weight"]))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["vit.embeddings.patch_embeddings.projection.bias"]))
            else:
                self.embeddings.cls_token.copy_(torch.from_numpy(weights["cls"]))
                self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
                conv_weight = weights["embedding/kernel"]
                # O, I, J, K = conv_weight.shape
                # conv_weight = conv_weight.reshape(K,J,O,I)
                conv_weight = conv_weight.transpose([3, 2, 0, 1])
                self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
                self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))
    @torch.no_grad()
    def _load_weights_last(self, weights, prune=False):
        if(prune):
            self.layernorm.weight.copy_(torch.from_numpy(weights["vit.layernorm.weight"]))
            self.layernorm.bias.copy_(torch.from_numpy(weights["vit.layernorm.bias"]))
        else:
            self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
            self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer_id, layer, prune=False):
        root = f"Transformer/encoderblock_{layer_id}/"
        hidden_size = self.config.hidden_size
        if layer.has_layer(0):
            if(prune):
                layer.layernorm_before.weight.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_before.weight".format(layer_id)]))
                layer.layernorm_before.bias.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_before.bias".format(layer_id)]))
                layer.self_attention.query.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.query.weight".format(layer_id)]))
                layer.self_attention.key.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.key.weight".format(layer_id)]))
                layer.self_attention.value.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.value.weight".format(layer_id)]))
                layer.self_attention.query.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.query.bias".format(layer_id)]))
                layer.self_attention.key.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.key.bias".format(layer_id)]))
                layer.self_attention.value.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_attention.value.bias".format(layer_id)]))
            else:
                layer.layernorm_before.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_0/scale"]))
                layer.layernorm_before.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_0/bias"]))
                layer.self_attention.query.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.key.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.value.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.query.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/bias"]).view(-1))
                layer.self_attention.key.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/bias"]).view(-1))
                layer.self_attention.value.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/bias"]).view(-1))
        if layer.has_layer(1):
            if(prune):
                layer.self_output.dense.weight.copy_(torch.from_numpy(weights["vit.layers.{}.self_output.dense.weight".format(layer_id)]))
                layer.self_output.dense.bias.copy_(torch.from_numpy(weights["vit.layers.{}.self_output.dense.bias".format(layer_id)]))
            else:
                layer.self_output.dense.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_output.dense.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/bias"]).view(-1))
        if layer.has_layer(2):
            if(prune):
                layer.layernorm_after.weight.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_after.weight".format(layer_id)]))
                layer.layernorm_after.bias.copy_(torch.from_numpy(weights["vit.layers.{}.layernorm_after.bias".format(layer_id)]))
                layer.intermediate.dense.weight.copy_(torch.from_numpy(weights["vit.layers.{}.intermediate.dense.weight".format(layer_id)]))
                layer.intermediate.dense.bias.copy_(torch.from_numpy(weights["vit.layers.{}.intermediate.dense.bias".format(layer_id)]))
            else:
                layer.layernorm_after.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_2/scale"]))
                layer.layernorm_after.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_2/bias"]))
                layer.intermediate.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/kernel"]).t())
                layer.intermediate.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/bias"]).t())
        if layer.has_layer(3):
            if(prune):
                layer.output.dense.weight.copy_(torch.from_numpy(weights["vit.layers.{}.output.dense.weight".format(layer_id)]))
                layer.output.dense.bias.copy_(torch.from_numpy(weights["vit.layers.{}.output.dense.bias".format(layer_id)]))                
            else:
                layer.output.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/kernel"]).t())
                layer.output.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/bias"]).t())

    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        if self.shard_config.is_first:
            data = self.embeddings(data)
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.layernorm(data)
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        if url is None:
            url = _WEIGHTS_URLS[model_name]
        logger.info('Downloading model: %s: %s', model_name, url)
        req = requests.get(url, stream=True, timeout=timeout_sec)
        req.raise_for_status()
        with open(model_file, 'wb') as file:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())



class ViTShardForImageClassification(ModuleShard):
    """Module shard based on `ViTForImageClassification`."""
    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping], prune=False):
        super().__init__(config, shard_config)
        self.vit = None
        self.classifier = None
            
        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights, prune)
        else:
            self._build_shard(model_weights, prune)

    def _build_shard(self, weights, prune=False):
        ## all shards use the inner ViT model
        self.vit = ViTModelShard(self.config, self.shard_config, weights, prune)

        if self.shard_config.is_last:
            logger.debug(">>>> Load classifier for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            self._load_weights_last(weights, prune)

    @torch.no_grad()
    def _load_weights_last(self, weights, prune=False):
        if(prune):
            self.classifier.weight.copy_(torch.from_numpy(weights['classifier.weight']))
            self.classifier.bias.copy_(torch.from_numpy(weights['classifier.bias']))
        else:
            self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
            self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))

    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        data = self.vit(data)
        if self.shard_config.is_last:
            data = self.classifier(data[:, 0, :])
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        ViTModelShard.save_weights(model_name, model_file, url=url, timeout_sec=timeout_sec)

    #SNIP Algorithm on Vit model
    def prune_snip(self, ubatch, ubatch_labels, keep_ratio = 0.9):
        net = copy.deepcopy(self)

        # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
        # instead of the weights
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                nn.init.xavier_normal_(layer.weight) 
                layer.weight.requires_grad = False
            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(snip_forward_conv2d, layer)
        
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(snip_forward_linear, layer)

        # forward and backward
        # Compute gradients (but don't apply them)
        net.zero_grad()
        outputs = net.forward(ubatch)
        loss = F.nll_loss(outputs, ubatch_labels)
        loss.backward()
        # snip_keep_masks according to layers

        grads_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))

        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
    
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
    
        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())
    
        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))


        prunable_layers = filter(
            lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.Linear), self.modules())
    
        for layer, keep_mask in zip(prunable_layers, keep_masks):
            assert (layer.weight.shape == keep_mask.shape)

            # mask[i] == 0 --> Prune parameter
            # mask[i] == 1 --> Keep parameter
    
            # Step 1: Set the masked weights to zero (NB the biases are ignored)
            # Step 2: Make sure their gradients remain zero
            layer.weight.data[keep_mask == 0.] = 0.
        state_dict = self.state_dict()
        weights = {}
        for key, val in state_dict.items():
            weights[key] = val
        return weights

def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)


def snip_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
