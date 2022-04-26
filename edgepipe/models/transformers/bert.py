"""BERT transformers."""
import math
import time
import numpy as np
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertSelfAttention, BertSelfOutput, BertIntermediate, BertOutput, BertLayer
from . import TransformerShard


def _forward_kernel(layer, x, skip, kernel_id):
    if kernel_id == 1:
        x = layer[0](x)
    elif kernel_id == 2:
        x = x[0]
        x = layer[0](x, skip)
        skip = x
    elif kernel_id == 3:
        x = layer[0](x)
    else:
        x = layer[0](x, skip)
        skip = x
    return x, skip


class BertTransformerShard(TransformerShard):
    """BERT transformer shard."""

    def __init__(self, stage, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight=True):
        super().__init__(stage, model_name, model_file, is_first, is_last, start_layer, end_layer, load_weight)
        self.embeddings = None

        print(f">>>> Model name {model_name}")
        if self.load_weight:
            print(f">>>> Load weight file {self.weights_file_name}")
            with np.load(self.weights_file_name) as weights:
                self._make_layer(weights)
        else:
            self._make_layer(None)

        print(f"======= Finish Build BertTransformerShard{self.stage} ==========")

    def _make_layer(self, weights):
        ## first Shard
        if self.is_first:
            self.embeddings = BertEmbeddings(self.config)
            self.embeddings.eval()
            print(">>>> Load embeddings layer for the first shard ")
            if self.load_weight:
                self._load_layer_weights(weights, 0, None, load_first = True, load_last=False, load_kernel = False, kernel_id=None)
                print(">>>> Load weights for embeddings layer ")

        current_layer_idx = self.start_layer

        ## first ununit part
        if self.start_layer %4 != 1 or (self.start_layer+3 > self.end_layer):
            print(f">>>> For the first model part, load weight is {self.load_weight}:")
            for i in range(self.start_layer, min(self.end_layer, math.ceil(self.start_layer/4)*4)+1):
                print(f"    Load the {i%4}-th operation ({self.operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
                layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1, self.load_weight)
                layer.eval()
                self.first_ops.append(layer)
            current_layer_idx = min(self.end_layer+1, math.ceil(self.start_layer/4)*4+1)

        ## mid unit part, the whole vit_layer
        while current_layer_idx + 3 <= self.end_layer:
            with torch.no_grad():
                layer = BertLayer(self.config)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(current_layer_idx/4)-1, layer)
            layer.eval()
            self.vit_layers.append(layer)
            print(f">>>> Load the {math.ceil(current_layer_idx/4)-1}-th {self.model_name} Layer, load weight is {self.load_weight}")
            current_layer_idx += 4

        ## last unit part
        if self.end_layer >= current_layer_idx:
            print(f">>>> For the last model part, load weight is {self.load_weight}:")
        for i in range(current_layer_idx, self.end_layer+1):
            print(f"    Load the {i%4}-th operation ({self.operators_list[(i-1)%4]}) for {math.ceil(i/4)-1}-th vit layer")
            layer = self._build_kernel(weights, i%4, math.ceil(i/4)-1, self.load_weight)
            if self.load_weight:
                layer = self._load_layer_weights(weights, math.ceil(i/4)-1, layer, False, False, True, i%4)
            layer.eval()
            self.last_ops.append(layer)

        ## last Shard
        if self.is_last:
            self.bertpooler = BertPooler(self.config)
            self.bertpooler.eval()
            if self.load_weight:
                self._load_layer_weights(weights, 0, None, load_first = False, load_last=True, load_kernel = False, kernel_id=None)
                print(">>>> Load weights for layernorm and last shard")


        if self.load_weight:
            print(">>>> Finish load weights")
        else:
            print(">>>> Do NOT load weights")

    def _build_kernel(self, weights, kernel_id, vit_layer_id, load_weight=True):
        layers = nn.ModuleList()
        if kernel_id == 1:
            layers.append(BertSelfAttention(self.config))
        elif kernel_id == 2:
            layers.append(BertSelfOutput(self.config))
        elif kernel_id == 3:
            layers.append(BertIntermediate(self.config))
        else:
            layers.append(BertOutput(self.config))
        if load_weight:
            self._load_layer_weights(weights, vit_layer_id, layers, False, False, load_weight, kernel_id)
        return layers

    def _load_layer_weights(self, weights, id, transformer_layer, load_first = False, load_last=False, load_kernel = False, kernel_id=None):
        ROOT = f"encoder.layer.{id}."
        ATTENTION_Q = "attention.self.query."
        ATTENTION_K = "attention.self.key."
        ATTENTION_V = "attention.self.value."
        ATTENTION_OUT_DENSE = "attention.output.dense."
        ATTENTION_OUT_LAYERNORM = "attention.output.LayerNorm."
        INTERMEDIATE = "intermediate.dense."
        OUTPUT_DENSE = "output.dense."
        OUTPUT_LAYER = "output.LayerNorm."
        WEIGHT = "weight"
        BIAS = "bias"
        if load_first:
            with torch.no_grad():
                self.embeddings.position_ids.copy_(torch.from_numpy((weights["embeddings.position_ids"])))
                self.embeddings.word_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.word_embeddings.weight']))
                self.embeddings.position_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.position_embeddings.weight']))
                self.embeddings.token_type_embeddings.weight.copy_(torch.from_numpy(weights['embeddings.token_type_embeddings.weight']))
                self.embeddings.LayerNorm.weight.copy_(torch.from_numpy(weights['embeddings.LayerNorm.weight']))
                self.embeddings.LayerNorm.bias.copy_(torch.from_numpy(weights['embeddings.LayerNorm.bias']))

        if load_last:
            with torch.no_grad():
                self.bertpooler.dense.weight.copy_(torch.from_numpy(weights["pooler.dense.weight"]))
                self.bertpooler.dense.bias.copy_(torch.from_numpy(weights['pooler.dense.bias']))


        if not load_first and not load_last:
            with torch.no_grad():
                if not load_kernel:
                    query_weight = torch.from_numpy(weights[ROOT + ATTENTION_Q + WEIGHT])
                    key_weight = torch.from_numpy(weights[ROOT + ATTENTION_K + WEIGHT])
                    value_weight = torch.from_numpy(weights[ROOT + ATTENTION_V + WEIGHT])
                    out_dense_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + WEIGHT])
                    output_layernorm_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + WEIGHT])
                    intermediate_dense_weight = torch.from_numpy(weights[ROOT+INTERMEDIATE+WEIGHT])
                    dense_weight = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + WEIGHT])
                    layernorm_weight = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + WEIGHT])

                    query_bias = torch.from_numpy(weights[ROOT + ATTENTION_Q + BIAS])
                    key_bias = torch.from_numpy(weights[ROOT + ATTENTION_K + BIAS])
                    value_bias = torch.from_numpy(weights[ROOT + ATTENTION_V + BIAS])
                    out_dense_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + BIAS])
                    output_layernorm_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + BIAS])
                    intermediate_dense_bias = torch.from_numpy(weights[ROOT+INTERMEDIATE+BIAS])
                    dense_bias = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + BIAS])
                    layernorm_bias = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + BIAS])

                    transformer_layer.attention.self.query.weight.copy_(query_weight)
                    transformer_layer.attention.self.key.weight.copy_(key_weight)
                    transformer_layer.attention.self.value.weight.copy_(value_weight)
                    transformer_layer.attention.output.dense.weight.copy_(out_dense_weight)
                    transformer_layer.attention.output.LayerNorm.weight.copy_(output_layernorm_weight)

                    transformer_layer.attention.self.query.bias.copy_(query_bias)
                    transformer_layer.attention.self.key.bias.copy_(key_bias)
                    transformer_layer.attention.self.value.bias.copy_(value_bias)
                    transformer_layer.attention.output.dense.bias.copy_(out_dense_bias)
                    transformer_layer.attention.output.LayerNorm.bias.copy_(output_layernorm_bias)

                    transformer_layer.intermediate.dense.weight.copy_(intermediate_dense_weight)
                    transformer_layer.intermediate.dense.bias.copy_(intermediate_dense_bias )

                    transformer_layer.output.dense.weight.copy_(dense_weight)
                    transformer_layer.output.dense.bias.copy_(dense_bias)
                    transformer_layer.output.LayerNorm.weight.copy_(layernorm_weight)
                    transformer_layer.output.LayerNorm.bias.copy_(layernorm_bias)
                    print(f"memory {self.process.memory_info().rss // 1000000} MB")

                elif kernel_id == 1:

                    query_weight = torch.from_numpy(weights[ROOT + ATTENTION_Q + WEIGHT])
                    key_weight = torch.from_numpy(weights[ROOT + ATTENTION_K + WEIGHT])
                    value_weight = torch.from_numpy(weights[ROOT + ATTENTION_V + WEIGHT])
                    query_bias = torch.from_numpy(weights[ROOT + ATTENTION_Q + BIAS])
                    key_bias = torch.from_numpy(weights[ROOT + ATTENTION_K + BIAS])
                    value_bias = torch.from_numpy(weights[ROOT + ATTENTION_V + BIAS])
                    transformer_layer[0].query.weight.copy_(query_weight)
                    transformer_layer[0].key.weight.copy_(key_weight)
                    transformer_layer[0].value.weight.copy_(value_weight)
                    transformer_layer[0].query.bias.copy_(query_bias)
                    transformer_layer[0].key.bias.copy_(key_bias)
                    transformer_layer[0].value.bias.copy_(value_bias)

                elif kernel_id == 2:
                    out_dense_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + WEIGHT])
                    output_layernorm_weight = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + WEIGHT])
                    out_dense_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_DENSE + BIAS])
                    output_layernorm_bias = torch.from_numpy(weights[ROOT + ATTENTION_OUT_LAYERNORM + BIAS])
                    transformer_layer[0].dense.weight.copy_(out_dense_weight)
                    transformer_layer[0].LayerNorm.weight.copy_(output_layernorm_weight)
                    transformer_layer[0].dense.bias.copy_(out_dense_bias)
                    transformer_layer[0].LayerNorm.bias.copy_(output_layernorm_bias)
                elif kernel_id == 3:
                    intermediate_dense_weight = torch.from_numpy(weights[ROOT+INTERMEDIATE+WEIGHT])
                    intermediate_dense_bias = torch.from_numpy(weights[ROOT+INTERMEDIATE+BIAS])
                    transformer_layer[0].dense.weight.copy_(intermediate_dense_weight)
                    transformer_layer[0].dense.bias.copy_(intermediate_dense_bias )
                elif kernel_id == 0:
                    dense_weight = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + WEIGHT])
                    layernorm_weight = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + WEIGHT])
                    dense_bias = torch.from_numpy(weights[ROOT + OUTPUT_DENSE + BIAS])
                    layernorm_bias = torch.from_numpy(weights[ROOT + OUTPUT_LAYER + BIAS])

                    transformer_layer[0].dense.weight.copy_(dense_weight)
                    transformer_layer[0].dense.bias.copy_(dense_bias)
                    transformer_layer[0].LayerNorm.weight.copy_(layernorm_weight)
                    transformer_layer[0].LayerNorm.bias.copy_(layernorm_bias)

        return transformer_layer

    @torch.no_grad()
    def forward(self, x):
        with self._lock:
            start = time.time()
            if self.is_first:
                x = self.embeddings(x)
                skip = x
            else:
                x, skip = x[0], x[1]

            for i, op in enumerate(self.first_ops):
                x, skip = _forward_kernel(op, x, skip, (self.start_layer+i)%4)

            for layer in self.vit_layers:
                with torch.no_grad():
                    x = layer(x)[0]
                    skip = x

            for i, op in enumerate(self.last_ops):
                # could drop modulus since 0<=i<4, but making 0<=kernel_id<4 is at least consistent with _load_layer_weights()
                x, skip = _forward_kernel(op, x, skip, (i+1)%4)

            if self.is_last:
                x = self.bertpooler(x)
            end = time.time()
            if self.total_batch == 0:
                self.batch_0_finish = time.time()
            else:
                finish_batch_time = time.time()
                self.total_data += x.shape[0] ##14 #split_size
                tmp_throughput = self.total_data/(finish_batch_time-self.batch_0_finish)
                print(f"total data is {self.total_data}, time is {finish_batch_time-self.batch_0_finish}, temporarily throughput is  {tmp_throughput} ")

        self.total_time +=  (end - start)
        self.total_batch += 1
        print(f"Round {self.total_batch}: memory {self.process.memory_info().rss // 1000000} MB")
        print(f"Shard{self.stage} finishes {self.total_batch} microbatch, time is {end -start}, total time is {self.total_time}")
        if self.is_last:
            return x
        return x, skip