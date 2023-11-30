from src.pipeedge.models.transformers.opt import OPTModelShard
import torch
from transformers import AutoTokenizer, OPTModel
from transformers import AutoConfig
import os, sys
from src.pipeedge.models import ModuleShard, ModuleShardConfig
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = OPTModel.from_pretrained("facebook/opt-350m")


model_file = 'OPT_350m'

if(not os.path.isfile('./' + model_file + '.npz')): 
    state_dict = model.state_dict()
    weights = {}
    for key, val in state_dict.items():
        weights[key] = val
    np.savez(model_file, **weights)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")['input_ids']
outputs = model(inputs)

# inputs = np.array([[    2,  2522,   964,   351,    75,   907,    42,  1966,     6,   905,
#           1937,     5,   220,    65,    52, 15393,     4],
#         [    2,  3762,    55, 38283,   937,  1938,     8,    38,   437,  1311,
#             62,     4,     1,     1,     1,     1,     1],
#         [    2,  3762,    55, 38283,   937,  1938,    50,    38,   437,  1311,
#             62,     4,     1,     1,     1,     1,     1],
#         [    2,   133,    55,    52,   892, 47041,     6,     5, 26002,   906,
#             51,   120,     4,     1,     1,     1,     1],
#         [    2, 10781,    30,   183,     5,  4905,    32,   562, 22802,   330,
#            906,     4,     1,     1,     1,     1,     1],
#         [    2,   100,   581,  4190,    47,    10,  4076,     4,     1,     1,
#              1,     1,     1,     1,     1,     1,     1],
#         [    2, 33153, 36408,     5,  3451,  3269,     4,     1,     1,     1,
#              1,     1,     1,     1,     1,     1,     1],
#         [    2, 19993, 15763,   571,  4183,    39,   169,    66,     9,     5,
#           2391,     4,     1,     1,     1,     1,     1]])

# inputs = torch.from_numpy(inputs)
# outputs = model(inputs)

config = AutoConfig.from_pretrained("facebook/opt-350m")
layer_start = 1
layer_end = 96
is_first = layer_start == 1
is_last = layer_end == 96
shard_config = ModuleShardConfig(layer_start=layer_start, layer_end=layer_end, is_first=is_first, is_last=is_last)
opt_model_shard = OPTModelShard(config, shard_config, "OPT_350m.npz")
outputs = outputs['last_hidden_state']
outputs1 = opt_model_shard.forward(inputs)

print(outputs == outputs1)
