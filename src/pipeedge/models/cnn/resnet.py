from typing import Optional, Union
import logging
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d
import numpy as np
from torchvision import models
from torchvision.models.resnet import ResNet
from .. import ModuleShard, ModuleShardConfig
from . import CNNShardData

logger = logging.getLogger(__name__)

class ResnetConfig:
    def __init__(self, model_name=None):
        self.name_or_path = ''
        self.info = {}
        torch_models = getattr(models, model_name.split('/')[1])
        model = torch_models(pretrained=True)
        if model:
            self.name_or_path = model.__class__.__name__
            self.generate_config(model)

    def get_layer_info(self, layer):
        info = {}
        if isinstance(layer, models.resnet.BasicBlock) or isinstance(layer, models.resnet.Bottleneck):
            for sub_name, sub_child in layer.named_children():
                if sub_name == "downsample":
                    info["downsample_conv"] = self.get_layer_info(sub_child[0])
                    info["downsample_bn"] = self.get_layer_info(sub_child[1])
                else:
                    info[sub_name] = self.get_layer_info(sub_child)

        elif isinstance(layer, nn.Conv2d):
            info['in_channels'] = layer.in_channels
            info['out_channels'] = layer.out_channels
            info['kernel_size'] = layer.kernel_size
            info['stride'] = layer.stride
            info['padding'] = layer.padding
            info['bias'] = layer.bias is not None

        elif isinstance(layer, nn.BatchNorm2d):
            info['num_features'] = layer.num_features
            info['eps'] = layer.eps
            info['momentum'] = layer.momentum
            info['affine'] = layer.affine
            info['track_running_stats'] = layer.track_running_stats

        elif isinstance(layer, nn.ReLU):
            info['inplace'] = layer.inplace

        elif isinstance(layer, nn.MaxPool2d):
            info['kernel_size'] = layer.kernel_size
            info['stride'] = layer.stride
            info['padding'] = layer.padding
            info['dilation'] = layer.dilation
            info['ceil_mode'] = layer.ceil_mode

        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            info['output_size'] = layer.output_size

        elif isinstance(layer, nn.Linear):
            info['in_features'] = layer.in_features
            info['out_features'] = layer.out_features
            info['bias'] = layer.bias is not None

        return info

    def generate_config(self, model):
        for name, child in model.named_children():
            if list(child.children()): 
                for sub_name, sub_child in child.named_children():
                    self.info[f"{name}_{sub_name}"] = self.get_layer_info(sub_child)
            else:
                self.info[name] = self.get_layer_info(child)

    def __getitem__(self, key):
        return self.info[key]



class ResNetLayerShard_BasicBlock(ModuleShard):
    def __init__(self, config: ResnetConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.conv2 = None
        self.bn2 = None
        self.downsample_conv = None
        self.downsample_bn = None

        self._build_shard()

    def _build_shard(self):
        if self.has_layer(0):
            self.conv1 = Conv2d(**self.config["conv1"])
            self.bn1 = BatchNorm2d(**self.config["bn1"])
            self.relu = ReLU(**self.config['relu'])
        if self.has_layer(1):
            self.conv2 = Conv2d(**self.config["conv2"])
            self.bn2 = BatchNorm2d(**self.config["bn2"])
        if self.has_layer(2):
            self.downsample_conv = Conv2d(**self.config["downsample_conv"])
            self.downsample_bn = BatchNorm2d(**self.config["downsample_bn"])
        if self.shard_config.is_last:
            self.relu = ReLU(inplace=True)

    @torch.no_grad()
    def forward(self, data_pack: CNNShardData):
        """Compute layer shard."""
        data = data_pack[0]
        identity = data_pack[1]
        if self.has_layer(0):
            data_conv = self.conv1(data)
            data_bn = self.bn1(data_conv)
            data = self.relu(data_bn)
        if self.has_layer(1):
            data_conv = self.conv2(data)
            data = self.bn2(data_conv)
        if self.has_layer(2):
            data_conv = self.downsample_conv(identity)
            identity = self.downsample_bn(data_conv)
        if self.shard_config.is_last:
            data += identity
            data = self.relu(data)
            return data, data
        return data, identity
    
    # For unit test only
    def load_weight(self, weight):
        if self.has_layer(0):
            self.conv1.load_state_dict(weight.conv1.state_dict())
            self.bn1.load_state_dict(weight.bn1.state_dict())
            self.relu.load_state_dict(weight.relu.state_dict())
        if self.has_layer(1):
            self.conv2.load_state_dict(weight.conv2.state_dict())
            self.bn2.load_state_dict(weight.bn2.state_dict())
        if self.has_layer(2):
            self.downsample_conv.load_state_dict(weight.downsample[0].state_dict())
            self.downsample_bn.load_state_dict(weight.downsample[1].state_dict())

class ResNetLayerShard_Bottleneck(ModuleShard):
    def __init__(self, config: ResnetConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.conv1 = None
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.conv3 = None
        self.bn3 = None
        self.relu = None
        self.downsample_conv = None
        self.downsample_bn = None

        self._build_shard()

    def _build_shard(self):
        if self.has_layer(0):
            self.conv1 = Conv2d(**self.config["conv1"])
            self.bn1 = BatchNorm2d(**self.config["bn1"])
            self.relu = ReLU(inplace=True)
        if self.has_layer(1):
            self.conv2 = Conv2d(**self.config["conv2"])
            self.bn2 = BatchNorm2d(**self.config["bn2"])
            self.relu = ReLU(inplace=True)
        if self.has_layer(2):
            self.conv3 = Conv2d(**self.config["conv3"])
            self.bn3 = BatchNorm2d(**self.config["bn3"])
        if self.has_layer(3):
            self.downsample_conv = Conv2d(**self.config["downsample_conv"])
            self.downsample_bn = BatchNorm2d(**self.config["downsample_bn"])
        if self.shard_config.is_last:
            self.relu = ReLU(inplace=True)

    @torch.no_grad()
    def forward(self, data_pack: CNNShardData):
        """Compute layer shard."""
        data = data_pack[0]
        identity = data_pack[1]
        if self.has_layer(0):
            data_conv = self.conv1(data)
            data_bn = self.bn1(data_conv)
            data = self.relu(data_bn)
        if self.has_layer(1):
            data_conv = self.conv2(data)
            data_bn = self.bn2(data_conv)
            data = self.relu(data_bn)
        if self.has_layer(2):
            data_conv = self.conv3(data)
            data = self.bn3(data_conv)
            
        if self.has_layer(3):
            data_conv = self.downsample_conv(identity)
            identity = self.downsample_bn(data_conv)
        if self.shard_config.is_last:
            data += identity
            data = self.relu(data)
            return data, data
        return data, identity
    
    # For unit test only
    def load_weight(self, weight):
        if self.has_layer(0):
            self.conv1.load_state_dict(weight.conv1.state_dict())
            self.bn1.load_state_dict(weight.bn1.state_dict())
            self.relu.load_state_dict(weight.relu.state_dict())
        if self.has_layer(1):
            self.conv2.load_state_dict(weight.conv2.state_dict())
            self.bn2.load_state_dict(weight.bn2.state_dict())
            self.relu.load_state_dict(weight.relu.state_dict())
        if self.has_layer(2):
            self.conv3.load_state_dict(weight.conv3.state_dict())
            self.bn3.load_state_dict(weight.bn3.state_dict())
            
        if self.has_layer(3):
            self.downsample_conv.load_state_dict(weight.downsample[0].state_dict())
            self.downsample_bn.load_state_dict(weight.downsample[1].state_dict())
            self.relu.load_state_dict(weight.relu.state_dict())

layershard_type = {
    18: ResNetLayerShard_BasicBlock,
    34: ResNetLayerShard_BasicBlock,
    50: ResNetLayerShard_Bottleneck,
    101: ResNetLayerShard_Bottleneck
}

class ResNetModelShard(ModuleShard):
    def __init__(self, config: ResnetConfig, shard_config: ModuleShardConfig,
                 model_weights: ResNet):
        super().__init__(config, shard_config)
        self.conv1 = None
        self.bn1 = None
        self.relu = None
        self.maxpool = None

        self.layers = nn.ModuleList()

        self.avgpool = None
        self.fc = None

        self.init_version_and_map()

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            weights = torch.load(model_weights)
            self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def init_version_and_map(self):
        self.version = 18
        self.bu_num = 21

        self.layer_map = {
            1: (2,5),
            2: (6,10),
            3: (11,15),
            4: (16,20)
        }

        self.block_map = {
            1: [2,2],
            2: [3,2],
            3: [3,2],
            4: [3,2]
        }

    def _build_shard(self, weights):
        if self.shard_config.is_first:
            self.conv1 = Conv2d(**self.config['conv1'])
            self.bn1 = BatchNorm2d(**self.config['bn1'])
            self.relu = ReLU(**self.config['relu'])
            self.maxpool = MaxPool2d(**self.config['maxpool'])
            self._load_weights_first(weights)
        if self.shard_config.layer_end > 1 and self.shard_config.layer_start < self.bu_num:
            layer_curr = 2 if self.shard_config.layer_start == 1 else self.shard_config.layer_start
            layer_end = self.shard_config.layer_end
            stop_flag = False
            for layer_id in self.layer_map:
                ori_layer_start = self.layer_map[layer_id][0]
                ori_layer_end = self.layer_map[layer_id][1]
                if  ori_layer_start <= layer_curr <= ori_layer_end:
                    offset =  0
                    bb_map = self.block_map[layer_id]
                    for layer_sub_id in range(len(bb_map)):
                        basic_block_range = bb_map[layer_sub_id]
                        if ori_layer_start + offset <= layer_curr < ori_layer_start + offset + basic_block_range:
                            sublayer_start = layer_curr - (ori_layer_start + offset)
                            sub_layer_is_first = True if sublayer_start == 0 else False
                            if layer_end >= ori_layer_start + offset + basic_block_range-1:
                                sublayer_end = basic_block_range - 1
                                sub_layer_is_last = True
                                if layer_end == ori_layer_start + offset + basic_block_range-1:
                                    stop_flag = True
                                else:
                                    layer_curr = ori_layer_start + offset + basic_block_range 
                            else:
                                sublayer_end = layer_end - (ori_layer_start + offset)
                                sub_layer_is_last = False
                                layer_curr = layer_end
                                stop_flag = True
                            layer_config = ModuleShardConfig(layer_start=sublayer_start, layer_end=sublayer_end
                                                            ,is_first = sub_layer_is_first, is_last = sub_layer_is_last)
                            sub_model_config = self.config[f'layer{layer_id}_{layer_sub_id}']
                            layer = layershard_type[self.version](sub_model_config, layer_config)
                            self._load_weights_layer(weights.__getattr__(f'layer{layer_id}')[layer_sub_id], layer)
                            self.layers.append(layer)
                        if stop_flag:
                            break
                        offset += basic_block_range

                    if stop_flag:
                        break
                if stop_flag:
                    break


        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.avgpool = nn.AdaptiveAvgPool2d(**self.config["avgpool"])
            self.fc = nn.Linear(**self.config["fc"])
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.conv1.load_state_dict(weights.conv1.state_dict())
        self.bn1.load_state_dict(weights.bn1.state_dict())
        self.relu.load_state_dict(weights.relu.state_dict())
        self.maxpool.load_state_dict(weights.maxpool.state_dict())

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.avgpool.load_state_dict(weights.avgpool.state_dict())
        self.fc.load_state_dict(weights.fc.state_dict())

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer):
        layer.load_weight(weights)

    @torch.no_grad()
    def forward(self, data: CNNShardData):
        """Compute shard layers."""
        if self.shard_config.is_first:
            data = self.conv1(data)
            data = self.bn1(data)
            data = self.relu(data)
            data = self.maxpool(data)
            data = [data, data]
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.avgpool(data[0])
            data = torch.flatten(data, 1)
            data = self.fc(data)
        return data
    
    @staticmethod
    def save_weights(model_name: str, model_file: str) -> None:
        torch_models = getattr(models, model_name.split('/')[1])
        model = torch_models(pretrained=True)
        # save model.state_dict() will generate a OrderedDict Object, save model will generate a ResNet Object, to compatible with loading function, need to use save(model) here
        torch.save(model, model_file)


class ResNet18ModelShard(ResNetModelShard):
    def __init__(self, config: ResnetConfig, shard_config: ModuleShardConfig,
                 model_weights: ResNet):
        super().__init__(config, shard_config, model_weights)

    def init_version_and_map(self):
        self.version = 18
        self.bu_num = 21

        self.layer_map = {
            1: (2,5),
            2: (6,10),
            3: (11,15),
            4: (16,20)
        }

        self.block_map = {
            1: [2,2],
            2: [3,2],
            3: [3,2],
            4: [3,2]
        }

class ResNet34ModelShard(ResNetModelShard):
    def __init__(self, config: ResnetConfig, shard_config: ModuleShardConfig,
                 model_weights: ResNet):
        super().__init__(config, shard_config, model_weights)

    def init_version_and_map(self):
        self.version = 34
        self.bu_num = 37
        self.layer_map = {
            1: (2,7),
            2: (8,16),
            3: (17,29),
            4: (30,36)
        }

        self.block_map = {
            1: [2,2,2],
            2: [3,2,2,2],
            3: [3,2,2,2,2,2],
            4: [3,2,2]
        }

class ResNet50ModelShard(ResNetModelShard):
    def __init__(self, config: ResnetConfig, shard_config: ModuleShardConfig,
                 model_weights: ResNet):
        super().__init__(config, shard_config, model_weights)

    def init_version_and_map(self):
        self.version = 50
        self.bu_num = 54
        self.layer_map = {
            1: (2,11),
            2: (12,24),
            3: (25,43),
            4: (44,53)
        }

        self.block_map = {
            1: [4,3,3],
            2: [4,3,3,3],
            3: [4,3,3,3,3,3],
            4: [4,3,3]
        }

class ResNet101ModelShard(ResNetModelShard):
    def __init__(self, config: ResnetConfig, shard_config: ModuleShardConfig,
                 model_weights: ResNet):
        super().__init__(config, shard_config, model_weights)

    def init_version_and_map(self):
        self.version = 101
        self.bu_num = 105
        self.layer_map = {
            1: (2,11),
            2: (12,24),
            3: (25,94),
            4: (95,104)
        }

        self.block_map = {
            1: [4,3,3],
            2: [4,3,3,3],
            3: [4]+[3]*22,
            4: [4,3,3]
        }