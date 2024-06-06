import logging
import torch
from torch import nn
from torch.nn import Conv2d, ReLU, AdaptiveAvgPool2d, MaxPool2d
from torchvision import models
from .. import ModuleShard, ModuleShardConfig


logger = logging.getLogger(__name__)


class AlexNetConfig:
    def __init__(self, model=None):
        self.info = {}
        if model:
            self.generate_config(model)
    
    def get_layer_info(self, layer):
        info = {}

        if isinstance(layer, nn.Conv2d):
            info['in_channels'] = layer.in_channels
            info['out_channels'] = layer.out_channels
            info['kernel_size'] = layer.kernel_size
            info['stride'] = layer.stride
            info['padding'] = layer.padding
            info['bias'] = layer.bias is not None

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

        elif isinstance(layer, nn.Dropout):
            info['p'] = layer.p
            info['inplace'] = layer.inplace

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
    
    def __getattr__(self, name):
        if name in self.info:
            return self.info[name]

     
class AlexNetLayerShard(ModuleShard):
    def __init__(self, config, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv5 = None

        self.relu1 = None
        self.relu2 = None
        self.relu3 = None
        self.relu4 = None
        self.relu5 = None

        self.maxpool1 = None
        self.maxpool2 = None
        self.maxpool5 = None

        self._build_shard()

    def _build_shard(self):
        if self.has_layer(1):
            self.conv1 = Conv2d(**self.config["features_0"])
            self.relu1 = ReLU(**self.config["features_1"])
            self.maxpool1 = MaxPool2d(**self.config["features_2"])
        if self.has_layer(2):
            self.conv2 = Conv2d(**self.config["features_3"])
            self.relu2 = ReLU(**self.config["features_4"])
            self.maxpool2 = MaxPool2d(**self.config["features_5"])
        if self.has_layer(3):
            self.conv3 = Conv2d(**self.config["features_6"])
            self.relu3 = ReLU(**self.config["features_7"])
        if self.has_layer(4):
            self.conv4 = Conv2d(**self.config["features_8"])
            self.relu4 = ReLU(**self.config["features_9"])
        if self.has_layer(5):
            self.conv5 = Conv2d(**self.config["features_10"])
            self.relu5 = ReLU(**self.config["features_11"])
            self.maxpool5 = MaxPool2d(**self.config["features_12"])

    @torch.no_grad()
    def forward(self, data_pack):
        """Compute layer shard."""
        data = data_pack
        if self.has_layer(1):
            print("in self.has_layer(1)")
            data_conv = self.conv1(data)
            data_relu = self.relu1(data_conv)
            data = self.maxpool1(data_relu)
        if self.has_layer(2):
            print(data.shape)
            data_conv = self.conv2(data)
            data_relu = self.relu2(data_conv)
            data = self.maxpool2(data_relu)
        if self.has_layer(3):
            data_conv = self.conv3(data)
            data = self.relu3(data_conv)
        if self.has_layer(4):
            data_conv = self.conv4(data)
            data = self.relu4(data_conv)
        if self.has_layer(5):
            data_conv = self.conv5(data)
            data_relu = self.relu5(data_conv)
            data = self.maxpool5(data_relu)
        return data
    
    def load_weight(self, weight):
        if self.has_layer(1):
            self.conv1.load_state_dict(weight[0].state_dict())
            self.relu1.load_state_dict(weight[1].state_dict())
            self.maxpool1.load_state_dict(weight[2].state_dict())
        if self.has_layer(2):
            self.conv2.load_state_dict(weight[3].state_dict())
            self.relu2.load_state_dict(weight[4].state_dict())
            self.maxpool2.load_state_dict(weight[5].state_dict())
        if self.has_layer(3):
            self.conv3.load_state_dict(weight[6].state_dict())
            self.relu3.load_state_dict(weight[7].state_dict())
            
        if self.has_layer(4):
            self.conv4.load_state_dict(weight[8].state_dict())
            self.relu4.load_state_dict(weight[9].state_dict())
            
        if self.has_layer(5):
            self.conv5.load_state_dict(weight[10].state_dict())
            self.relu5.load_state_dict(weight[11].state_dict())
            self.maxpool5.load_state_dict(weight[12].state_dict())


class AlexNetModelShard(ModuleShard):

    def __init__(self, config, shard_config: ModuleShardConfig,
                 model_weights):
        super().__init__(config, shard_config)
        self.layers = nn.ModuleList()

        self.avgpool = None
        self.drop1 = None
        self.fc1 = None
        self.relu_fc1 = None
        self.drop2 = None
        self.fc2 = None
        self.relu_fc2 = None
        self.fc3 = None

        # self.model = model_weights
        self._build_shard(model_weights)

    def _build_shard(self, weights):
        # print(weights)
        layer_curr = self.shard_config.layer_start
        while layer_curr <= self.shard_config.layer_end:
            print(layer_curr)
           
            layer_config = ModuleShardConfig(layer_start=layer_curr, layer_end=layer_curr,
                                                 is_first = True, is_last = False)
            sub_model_config = self.config
            layer = AlexNetLayerShard(sub_model_config, layer_config)
            # print(layer)
            self._load_weights_layer(weights, layer)
            self.layers.append(layer)

            layer_curr += 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load config for the last shard")
            self.avgpool = nn.AdaptiveAvgPool2d(**self.config['avgpool'])
            self.drop1 = nn.Dropout(**self.config['classifier_0'])
            self.fc1 = nn.Linear(**self.config['classifier_1'])
            self.relu_fc1 = nn.ReLU(**self.config['classifier_2'])
            self.drop2 = nn.Dropout(**self.config['classifier_3'])
            self.fc2 = nn.Linear(**self.config['classifier_4'])
            self.relu_fc2 = nn.ReLU(**self.config['classifier_5'])
            self.fc3 = nn.Linear(**self.config['classifier_6'])
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.avgpool.load_state_dict(weights.avgpool.state_dict())
        self.drop1.load_state_dict(weights.classifier[0].state_dict())
        self.fc1.load_state_dict(weights.classifier[1].state_dict())
        self.relu_fc1.load_state_dict(weights.classifier[2].state_dict())
        self.drop2.load_state_dict(weights.classifier[3].state_dict())
        self.fc2.load_state_dict(weights.classifier[4].state_dict())
        self.relu_fc2.load_state_dict(weights.classifier[5].state_dict())
        self.fc3.load_state_dict(weights.classifier[6].state_dict())

    @torch.no_grad()
    def _load_weights_layer(self, weights, layer):
        if layer.has_layer(1):
            layer.conv1.load_state_dict(weights.features[0].state_dict())
            layer.relu1.load_state_dict(weights.features[1].state_dict())
            layer.maxpool1.load_state_dict(weights.features[2].state_dict())
        if layer.has_layer(2):
            layer.conv2.load_state_dict(weights.features[3].state_dict())
            layer.relu2.load_state_dict(weights.features[4].state_dict())
            layer.maxpool2.load_state_dict(weights.features[5].state_dict())
        if layer.has_layer(3):
            layer.conv3.load_state_dict(weights.features[6].state_dict())
            layer.relu3.load_state_dict(weights.features[7].state_dict())
            
        if layer.has_layer(4):
            layer.conv4.load_state_dict(weights.features[8].state_dict())
            layer.relu4.load_state_dict(weights.features[9].state_dict())
            
        if layer.has_layer(5):
            layer.conv5.load_state_dict(weights.features[10].state_dict())
            layer.relu5.load_state_dict(weights.features[11].state_dict())
            layer.maxpool5.load_state_dict(weights.features[12].state_dict())
    
    @torch.no_grad()
    def forward(self, data):
        """Compute shard layers."""
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.avgpool(data)
            data = torch.flatten(data, 1)
            data = self.drop1(data)
            data = self.fc1(data)
            data = self.relu_fc1(data)
            data = self.drop2(data)
            data = self.fc2(data)
            data = self.relu_fc2(data)
            data = self.fc3(data)
        return data