"""Models module."""
from typing import Tuple, Type, Union
from torch import nn, Tensor

ModuleShardData: Type = Union[Tensor, Tuple[Tensor, ...]]
"""A module shard input/output type."""


class ModuleShardConfig:
    """Base class for shard configurations (distinct from model configurations)."""
    # pylint: disable=too-few-public-methods

    def __init__(self, **kwargs: dict):
        # Attributes with default values
        self.stage = kwargs.pop('stage', 0)
        self.layer_start = kwargs.pop('layer_start', 0)
        self.layer_end = kwargs.pop('layer_end', 0)

        # Attributes without default values
        for key, value in kwargs.items():
            setattr(self, key, value)


class ModuleShard(nn.Module):
    """Abstract parent class for module shards."""
    # pylint: disable=abstract-method

    def __init__(self, shard_config: ModuleShardConfig):
        super().__init__()
        self.shard_config = shard_config


def get_microbatch_size(shard_data: ModuleShardData, verify: bool=False):
    """Get the microbatch size from shard data."""
    if isinstance(shard_data, Tensor):
        shard_data = (shard_data,)
    ubatch_size = 0 if len(shard_data) == 0 else len(shard_data[0])
    if verify:
        # Sanity check that tensors are the same length
        for tensor in shard_data:
            assert isinstance(tensor, Tensor)
            assert len(tensor) == ubatch_size
    return ubatch_size
