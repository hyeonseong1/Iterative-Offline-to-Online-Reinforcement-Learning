from typing import Type

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from .simba_layers import SimBaMlpExtractor


class SimBaActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy that replaces the default MLP with the SimBa backbone."""

    def __init__(
        self,
        *args,
        simba_hidden_dim: int = 256,
        simba_num_blocks: int = 2,
        simba_activation_cls: Type[nn.Module] = nn.ReLU,
        **kwargs,
    ):
        self.simba_hidden_dim = simba_hidden_dim
        self.simba_num_blocks = simba_num_blocks
        self.simba_activation_cls = simba_activation_cls
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Override default MLP with SimBa backbone."""
        self.mlp_extractor = SimBaMlpExtractor(
            feature_dim=self.features_dim,
            hidden_dim=self.simba_hidden_dim,
            num_blocks=self.simba_num_blocks,
            activation_factory=self.simba_activation_cls,
        )

