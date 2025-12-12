from typing import Callable, Tuple

import torch as th
import torch.nn as nn


class RunningStatsNorm(nn.Module):
    """Simple running statistics normalization used in SimBa."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("mu", th.zeros(dim))
        self.register_buffer("var", th.ones(dim))
        self.register_buffer("count", th.zeros(1, dtype=th.long))
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True

        if self.training:
            with th.no_grad():
                batch_mean = x.detach().mean(dim=0)
                batch_var = x.detach().var(dim=0, unbiased=False)
                batch_count = x.shape[0]

                if self.count.item() == 0:
                    self.mu.copy_(batch_mean)
                    self.var.copy_(batch_var + self.eps)
                    self.count += batch_count
                else:
                    total_count = self.count + batch_count
                    delta = batch_mean - self.mu
                    new_mu = self.mu + delta * batch_count / total_count

                    m_a = self.var * self.count.float()
                    m_b = batch_var * batch_count
                    new_var = (m_a + m_b + delta.pow(2) * self.count.float() * batch_count / total_count) / total_count

                    self.mu.copy_(new_mu)
                    self.var.copy_(th.clamp(new_var, min=1e-6))
                    self.count += batch_count

        y = (x - self.mu) / th.sqrt(self.var + self.eps)
        if squeezed:
            y = y.squeeze(0)
        return y


class SimBaResidualBlock(nn.Module):
    """Residual feedforward block with pre-layer normalization."""

    def __init__(self, hidden_dim: int, activation_factory: Callable[[], nn.Module]):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation_factory()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.linear2(y)
        return residual + y


class SimBaBackbone(nn.Module):
    """RSNorm -> Linear -> (Residual Block)xN -> LayerNorm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        activation_factory: Callable[[], nn.Module],
    ):
        super().__init__()
        self.rs_norm = RunningStatsNorm(input_dim)
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [SimBaResidualBlock(hidden_dim, activation_factory) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.rs_norm(x)
        x = self.input_linear(x)
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class SimBaMlpExtractor(nn.Module):
    """Drop-in replacement for SB3's MlpExtractor using the SimBa architecture."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim
        self.policy_net = SimBaBackbone(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation_factory=activation_factory,
        )
        self.value_net = SimBaBackbone(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation_factory=activation_factory,
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

