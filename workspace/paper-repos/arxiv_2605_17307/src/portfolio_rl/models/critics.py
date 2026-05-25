"""Twin Q-network for SAC.

ASSUMED: MLP with hidden layers [256, 256] (SAC default).
"""
from __future__ import annotations

import torch
from torch import nn


def _mlp(in_dim: int, hidden: list[int], out_dim: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: list[int] = (256, 256)):
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, list(hidden))
        self.q2 = _mlp(state_dim + action_dim, list(hidden))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
