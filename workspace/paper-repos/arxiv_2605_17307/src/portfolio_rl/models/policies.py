"""Dirichlet policy heads (flat + hierarchical)."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Beta, Dirichlet


def _alpha(x: torch.Tensor) -> torch.Tensor:
    """Map raw logits -> positive concentration via softplus + small floor."""
    return torch.nn.functional.softplus(x) + 1e-3


class FlatDirichletPolicy(nn.Module):
    """Dirichlet over (k+1) actions when allow_cash else k actions."""

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 256, allow_cash: bool = True):
        super().__init__()
        self.allow_cash = allow_cash
        self.n_out = n_assets + (1 if allow_cash else 0)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_out),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = _alpha(self.net(state))
        dist = Dirichlet(alpha)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        alpha = _alpha(self.net(state))
        return alpha / alpha.sum(dim=-1, keepdim=True)


class HierarchicalDirichletPolicy(nn.Module):
    """Two-stage: Beta(equity-vs-cash) × Dirichlet(over assets).

    ASSUMED: Beta distribution for equity split — paper does not specify (§5.4.3).
    """

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 256):
        super().__init__()
        self.n_assets = n_assets
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.beta_head = nn.Linear(hidden, 2)             # Beta(α, β) over equity_fraction
        self.dir_head = nn.Linear(hidden, n_assets)        # Dirichlet over assets

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        ab = _alpha(self.beta_head(h))
        beta = Beta(ab[..., 0], ab[..., 1])
        equity = beta.rsample()                            # (B,)
        cash = 1.0 - equity                                # (B,)

        alpha_assets = _alpha(self.dir_head(h))
        dir_ = Dirichlet(alpha_assets)
        asset_w = dir_.rsample()                           # (B, k)

        action = torch.cat([equity.unsqueeze(-1) * asset_w, cash.unsqueeze(-1)], dim=-1)
        log_prob = beta.log_prob(equity).unsqueeze(-1) + dir_.log_prob(asset_w).unsqueeze(-1)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        h = self.trunk(state)
        ab = _alpha(self.beta_head(h))
        equity = ab[..., 0] / ab.sum(dim=-1)
        cash = 1.0 - equity
        alpha_assets = _alpha(self.dir_head(h))
        asset_w = alpha_assets / alpha_assets.sum(dim=-1, keepdim=True)
        return torch.cat([equity.unsqueeze(-1) * asset_w, cash.unsqueeze(-1)], dim=-1)


def build_policy(cfg: dict, state_dim: int, n_assets: int) -> nn.Module:
    pt = cfg["model"]["policy_type"]
    allow_cash = cfg["model"]["allow_cash"]
    if pt == "flat_dirichlet":
        return FlatDirichletPolicy(state_dim, n_assets, allow_cash=allow_cash)
    if pt == "hierarchical_dirichlet":
        if not allow_cash:
            raise ValueError("Hierarchical policy requires allow_cash=True")
        return HierarchicalDirichletPolicy(state_dim, n_assets)
    raise ValueError(f"Unknown policy_type: {pt}")
