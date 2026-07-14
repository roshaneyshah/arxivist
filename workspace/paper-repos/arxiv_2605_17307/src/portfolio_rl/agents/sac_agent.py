"""Soft Actor-Critic agent (custom, supports Dirichlet policy).

Implements Table 4 hyperparameters:
    actor_lr = 3e-4, critic_lr = 5e-4, entropy_lr = 3e-4
    alpha = 0.2 (fixed), gamma = 0.99, tau = 0.005
    batch = 128, gradient_steps = 2 per env step
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..models.critics import TwinCritic
from ..models.encoders import CrossSectionalAttention, build_encoder
from ..models.policies import build_policy
from .replay_buffer import ReplayBuffer


@dataclass
class SACConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    gradient_steps: int = 2
    warmup_steps: int = 500
    replay_capacity: int = 20_000


class StateEncoder(nn.Module):
    """Wraps sequence encoder + cross-sectional attention into a single module
    that converts a flat env observation back to (B, H_state)."""

    def __init__(self, cfg: dict, k: int, lookback: int, f_asset: int, f_global: int):
        super().__init__()
        self.k = k
        self.lookback = lookback
        self.f_asset = f_asset
        self.f_global = f_global
        self.seq_encoder, hidden = build_encoder(cfg, f_asset)
        self.xattn = CrossSectionalAttention(hidden, f_global)
        self.out_dim = self.xattn.out_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        asset_size = self.lookback * self.k * self.f_asset
        x = obs[:, :asset_size].reshape(B, self.lookback, self.k, self.f_asset)
        x = x.permute(0, 2, 1, 3).contiguous()           # (B, k, T, F)
        g = obs[:, asset_size:]
        emb = self.seq_encoder(x)                        # (B, k, H)
        return self.xattn(emb, g)                        # (B, H_state)


class SACAgent:
    def __init__(self, cfg: dict, env, device: torch.device):
        self.cfg = cfg
        self.device = device
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        n_assets = env.k

        lookback = env.lookback
        f_asset = env.F_asset
        f_global = env.F_global

        self.encoder = StateEncoder(cfg, n_assets, lookback, f_asset, f_global).to(device)
        state_dim = self.encoder.out_dim

        self.policy = build_policy(cfg, state_dim, n_assets).to(device)
        self.critic = TwinCritic(state_dim, act_dim, cfg["model"]["critic_hidden"]).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)

        self.opt_actor = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=cfg["training"]["lr_actor"],
        )
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg["training"]["lr_critic"])

        self.cfg_sac = SACConfig(
            lr_actor=cfg["training"]["lr_actor"],
            lr_critic=cfg["training"]["lr_critic"],
            alpha=cfg["training"]["alpha"],
            gamma=cfg["training"]["gamma"],
            tau=cfg["training"]["tau"],
            batch_size=cfg["training"]["batch_size"],
            gradient_steps=cfg["training"]["gradient_steps"],
            warmup_steps=cfg["training"]["warmup_steps"],
            replay_capacity=cfg["training"]["replay_capacity"],
        )
        self.buffer = ReplayBuffer(self.cfg_sac.replay_capacity)
        self.steps = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = self.encoder(obs_t)
        if deterministic:
            action = self.policy.deterministic(state)
        else:
            action, _ = self.policy(state)
        return action.squeeze(0).cpu().numpy()

    def update(self) -> dict | None:
        if len(self.buffer) < max(self.cfg_sac.warmup_steps, self.cfg_sac.batch_size):
            return None
        losses = {}
        for _ in range(self.cfg_sac.gradient_steps):
            s, a, r, s_next, done = self.buffer.sample(self.cfg_sac.batch_size, self.device)
            with torch.no_grad():
                state_next = self.encoder(s_next)
                a_next, log_p_next = self.policy(state_next)
                q1_t, q2_t = self.critic_target(state_next, a_next)
                q_target = torch.min(q1_t, q2_t) - self.cfg_sac.alpha * log_p_next
                y = r + (1 - done) * self.cfg_sac.gamma * q_target

            state = self.encoder(s)
            q1, q2 = self.critic(state.detach(), a)
            critic_loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

            state = self.encoder(s)  # re-encode with grads for actor
            a_new, log_p_new = self.policy(state)
            q1_new, q2_new = self.critic(state, a_new)
            actor_loss = (self.cfg_sac.alpha * log_p_new - torch.min(q1_new, q2_new)).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            with torch.no_grad():
                tau = self.cfg_sac.tau
                for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)

            losses = {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
        return losses
