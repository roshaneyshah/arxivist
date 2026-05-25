"""Gymnasium portfolio environment implementing reward (Eq. 16/17), turnover and HHI penalties."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """One step = one trading day. State = flattened (k, T, F_asset) + (F_global,)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        asset_features: np.ndarray,      # (T_full, k, F_asset)  — pre-sliced for the fold
        global_features: np.ndarray,     # (T_full, F_global)
        asset_returns: np.ndarray,       # (T_full, k)  — simple daily returns
        benchmark_returns: np.ndarray,   # (T_full,)
        lookback_window: int = 60,
        transaction_cost_bps: float = 2.0,
        lambda_turnover: float = 0.003,
        lambda_concentration: float = 0.1,
        reward_type: str = "log_return_absolute",
        allow_cash: bool = True,
    ):
        super().__init__()
        self.X = asset_features.astype(np.float32)
        self.G = global_features.astype(np.float32)
        self.R = asset_returns.astype(np.float32)
        self.R_b = benchmark_returns.astype(np.float32)
        self.T_full, self.k, self.F_asset = self.X.shape
        self.F_global = self.G.shape[1]
        self.lookback = lookback_window
        self.tc = transaction_cost_bps / 10_000.0
        self.lambda_to = lambda_turnover
        self.lambda_conc = lambda_concentration
        self.reward_type = reward_type
        self.allow_cash = allow_cash
        self.k_eff = self.k + (1 if allow_cash else 0)

        obs_dim = self.lookback * self.k * self.F_asset + self.F_global
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(self.k_eff,), dtype=np.float32)

        self.t = self.lookback
        self.w_prev = self._uniform_weights()

    def _uniform_weights(self) -> np.ndarray:
        w = np.ones(self.k_eff, dtype=np.float32) / self.k_eff
        return w

    def _get_obs(self) -> np.ndarray:
        window = self.X[self.t - self.lookback : self.t]   # (lookback, k, F)
        g = self.G[self.t - 1]
        # impute NaNs with zeros (warmup)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate([window.flatten(), g]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.w_prev = self._uniform_weights()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # normalise to simplex
        a = np.clip(action, 0.0, None) + 1e-8
        w_new = a / a.sum()

        r_assets = self.R[self.t]  # (k,)
        if self.allow_cash:
            r_p = float(np.dot(w_new[:-1], r_assets))
        else:
            r_p = float(np.dot(w_new, r_assets))

        turnover = float(np.abs(w_new - self.w_prev).sum())
        r_p_net = r_p - self.tc * turnover / 2.0

        log_r = np.log1p(r_p_net)
        hhi = float((w_new ** 2).sum())
        hhi_min = 1.0 / self.k_eff
        penalty = self.lambda_to * turnover * 100.0 + self.lambda_conc * (hhi - hhi_min) * 100.0

        if self.reward_type == "benchmark_relative":
            log_b = np.log1p(self.R_b[self.t])
            reward = 1000.0 * (log_r - log_b) - penalty
        else:
            reward = 1000.0 * log_r - penalty

        self.w_prev = w_new
        self.t += 1
        terminated = self.t >= self.T_full - 1
        info = {
            "portfolio_return": r_p_net,
            "turnover": turnover,
            "hhi": hhi,
            "weights": w_new,
        }
        return self._get_obs(), float(reward), bool(terminated), False, info
