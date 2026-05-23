"""
utils/config.py — Configuration loading and seed utilities.

Implements experiment configuration management for the RL trade execution system.
All random state is managed through this module to ensure reproducibility.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class ExperimentConfig:
    """Top-level configuration container for RL trade execution experiments.

    Loaded from a YAML file. All paths are relative to the project root
    unless they are absolute.

    Paper reference: Section 3 (Experimental Methodology), Section 5.
    """

    # Experiment identity
    stock: str = "AMZN"
    V: int = 10000
    H_minutes: int = 2
    side: str = "sell"
    seed: int = 42

    # State space
    T: int = 8
    I: int = 8
    market_variables: List[str] = field(default_factory=lambda: ["bid_ask_spread", "immediate_market_order_cost"])
    n_bins_market: int = 3  # ASSUMED: confidence 0.80

    # Action space
    L: int = 21             # ASSUMED: confidence 0.60 — TODO: verify from paper
    action_min: int = -6
    action_max: int = 14

    # Training
    data_path: str = "data/raw/"
    train_months: int = 12
    test_months: int = 6
    log_every_n_steps: int = 1000
    save_checkpoint: bool = True
    checkpoint_dir: str = "models/"

    # Evaluation
    output_dir: str = "results/"

    # Data
    signed_volume_window_seconds: int = 15
    episode_overlap: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            ExperimentConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If required fields are missing or invalid.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()

        # Flatten nested YAML into flat config attributes
        exp = raw.get("experiment", {})
        cfg.stock = exp.get("stock", cfg.stock)
        cfg.V = exp.get("V", cfg.V)
        cfg.H_minutes = exp.get("H_minutes", cfg.H_minutes)
        cfg.side = exp.get("side", cfg.side)
        cfg.seed = exp.get("seed", cfg.seed)

        ss = raw.get("state_space", {})
        cfg.T = ss.get("T", cfg.T)
        cfg.I = ss.get("I", cfg.I)
        cfg.market_variables = ss.get("market_variables", cfg.market_variables)
        cfg.n_bins_market = ss.get("n_bins_market", cfg.n_bins_market)

        act = raw.get("action_space", {})
        cfg.L = act.get("L", cfg.L)
        cfg.action_min = act.get("action_min", cfg.action_min)
        cfg.action_max = act.get("action_max", cfg.action_max)

        tr = raw.get("training", {})
        cfg.data_path = tr.get("data_path", cfg.data_path)
        cfg.train_months = tr.get("train_months", cfg.train_months)
        cfg.test_months = tr.get("test_months", cfg.test_months)
        cfg.log_every_n_steps = tr.get("log_every_n_steps", cfg.log_every_n_steps)
        cfg.save_checkpoint = tr.get("save_checkpoint", cfg.save_checkpoint)
        cfg.checkpoint_dir = tr.get("checkpoint_dir", cfg.checkpoint_dir)

        ev = raw.get("evaluation", {})
        cfg.output_dir = ev.get("output_dir", cfg.output_dir)

        da = raw.get("data", {})
        cfg.signed_volume_window_seconds = da.get("signed_volume_window_seconds", cfg.signed_volume_window_seconds)
        cfg.episode_overlap = da.get("episode_overlap", cfg.episode_overlap)

        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate configuration values. Raises ValueError on invalid config."""
        if self.stock not in ("AMZN", "NVDA", "QCOM"):
            raise ValueError(f"stock must be one of AMZN/NVDA/QCOM, got: {self.stock}")
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got: {self.side}")
        if self.V <= 0:
            raise ValueError(f"V (shares) must be positive, got: {self.V}")
        if self.H_minutes <= 0:
            raise ValueError(f"H_minutes must be positive, got: {self.H_minutes}")
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got: {self.T}")
        if self.I < 1:
            raise ValueError(f"I must be >= 1, got: {self.I}")
        if self.action_max - self.action_min + 1 != self.L:
            raise ValueError(
                f"L ({self.L}) must equal action_max - action_min + 1 "
                f"({self.action_max} - {self.action_min} + 1 = {self.action_max - self.action_min + 1})"
            )

    def action_index(self, a: int) -> int:
        """Convert a raw action value to a 0-based index into the action array."""
        return a - self.action_min

    def index_to_action(self, idx: int) -> int:
        """Convert a 0-based action index back to the raw action value."""
        return idx + self.action_min

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(stock={self.stock}, V={self.V}, H={self.H_minutes}min, "
            f"T={self.T}, I={self.I}, L={self.L}, "
            f"market_vars={self.market_variables})"
        )
