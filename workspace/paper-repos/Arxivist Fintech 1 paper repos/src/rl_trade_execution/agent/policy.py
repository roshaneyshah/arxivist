"""
agent/policy.py — Optimal execution policy extracted from a trained Q-table.

Paper reference: Section 3 — "Select the highest-payout action argmax c(y,p)
in every state y to output optimal policy."
"""

from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np

from rl_trade_execution.agent.q_table import QTable
from rl_trade_execution.env.market_env import TradeExecutionEnv
from rl_trade_execution.utils.config import ExperimentConfig


class OptimalPolicy:
    """Greedy policy derived from a trained Q-table.

    At each state, selects the action with minimum expected execution cost.

    Paper reference: Section 3, Optimal_strategy pseudocode:
      "Select the highest-payout action argmax c(y, p) in every state y"

    Attributes:
        q_table: Trained Q-table.
        config: Experiment configuration (for action decoding).
    """

    def __init__(self, q_table: QTable, config: ExperimentConfig):
        """
        Args:
            q_table: Trained Q-table with cost estimates.
            config: Experiment configuration.
        """
        self.q_table = q_table
        self.config = config

    def act(self, state_idx: int) -> int:
        """Select the cost-minimizing action for a given state.

        Args:
            state_idx: Flat encoded state index.

        Returns:
            0-based action index (maps to price offset via config.index_to_action).
        """
        return self.q_table.best_action(state_idx)

    def act_decoded(self, state_idx: int) -> int:
        """Select the cost-minimizing action and return as raw price offset.

        Args:
            state_idx: Flat encoded state index.

        Returns:
            Raw price offset (e.g., -3 means 3 ticks below ask for a sell order).
        """
        action_idx = self.act(state_idx)
        return self.config.index_to_action(action_idx)

    def save(self, path: str) -> None:
        """Serialize policy to disk.

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q_table_path": None, "q_table": self.q_table, "config": self.config}, f)

    @classmethod
    def load(cls, path: str) -> "OptimalPolicy":
        """Load a saved policy from disk.

        Args:
            path: File path previously saved with OptimalPolicy.save().

        Returns:
            Loaded OptimalPolicy instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["q_table"], data["config"])

    @classmethod
    def from_q_table(cls, q_table: QTable, config: ExperimentConfig) -> "OptimalPolicy":
        """Construct a policy from a trained Q-table.

        Args:
            q_table: Trained Q-table.
            config: Experiment config.

        Returns:
            OptimalPolicy instance.
        """
        return cls(q_table, config)

    def __repr__(self) -> str:
        return f"OptimalPolicy(config={self.config}, q_table={self.q_table})"
