"""
agent/q_table.py — Tabular Q-value (cost) table with incremental update.

Implements the Q-learning update rule from Section 3 of:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

Core update equation (Section 3, Algorithm):
  c(x, a) = n/(n+1) * c(x, a) + 1/(n+1) * [c_im(x, a) + argmax_p c(y, p)]

where:
  c(x, a)     = expected execution cost of action a in state x
  c_im(x, a)  = immediate (1-step) execution cost
  y           = next state after taking a in x
  n           = number of times (x, a) has been visited
  argmax_p    = best (lowest cost) action in next state y
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np


class QTable:
    """Tabular cost table mapping (state, action) -> expected execution cost.

    Costs are measured in basis points (bps). Lower cost = better execution.
    The "Q-value" here is actually an expected COST, so we minimize (not maximize).

    Paper reference: Section 3 — "Q-learning and dynamic programming" hybrid.

    Attributes:
        n_states: Total number of discrete states.
        n_actions: Number of discrete actions (L).
        costs: Float64 array of shape [n_states, n_actions].
        visit_counts: Integer array of shape [n_states, n_actions].
    """

    def __init__(self, n_states: int, n_actions: int, init_value: float = 1e6):
        """
        Args:
            n_states: Total number of discrete states in the state space.
            n_actions: Number of discrete actions L.
            init_value: Initial cost estimate for unvisited state-action pairs.
                        High default encourages exploration by discouraging immediate dismissal.
        """
        assert n_states > 0, f"n_states must be positive, got {n_states}"
        assert n_actions > 0, f"n_actions must be positive, got {n_actions}"

        self.n_states = n_states
        self.n_actions = n_actions

        # Initialize all costs to a large value (unvisited states look expensive)
        self.costs = np.full((n_states, n_actions), init_value, dtype=np.float64)
        self.visit_counts = np.zeros((n_states, n_actions), dtype=np.int64)

    def get(self, state_idx: int, action_idx: int) -> float:
        """Get cost estimate for (state, action).

        Args:
            state_idx: Flat state index.
            action_idx: 0-based action index.

        Returns:
            Expected execution cost in basis points.
        """
        assert 0 <= state_idx < self.n_states, (
            f"state_idx {state_idx} out of range [0, {self.n_states})"
        )
        assert 0 <= action_idx < self.n_actions, (
            f"action_idx {action_idx} out of range [0, {self.n_actions})"
        )
        return float(self.costs[state_idx, action_idx])

    def update(
        self, state_idx: int, action_idx: int, immediate_cost: float, best_future_cost: float
    ) -> None:
        """Incremental averaging update for Q-value.

        Implements: c(x, a) = n/(n+1) * c(x,a) + 1/(n+1) * [c_im(x,a) + best_future]

        Paper reference: Section 3, cost update rule equation.

        Args:
            state_idx: Flat state index.
            action_idx: 0-based action index.
            immediate_cost: c_im(x, a) in basis points.
            best_future_cost: argmax_p c(y, p) — best cost achievable from next state.
        """
        n = self.visit_counts[state_idx, action_idx]
        current = self.costs[state_idx, action_idx]
        target = immediate_cost + best_future_cost

        # c(x,a) = n/(n+1) * c(x,a) + 1/(n+1) * [c_im + best_future]
        # Eq. from Section 3 of paper
        if n == 0:
            # First visit: just set to target
            self.costs[state_idx, action_idx] = target
        else:
            self.costs[state_idx, action_idx] = (
                (n / (n + 1)) * current + (1 / (n + 1)) * target
            )
        self.visit_counts[state_idx, action_idx] = n + 1

    def best_action(self, state_idx: int) -> int:
        """Return the action with minimum expected cost in this state.

        Args:
            state_idx: Flat state index.

        Returns:
            0-based action index with minimum cost.
        """
        assert 0 <= state_idx < self.n_states
        return int(np.argmin(self.costs[state_idx]))

    def best_cost(self, state_idx: int) -> float:
        """Return the minimum expected cost achievable from this state.

        Args:
            state_idx: Flat state index.

        Returns:
            Minimum cost in basis points.
        """
        assert 0 <= state_idx < self.n_states
        return float(np.min(self.costs[state_idx]))

    def coverage(self) -> float:
        """Fraction of (state, action) pairs that have been visited at least once."""
        visited = np.sum(self.visit_counts > 0)
        return float(visited) / (self.n_states * self.n_actions)

    def save(self, path: str) -> None:
        """Save Q-table to disk.

        Args:
            path: File path (will use pickle format).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "costs": self.costs,
                "visit_counts": self.visit_counts,
                "n_states": self.n_states,
                "n_actions": self.n_actions,
            }, f)

    @classmethod
    def load(cls, path: str) -> "QTable":
        """Load a saved Q-table from disk.

        Args:
            path: File path previously saved with QTable.save().

        Returns:
            Loaded QTable instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        qt = cls(data["n_states"], data["n_actions"])
        qt.costs = data["costs"]
        qt.visit_counts = data["visit_counts"]
        return qt

    def __repr__(self) -> str:
        return (
            f"QTable(n_states={self.n_states}, n_actions={self.n_actions}, "
            f"coverage={self.coverage():.2%})"
        )
