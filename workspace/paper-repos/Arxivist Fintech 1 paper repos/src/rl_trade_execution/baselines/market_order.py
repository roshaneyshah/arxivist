"""
baselines/market_order.py — Immediate market order baseline.

Paper reference: Section 2 — "submit a market order immediately at the beginning
of the time interval — effectively a limit order at price 0 which takes the V
best prices in the buy book at that moment."
"""

from __future__ import annotations

from typing import List

import numpy as np

from rl_trade_execution.env.order_book import OrderBookSnapshot
from rl_trade_execution.utils.config import ExperimentConfig


class MarketOrderPolicy:
    """Baseline that immediately executes everything as a market order.

    This represents the worst-case baseline in the paper — it has maximum
    market impact and no price improvement. RL outperforms this by several
    times when market variables are included.

    Paper reference: Section 2 — immediate full market order is the naive baseline.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def act(self, state_idx: int) -> int:
        """Return most aggressive action (maximum price crossing).

        Args:
            state_idx: Ignored.

        Returns:
            Action index corresponding to action_max (most aggressive).
        """
        return self.config.L - 1  # most aggressive action

    def evaluate_episodes(
        self,
        episodes: List[List[OrderBookSnapshot]],
    ) -> float:
        """Evaluate immediate market order cost on test episodes."""
        costs = []
        for snaps in episodes:
            snap = snaps[0]
            mid_start = snap.mid()
            if mid_start == 0:
                continue
            cost = snap.market_order_cost_bps(self.config.V, self.config.side)
            costs.append(cost)
        return float(np.mean(costs)) if costs else 0.0

    def __repr__(self) -> str:
        return f"MarketOrderPolicy(stock={self.config.stock})"
