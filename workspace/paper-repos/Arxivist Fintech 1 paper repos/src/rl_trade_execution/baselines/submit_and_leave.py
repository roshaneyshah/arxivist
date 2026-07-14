"""
baselines/submit_and_leave.py — Submit-and-Leave (S&L) baseline strategy.

Implements the S&L baseline described in Section 2 of:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

"Submit and leave policies: pick a fixed limit order price p and place a sell order
for all V shares at price p. After H minutes, go to market with any remaining shares."

This is the primary comparison baseline. RL outperforms it by 27-50%+ in the paper.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from rl_trade_execution.env.order_book import OrderBookSnapshot
from rl_trade_execution.utils.config import ExperimentConfig


class SubmitAndLeavePolicy:
    """Fixed limit-price execution policy.

    Places all shares at a fixed offset from the ask/bid and waits.
    Any remaining shares at horizon end are executed as a market order.

    Paper reference: Section 2 — "submit and leave (S&L) policies"
    Table 1 in paper uses optimized S&L as the primary comparison baseline.

    Attributes:
        config: Experiment configuration.
        limit_offset: Fixed price offset (same semantics as RL action: ask - offset).
    """

    def __init__(self, config: ExperimentConfig, limit_offset: int = 0):
        """
        Args:
            config: Experiment configuration.
            limit_offset: Fixed price offset from ask (sell) or bid (buy).
                          0 = at-ask/at-bid; positive = more aggressive.
        """
        self.config = config
        self.limit_offset = limit_offset

    def act(self, state_idx: int) -> int:
        """Return fixed action regardless of state.

        Args:
            state_idx: Ignored — S&L is state-independent.

        Returns:
            Fixed action index corresponding to self.limit_offset.
        """
        return self.config.action_index(self.limit_offset)

    def evaluate_episodes(
        self,
        episodes: List[List[OrderBookSnapshot]],
    ) -> float:
        """Evaluate this S&L policy on a set of test episodes.

        Args:
            episodes: List of episodes (each a list of snapshots).

        Returns:
            Mean trading cost in basis points.
        """
        costs = []
        for snaps in episodes:
            cost = self._run_episode(snaps)
            costs.append(cost)
        return float(np.mean(costs))

    def _run_episode(self, snaps: List[OrderBookSnapshot]) -> float:
        """Simulate one S&L episode.

        Args:
            snaps: Ordered snapshots for this episode.

        Returns:
            Total trading cost in basis points.
        """
        mid_start = snaps[0].mid()
        if mid_start == 0:
            return 0.0

        remaining = self.config.V
        total_cost = 0.0
        snap = snaps[0]  # S&L submits at the start and doesn't monitor

        if self.config.side == "sell":
            limit_price = snap.ask() - self.limit_offset
            filled, avg_price, remaining_after = snap.simulate_sell_execution(remaining, limit_price)
        else:
            limit_price = snap.bid() + self.limit_offset
            filled, avg_price, remaining_after = snap.simulate_buy_execution(remaining, limit_price)

        if filled > 0:
            if self.config.side == "sell":
                total_cost += (mid_start - avg_price) / mid_start * 10000.0 * (filled / self.config.V)
            else:
                total_cost += (avg_price - mid_start) / mid_start * 10000.0 * (filled / self.config.V)

        # Terminal: forced market order for remaining shares
        if remaining_after > 0:
            terminal_snap = snaps[-1]
            terminal_cost = terminal_snap.market_order_cost_bps(remaining_after, self.config.side)
            total_cost += terminal_cost * (remaining_after / self.config.V)

        return total_cost

    @classmethod
    def optimize(
        cls,
        config: ExperimentConfig,
        train_episodes: List[List[OrderBookSnapshot]],
    ) -> "SubmitAndLeavePolicy":
        """Find the best fixed limit offset by grid search over training episodes.

        Paper reference: Section 5 — "the optimized submit-and-leave strategy"
        is used as the primary comparison baseline against RL.

        Args:
            config: Experiment configuration.
            train_episodes: Training episodes to optimize over.

        Returns:
            S&L policy with the best (lowest-cost) fixed offset.
        """
        best_offset = 0
        best_cost = float("inf")

        for offset_idx in range(config.L):
            offset = config.index_to_action(offset_idx)
            policy = cls(config, offset)
            mean_cost = policy.evaluate_episodes(train_episodes)
            if mean_cost < best_cost:
                best_cost = mean_cost
                best_offset = offset

        print(f"Optimized S&L: best offset={best_offset}, train cost={best_cost:.2f} bps")
        return cls(config, best_offset)

    def __repr__(self) -> str:
        return f"SubmitAndLeavePolicy(offset={self.limit_offset}, stock={self.config.stock})"
