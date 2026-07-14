"""
evaluation/metrics.py — Execution quality metrics.

Implements the evaluation methodology from Section 3 and Section 5 of:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

Primary metric: trading cost in basis points (bps) relative to mid-spread at episode start.
  cost_bps = (mid_start - avg_execution_price) / mid_start * 10000  [for selling]

Paper: "we always measure execution prices relative to the mid-spread price at the start
of the episode. Trading cost = underperformance compared to mid-spread baseline, in bps."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EpisodeResult:
    """Results for a single execution episode."""
    episode_id: int
    stock: str
    V: int
    H_minutes: int
    policy_name: str
    trading_cost_bps: float
    shares_executed: int
    avg_execution_price: float
    mid_start: float


class ExecutionMetrics:
    """Compute and aggregate execution quality metrics.

    Paper reference: Section 3 Rewards, Section 4 Experimental Results.
    """

    @staticmethod
    def trading_cost_bps(
        avg_execution_price: float, mid_start: float, side: str = "sell"
    ) -> float:
        """Compute trading cost in basis points relative to episode-start mid.

        Paper reference: Section 3 — "trading cost = underperformance vs mid-spread,
        measured in basis points (1/100 of a percent)"

        For selling: cost = (mid - execution) / mid * 10000
        For buying:  cost = (execution - mid) / mid * 10000

        Args:
            avg_execution_price: Volume-weighted average price achieved.
            mid_start: Mid-spread price at episode start.
            side: "sell" or "buy".

        Returns:
            Trading cost in basis points (positive = worse than mid).
        """
        if mid_start == 0:
            return 0.0
        if side == "sell":
            return (mid_start - avg_execution_price) / mid_start * 10000.0
        else:
            return (avg_execution_price - mid_start) / mid_start * 10000.0

    @staticmethod
    def relative_improvement(cost_a: float, cost_b: float) -> float:
        """Relative improvement of policy A over baseline B.

        Paper reports improvement as: (cost_baseline - cost_rl) / cost_baseline

        Args:
            cost_a: Cost of the improved policy (RL).
            cost_b: Cost of the baseline (S&L).

        Returns:
            Relative improvement as a fraction (e.g., 0.35 = 35% improvement).
        """
        if cost_b == 0:
            return 0.0
        return (cost_b - cost_a) / abs(cost_b)

    @staticmethod
    def aggregate_episodes(episode_costs: List[float]) -> Dict[str, float]:
        """Compute aggregate statistics over multiple episodes.

        Args:
            episode_costs: List of per-episode trading costs in bps.

        Returns:
            Dict with mean, std, median, min, max, and count.
        """
        arr = np.array(episode_costs, dtype=float)
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr),
        }

    @staticmethod
    def compare_policies(
        rl_costs: List[float],
        baseline_costs: List[float],
        policy_name: str = "RL",
        baseline_name: str = "S&L",
    ) -> Dict[str, object]:
        """Compare two policies and report metrics matching the paper's Table 1 format.

        Args:
            rl_costs: Per-episode costs for RL policy.
            baseline_costs: Per-episode costs for baseline policy.
            policy_name: Name of improved policy.
            baseline_name: Name of baseline.

        Returns:
            Dict with mean costs, improvement, and summary string.
        """
        rl_mean = float(np.mean(rl_costs))
        base_mean = float(np.mean(baseline_costs))
        improvement = ExecutionMetrics.relative_improvement(rl_mean, base_mean)

        return {
            f"{policy_name}_mean_bps": rl_mean,
            f"{baseline_name}_mean_bps": base_mean,
            "relative_improvement": improvement,
            "absolute_improvement_bps": base_mean - rl_mean,
            "summary": (
                f"{policy_name}: {rl_mean:.2f} bps | "
                f"{baseline_name}: {base_mean:.2f} bps | "
                f"Improvement: {improvement:.1%}"
            ),
        }
