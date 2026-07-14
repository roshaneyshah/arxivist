"""
env/market_env.py — Trade execution environment with state encoding.

Implements the state/action/reward formulation from Section 3 of:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

State: x = <t, i, o1, ..., oR>
  t = elapsed time (decision point index, 0..T)
  i = remaining inventory units (0..I)
  o1..oR = discretized market variables

Action: a = limit price offset relative to ask (sell) or bid (buy)
  a > 0: cross spread toward opposing book (aggressive)
  a < 0: place deeper in own book (passive)

Reward: immediate execution proceeds; terminal forced market order at end of horizon H.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from rl_trade_execution.env.market_features import MarketFeatureExtractor
from rl_trade_execution.env.order_book import OrderBookSnapshot
from rl_trade_execution.utils.config import ExperimentConfig


class TradeExecutionEnv:
    """Tabular trade execution environment.

    Manages a single execution episode: sell (or buy) V shares within
    H time steps while minimizing trading cost relative to the mid-spread.

    Paper reference: Section 3 — "An RL Formulation of Optimized Execution"

    Attributes:
        config: Experiment configuration.
        feature_extractor: Computes and discretizes market variables.
        episode_snapshots: Sequence of order book snapshots for current episode.
        t: Current decision point (0 = start, T = end).
        i: Remaining inventory in units of V/I shares.
        episode_start_mid: Mid-spread at episode start (for cost calculation).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        feature_extractor: MarketFeatureExtractor,
    ):
        """
        Args:
            config: Experiment configuration with T, I, L, V, H_minutes.
            feature_extractor: Fitted market feature extractor.
        """
        self.config = config
        self.feature_extractor = feature_extractor

        # Compute state space dimensions
        self.market_dims = feature_extractor.state_dims  # e.g. [3, 3] for 2 vars
        self._state_strides = self._compute_strides()
        self._total_states = (config.T + 1) * (config.I + 1) * int(np.prod(self.market_dims))

        # Episode state
        self.episode_snapshots: List[OrderBookSnapshot] = []
        self.t: int = 0
        self.i: int = config.I  # start with full inventory
        self.episode_start_mid: float = 0.0
        self._snap_idx: int = 0  # current position in episode_snapshots

    def _compute_strides(self) -> List[int]:
        """Compute strides for flat state index encoding.

        State index = t * stride_t + i * stride_i + o1 * stride_o1 + ... + oR * stride_oR
        """
        dims = [self.config.T + 1, self.config.I + 1] + self.market_dims
        strides = []
        stride = 1
        for d in reversed(dims):
            strides.insert(0, stride)
            stride *= d
        return strides

    def reset(self, snapshots: List[OrderBookSnapshot]) -> int:
        """Start a new episode.

        Args:
            snapshots: Ordered sequence of order book snapshots covering horizon H.
                       Length should be >= T decision points.

        Returns:
            Initial state index.
        """
        assert len(snapshots) >= self.config.T, (
            f"Need at least T={self.config.T} snapshots, got {len(snapshots)}"
        )
        self.episode_snapshots = snapshots
        self.t = 0
        self.i = self.config.I
        self.episode_start_mid = snapshots[0].mid()
        self._snap_idx = 0
        return self.encode_state(self.t, self.i, self._current_market_vars())

    def step(self, action_idx: int) -> Tuple[int, float, bool, Dict]:
        """Execute one decision step.

        Submits a limit order at (ask - action) for remaining shares,
        simulates execution against current order book, advances time.

        Paper reference: Section 3 — Actions and Rewards.

        Args:
            action_idx: 0-based index into action space.

        Returns:
            Tuple of (next_state, immediate_cost_bps, done, info).
        """
        assert 0 <= action_idx < self.config.L, (
            f"action_idx {action_idx} out of range [0, {self.config.L})"
        )
        snap = self.episode_snapshots[self._snap_idx]
        action = self.config.index_to_action(action_idx)

        # Compute limit price: ask - a (sell side)
        # Paper Section 3 Actions: "action a corresponds to placing a limit order
        # at price ask - a"
        if self.config.side == "sell":
            limit_price = snap.ask() - action
            shares_to_execute = self._inventory_in_shares()
            filled, avg_price, remaining_shares = snap.simulate_sell_execution(
                shares_to_execute, limit_price
            )
        else:
            limit_price = snap.bid() + action
            shares_to_execute = self._inventory_in_shares()
            filled, avg_price, remaining_shares = snap.simulate_buy_execution(
                shares_to_execute, limit_price
            )

        # Convert filled shares back to inventory units
        filled_units = min(self.i, filled // self._shares_per_unit())
        self.i = max(0, self.i - filled_units)

        # Immediate cost for executed shares (in basis points)
        # Paper Section 3 Rewards: "immediate rewards = proceeds from partial execution"
        immediate_cost_bps = self._compute_cost_bps(avg_price, filled)

        # Advance time
        self.t += 1
        self._snap_idx = min(self._snap_idx + 1, len(self.episode_snapshots) - 1)

        done = self.t >= self.config.T

        # If last step and inventory remains, forced market order
        terminal_cost_bps = 0.0
        if done and self.i > 0:
            terminal_snap = self.episode_snapshots[self._snap_idx]
            forced_shares = self._inventory_in_shares()
            if self.config.side == "sell":
                filled_t, avg_t, _ = terminal_snap.simulate_sell_execution(forced_shares, 0.0)
            else:
                filled_t, avg_t, _ = terminal_snap.simulate_buy_execution(forced_shares, float("inf"))
            terminal_cost_bps = self._compute_cost_bps(avg_t, filled_t)
            self.i = 0

        total_step_cost = immediate_cost_bps + terminal_cost_bps

        next_state = self.encode_state(self.t, self.i, self._current_market_vars())

        info = {
            "action": action,
            "limit_price": limit_price,
            "filled_shares": filled,
            "remaining_shares": remaining_shares,
            "immediate_cost_bps": immediate_cost_bps,
            "terminal_cost_bps": terminal_cost_bps,
        }

        return next_state, total_step_cost, done, info

    def encode_state(self, t: int, i: int, market_vars: List[int]) -> int:
        """Encode (t, i, o1..oR) into a flat integer state index.

        Paper reference: Section 3 — States: "state x = <t, i, o1, ..., oR>"
        """
        components = [t, i] + market_vars
        assert len(components) == len(self._state_strides), (
            f"State components {len(components)} != strides {len(self._state_strides)}"
        )
        return sum(c * s for c, s in zip(components, self._state_strides))

    def decode_state(self, state_idx: int) -> Tuple[int, int, List[int]]:
        """Decode a flat state index back to (t, i, market_vars)."""
        dims = [self.config.T + 1, self.config.I + 1] + self.market_dims
        components = []
        remaining = state_idx
        for stride, dim in zip(self._state_strides, dims):
            components.append(remaining // stride)
            remaining = remaining % stride
        t, i = components[0], components[1]
        market_vars = components[2:]
        return t, i, market_vars

    def _current_market_vars(self) -> List[int]:
        """Extract market features from current order book snapshot."""
        if not self.episode_snapshots or self._snap_idx >= len(self.episode_snapshots):
            return [0] * len(self.feature_extractor.feature_names)
        snap = self.episode_snapshots[self._snap_idx]
        volume = self._inventory_in_shares()
        return self.feature_extractor.extract(snap, volume, self.config.side)

    def _inventory_in_shares(self) -> int:
        """Convert inventory units to actual shares."""
        return self.i * self._shares_per_unit()

    def _shares_per_unit(self) -> int:
        """Shares per inventory unit: V / I."""
        return max(1, self.config.V // self.config.I)

    def _compute_cost_bps(self, avg_price: float, shares: int) -> float:
        """Compute trading cost in basis points relative to episode-start mid.

        Paper reference: Section 3 Rewards:
        "trading cost = underperformance compared to mid-spread baseline,
        measured in basis points (1/100 of a percent)"

        Cost formula: (mid - execution_price) / mid * 10000  [for selling]
        """
        if shares == 0 or self.episode_start_mid == 0:
            return 0.0
        if self.config.side == "sell":
            return (self.episode_start_mid - avg_price) / self.episode_start_mid * 10000.0
        else:
            return (avg_price - self.episode_start_mid) / self.episode_start_mid * 10000.0

    @property
    def total_states(self) -> int:
        """Total number of states in the discretized state space."""
        return self._total_states

    def __repr__(self) -> str:
        return (
            f"TradeExecutionEnv(T={self.config.T}, I={self.config.I}, "
            f"L={self.config.L}, market_dims={self.market_dims}, "
            f"total_states={self.total_states})"
        )
