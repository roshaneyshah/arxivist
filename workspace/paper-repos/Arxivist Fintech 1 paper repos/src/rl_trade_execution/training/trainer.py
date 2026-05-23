"""
training/trainer.py — Backward induction trainer implementing the paper's core algorithm.

Implements the Optimal_strategy() algorithm from Section 3 of:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

Algorithm pseudocode (from paper):
  Optimal_strategy(V, H, T, I, L):
    For t = T to 0:
      While (not end of data):
        Transform(order_book) → o1..oR
        For i = 0 to I:
          For a = 0 to L:
            Set x = {t, i, o1..oR}
            Simulate transition x → y
            Calculate c_im(x, a)
            Look up argmax c(y, p)
            Update c(<t, v, o1..oR>, a)
        Select highest-payout action argmax c(y,p) in every state y → optimal policy

Key insight: approximate independence of private (t, i) and market variables means
we only need T*I*L passes over the data. Running time is independent of R.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # type: ignore[misc]
        return x

from rl_trade_execution.agent.policy import OptimalPolicy
from rl_trade_execution.agent.q_table import QTable
from rl_trade_execution.env.market_env import TradeExecutionEnv
from rl_trade_execution.env.order_book import OrderBookSnapshot
from rl_trade_execution.utils.config import ExperimentConfig


class BackwardInductionTrainer:
    """Trains an optimal trade execution policy via backward induction.

    The key algorithmic trick (Section 3): because market variables evolve
    independently of our actions (during training), we can try every possible
    (t, i) combination for each observed order book state. This makes computation
    time O(T * I * L) passes over data, independent of the number of market vars R.

    Paper reference: Section 3, Algorithm, and the independence assumption:
      "we will assume that our actions do not affect market state variables o1..oR"

    Attributes:
        config: Experiment configuration.
        env: Trade execution environment.
        q_table: Q-table being trained.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        env: TradeExecutionEnv,
        q_table: Optional[QTable] = None,
    ):
        """
        Args:
            config: Experiment configuration.
            env: Initialized trade execution environment with fitted feature extractor.
            q_table: Optional pre-existing Q-table to continue training from.
        """
        self.config = config
        self.env = env

        if q_table is None:
            self.q_table = QTable(env.total_states, config.L)
        else:
            self.q_table = q_table

        self._step = 0
        self._start_time = 0.0

    def train(
        self,
        all_episodes: List[List[OrderBookSnapshot]],
        log_every: int = 1000,
        checkpoint_dir: Optional[str] = None,
    ) -> QTable:
        """Run the full backward induction training loop.

        Iterates t from T down to 0. For each snapshot in training data,
        tries all combinations of remaining inventory i and action a.

        Paper reference: Section 3, Optimal_strategy pseudocode.

        Complexity: O(T * I * L) passes over data.
        Runtime is INDEPENDENT of the number of market variables R.

        Args:
            all_episodes: List of episodes, each a list of OrderBookSnapshots.
            log_every: Print progress every N updates.
            checkpoint_dir: If set, save Q-table to this directory periodically.

        Returns:
            Trained QTable.
        """
        self._start_time = time.time()
        T = self.config.T
        I = self.config.I
        L = self.config.L
        total_snapshots = sum(len(ep) for ep in all_episodes)

        print(f"\nBackward Induction Training")
        print(f"  Stock: {self.config.stock}, V={self.config.V}, H={self.config.H_minutes}min")
        print(f"  T={T}, I={I}, L={L}, R={len(self.config.market_variables)}")
        print(f"  Total states: {self.env.total_states:,}")
        print(f"  Training episodes: {len(all_episodes):,} ({total_snapshots:,} snapshots)")
        print(f"  Data passes required: T*I*L = {T}*{I}*{L} = {T*I*L:,}\n")

        # Outer loop: backward in time (Section 3: "For t = T to 0")
        for t in range(T, -1, -1):
            print(f"[t={t}/{T}] Processing all snapshots...")
            t_updates = 0

            for ep_idx, episode_snaps in enumerate(tqdm(all_episodes, desc=f"t={t}", leave=False)):
                for snap_idx, snap in enumerate(episode_snaps):
                    # Extract market features for this snapshot
                    # Independence assumption: we try ALL values of i for each snapshot
                    # Paper Section 3: "we try every possible value of i ∈ [0, V/I]"
                    market_vars = self.env.feature_extractor.extract(
                        snap,
                        self.config.V,  # use max volume for feature extraction during training
                        self.config.side,
                    )

                    # Inner loop over all inventory levels
                    for i in range(I + 1):
                        state_idx = self.env.encode_state(t, i, market_vars)

                        # Inner loop over all actions
                        for action_idx in range(L):
                            action = self.config.index_to_action(action_idx)

                            # Compute immediate cost: simulate execution
                            immediate_cost, next_i = self._simulate_step(
                                snap, i, action
                            )

                            # Look up best future cost from already-optimized next state
                            # (t+1 has already been processed since we go backward)
                            if t < T:
                                # Advance to next snapshot for next state market vars
                                next_snap_idx = min(snap_idx + 1, len(episode_snaps) - 1)
                                next_snap = episode_snaps[next_snap_idx]
                                next_market_vars = self.env.feature_extractor.extract(
                                    next_snap,
                                    max(0, next_i) * (self.config.V // self.config.I),
                                    self.config.side,
                                )
                                next_state_idx = self.env.encode_state(
                                    t + 1, next_i, next_market_vars
                                )
                                best_future = self.q_table.best_cost(next_state_idx)
                            else:
                                # t == T: forced market order terminal cost
                                # Paper: "when time runs out, forced market order for remaining shares"
                                best_future = self._terminal_cost(snap, i)

                            # Q-table update
                            # Paper Eq: c(x,a) = n/(n+1)*c(x,a) + 1/(n+1)*[c_im(x,a) + argmax c(y,p)]
                            self.q_table.update(state_idx, action_idx, immediate_cost, best_future)
                            t_updates += 1
                            self._step += 1

                            if self._step % log_every == 0:
                                elapsed = time.time() - self._start_time
                                print(
                                    f"  Step {self._step:,} | t={t} | "
                                    f"coverage={self.q_table.coverage():.1%} | "
                                    f"elapsed={elapsed:.0f}s"
                                )

            print(f"  t={t}: {t_updates:,} updates, coverage={self.q_table.coverage():.2%}")

            if checkpoint_dir and t % max(1, T // 4) == 0:
                path = os.path.join(checkpoint_dir, f"q_table_t{t}.pkl")
                self.q_table.save(path)
                print(f"  Checkpoint saved: {path}")

        elapsed = time.time() - self._start_time
        print(f"\nTraining complete in {elapsed:.1f}s | Q-table coverage: {self.q_table.coverage():.2%}")
        return self.q_table

    def _simulate_step(
        self, snap: OrderBookSnapshot, i_units: int, action: int
    ) -> Tuple[float, int]:
        """Simulate one execution step and return (immediate_cost_bps, next_i_units).

        Args:
            snap: Current order book snapshot.
            i_units: Current inventory in units (0..I).
            action: Raw price offset (negative=passive, positive=aggressive).

        Returns:
            Tuple of (immediate_cost_bps, next_inventory_units).
        """
        shares = i_units * (self.config.V // max(1, self.config.I))
        if shares == 0:
            return 0.0, 0

        mid = self.env.episode_start_mid if self.env.episode_start_mid > 0 else snap.mid()

        if self.config.side == "sell":
            limit_price = snap.ask() - action
            filled, avg_price, _ = snap.simulate_sell_execution(shares, limit_price)
            if filled > 0 and mid > 0:
                cost_bps = (mid - avg_price) / mid * 10000.0
            else:
                cost_bps = 0.0
        else:
            limit_price = snap.bid() + action
            filled, avg_price, _ = snap.simulate_buy_execution(shares, limit_price)
            if filled > 0 and mid > 0:
                cost_bps = (avg_price - mid) / mid * 10000.0
            else:
                cost_bps = 0.0

        filled_units = min(i_units, filled // max(1, self.config.V // self.config.I))
        next_i = max(0, i_units - filled_units)
        return cost_bps, next_i

    def _terminal_cost(self, snap: OrderBookSnapshot, i_units: int) -> float:
        """Cost of forced terminal market order for remaining inventory.

        Paper Section 3: "any inventory remaining at the end of time H is immediately
        executed at market prices — we eat into the opposing book no matter how poor the prices."

        Args:
            snap: Terminal order book snapshot.
            i_units: Remaining inventory units.

        Returns:
            Expected cost in basis points.
        """
        shares = i_units * (self.config.V // max(1, self.config.I))
        if shares == 0:
            return 0.0
        return snap.market_order_cost_bps(shares, self.config.side)
