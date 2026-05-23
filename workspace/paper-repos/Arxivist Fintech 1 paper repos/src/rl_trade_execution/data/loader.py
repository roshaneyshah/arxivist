"""
data/loader.py — Data loading and episode partitioning for INET ECN order book data.

The paper uses 1.5 years of millisecond-scale INET ECN data for 3 NASDAQ stocks:
  AMZN, NVDA, QCOM
  12 months training / 6 months test

IMPORTANT: The INET ECN historical data used in the paper is PROPRIETARY and not
publicly available. This loader provides:
  1. An interface for loading order book data from CSV files in standard format.
  2. A SyntheticOrderBookGenerator for testing and development.

See data/README_data.md for instructions on obtaining real order book data.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from rl_trade_execution.env.order_book import OrderBookSnapshot, PriceLevel


@dataclass
class Episode:
    """A single execution episode: H minutes of order book snapshots."""
    episode_id: int
    stock: str
    snapshots: List[OrderBookSnapshot]
    start_timestamp: int


class INETDataLoader:
    """Loads INET ECN order book data and partitions into episodes.

    Expected CSV format (one row per order book update):
      timestamp_ms, bid_p1, bid_v1, bid_p2, bid_v2, ..., ask_p1, ask_v1, ...,
      signed_volume_15s

    Paper reference: Section 2 — "1.5 years of very high-frequency (millisecond)
    microstructure data for AMZN, NVDA, QCOM from INET ECN"
    """

    SUPPORTED_STOCKS = ["AMZN", "NVDA", "QCOM"]

    def __init__(self, data_dir: str, n_book_levels: int = 10):
        """
        Args:
            data_dir: Root directory containing per-stock subdirectories.
            n_book_levels: Number of price levels to load per side.
        """
        self.data_dir = data_dir
        self.n_book_levels = n_book_levels

    def load_order_book(self, filepath: str) -> List[OrderBookSnapshot]:
        """Load order book snapshots from a CSV file.

        Args:
            filepath: Path to the CSV file.

        Returns:
            Ordered list of OrderBookSnapshot objects.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Order book file not found: {filepath}\n"
                f"See data/README_data.md for instructions on obtaining INET data."
            )

        df = pd.read_csv(filepath)
        snapshots = []

        for _, row in df.iterrows():
            bids = []
            asks = []

            for k in range(1, self.n_book_levels + 1):
                bp_col, bv_col = f"bid_p{k}", f"bid_v{k}"
                ap_col, av_col = f"ask_p{k}", f"ask_v{k}"

                if bp_col in row and bv_col in row and pd.notna(row[bp_col]):
                    bids.append(PriceLevel(float(row[bp_col]), int(row[bv_col])))
                if ap_col in row and av_col in row and pd.notna(row[ap_col]):
                    asks.append(PriceLevel(float(row[ap_col]), int(row[av_col])))

            signed_vol = int(row.get("signed_volume_15s", 0))

            snapshots.append(OrderBookSnapshot(
                timestamp=int(row["timestamp_ms"]),
                bids=bids,
                asks=asks,
                recent_trade_volume=signed_vol,
            ))

        return snapshots

    def partition_episodes(
        self,
        snapshots: List[OrderBookSnapshot],
        H_minutes: int,
        T: int,
        stock: str = "UNKNOWN",
    ) -> List[Episode]:
        """Partition a sequence of snapshots into non-overlapping episodes of length H.

        Each episode contains exactly T+1 decision-point snapshots evenly spaced
        over the H-minute window.

        Paper reference: Section 5 — "we partitioned our 1-year INET training data
        into approximately 45,000 episodes" (for H=2min)

        Args:
            snapshots: Full ordered sequence of order book snapshots.
            H_minutes: Episode horizon in minutes.
            T: Number of decision points per episode.
            stock: Stock ticker (for labeling).

        Returns:
            List of Episode objects.
        """
        H_ms = H_minutes * 60 * 1000  # H in milliseconds
        episodes = []
        ep_id = 0
        i = 0

        while i < len(snapshots):
            ep_start_ts = snapshots[i].timestamp
            ep_end_ts = ep_start_ts + H_ms

            # Collect all snapshots in this window
            window = []
            j = i
            while j < len(snapshots) and snapshots[j].timestamp < ep_end_ts:
                window.append(snapshots[j])
                j += 1

            if len(window) >= T:
                # Sample T+1 evenly-spaced snapshots from the window
                indices = np.linspace(0, len(window) - 1, T + 1, dtype=int)
                ep_snaps = [window[k] for k in indices]

                episodes.append(Episode(
                    episode_id=ep_id,
                    stock=stock,
                    snapshots=ep_snaps,
                    start_timestamp=ep_start_ts,
                ))
                ep_id += 1

            i = j if j > i else i + 1

        return episodes

    def train_test_split(
        self,
        episodes: List[Episode],
        train_months: int = 12,
        test_months: int = 6,
    ) -> Tuple[List[Episode], List[Episode]]:
        """Split episodes chronologically into train and test sets.

        Paper: 12 months training, 6 months test.

        Args:
            episodes: All episodes sorted by timestamp.
            train_months: Number of months for training.
            test_months: Number of months for testing.

        Returns:
            Tuple of (train_episodes, test_episodes).
        """
        total = train_months + test_months
        split_idx = int(len(episodes) * train_months / total)
        return episodes[:split_idx], episodes[split_idx:]


class SyntheticOrderBookGenerator:
    """Generate synthetic order book data for testing and development.

    Uses a simple random walk for mid-price with Poisson-distributed order arrivals.
    This is NOT a realistic market simulator — it exists purely to enable running
    the code without proprietary INET data.

    IMPORTANT: Results from synthetic data will NOT match the paper's results.
    The paper uses real INET ECN data which is not publicly available.
    """

    def __init__(
        self,
        seed: int = 42,
        initial_price: float = 50.0,
        spread_bps: float = 10.0,
        volatility: float = 0.0001,
        n_levels: int = 5,
    ):
        """
        Args:
            seed: Random seed.
            initial_price: Starting mid-price.
            spread_bps: Bid-ask spread in basis points.
            volatility: Per-tick price volatility (std dev as fraction of price).
            n_levels: Number of order book levels to generate.
        """
        self.rng = np.random.default_rng(seed)
        self.initial_price = initial_price
        self.spread_bps = spread_bps
        self.volatility = volatility
        self.n_levels = n_levels

    def generate_episodes(
        self, n_episodes: int, T: int, stock: str = "SYN"
    ) -> List[Episode]:
        """Generate synthetic episodes.

        Args:
            n_episodes: Number of episodes to generate.
            T: Decision points per episode.
            stock: Synthetic stock ticker.

        Returns:
            List of Episode objects with synthetic data.
        """
        episodes = []
        for ep_id in range(n_episodes):
            snaps = self._generate_episode_snapshots(T + 1, ep_id * (T + 2) * 1000)
            episodes.append(Episode(
                episode_id=ep_id,
                stock=stock,
                snapshots=snaps,
                start_timestamp=ep_id * (T + 2) * 1000,
            ))
        return episodes

    def _generate_episode_snapshots(
        self, n_snaps: int, base_timestamp: int
    ) -> List[OrderBookSnapshot]:
        """Generate one episode's worth of snapshots."""
        price = self.initial_price
        snaps = []
        half_spread = price * self.spread_bps / 20000.0  # spread/2 in price units

        for k in range(n_snaps):
            # Random walk for mid price
            price *= (1 + self.rng.normal(0, self.volatility))
            price = max(1.0, price)
            half_spread = price * self.spread_bps / 20000.0
            tick = price * 0.01  # 1 cent tick

            bid_base = price - half_spread
            ask_base = price + half_spread

            bids = [
                PriceLevel(
                    price=round(bid_base - j * tick, 4),
                    volume=int(self.rng.integers(100, 2000))
                )
                for j in range(self.n_levels)
            ]
            asks = [
                PriceLevel(
                    price=round(ask_base + j * tick, 4),
                    volume=int(self.rng.integers(100, 2000))
                )
                for j in range(self.n_levels)
            ]

            signed_vol = int(self.rng.integers(-5000, 5000))

            snaps.append(OrderBookSnapshot(
                timestamp=base_timestamp + k * 1000,
                bids=bids,
                asks=asks,
                recent_trade_volume=signed_vol,
            ))

        return snaps
