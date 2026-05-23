"""
env/market_features.py — Market variable extraction and discretization.

Implements the market variables described in Section 4.2 of:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

Market variables convert raw order book data into low-resolution discrete features
that are included in the RL state vector. The paper describes these as "blurry
representations of the real world."

Variables implemented:
  - bid_ask_spread: current spread size (Section 4.2)
  - immediate_market_order_cost: cost to submit full market order now (Section 4.2)
  - bid_ask_volume_imbalance: signed volume difference at bid vs ask (Section 4.2)
  - signed_transaction_volume: net buy/sell volume over last 15 seconds (Section 4.2)
"""

from __future__ import annotations

from typing import List

import numpy as np

from rl_trade_execution.env.order_book import OrderBookSnapshot


class MarketFeatureExtractor:
    """Extracts and discretizes market features from order book snapshots.

    Each feature is mapped to an integer bin in {0, 1, ..., n_bins-1}.
    Bin boundaries are computed from training data statistics (or set to
    equal-width intervals as a default).

    Paper reference: Section 3 — "Market variables summarized a variety of information
    from the order books into a number of low-resolution features."

    Attributes:
        n_bins: Number of discrete bins per feature (default 3: low/medium/high).
            ASSUMED: paper explicitly states 3 for transaction volume; implied for others.
            Confidence: 0.80
        feature_names: Ordered list of enabled feature names.
        bin_edges: Dict mapping feature name to array of bin edge values.
    """

    SUPPORTED_FEATURES = [
        "bid_ask_spread",
        "immediate_market_order_cost",
        "bid_ask_volume_imbalance",
        "signed_transaction_volume",
    ]

    def __init__(self, feature_names: List[str], n_bins: int = 3):
        """
        Args:
            feature_names: List of market variable names to compute.
            n_bins: Discretization bins per feature. ASSUMED: 3 (confidence 0.80).
        """
        for name in feature_names:
            if name not in self.SUPPORTED_FEATURES:
                raise ValueError(
                    f"Unknown market feature '{name}'. "
                    f"Supported: {self.SUPPORTED_FEATURES}"
                )
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.bin_edges: dict = {}

    def fit(self, snapshots: List[OrderBookSnapshot], volume: int, side: str = "sell") -> None:
        """Compute bin edges from a set of snapshots (training data).

        Uses percentile-based binning to create roughly equal-frequency bins.

        Args:
            snapshots: List of order book snapshots from training data.
            volume: Trade volume used for market order cost calculation.
            side: "sell" or "buy".
        """
        raw_values: dict = {name: [] for name in self.feature_names}

        for snap in snapshots:
            raw = self._extract_raw(snap, volume, side)
            for name, val in zip(self.feature_names, raw):
                raw_values[name].append(val)

        percentiles = np.linspace(0, 100, self.n_bins + 1)
        for name, vals in raw_values.items():
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                self.bin_edges[name] = np.zeros(self.n_bins - 1)
            else:
                edges = np.percentile(arr, percentiles[1:-1])
                self.bin_edges[name] = edges

    def extract(
        self, snapshot: OrderBookSnapshot, volume: int, side: str = "sell"
    ) -> List[int]:
        """Extract discretized market features from a snapshot.

        Args:
            snapshot: Current order book state.
            volume: Remaining inventory to execute (for market order cost).
            side: "sell" or "buy".

        Returns:
            List of integer bin indices, one per feature in self.feature_names.
        """
        raw = self._extract_raw(snapshot, volume, side)
        result = []
        for name, val in zip(self.feature_names, raw):
            result.append(self._discretize(val, self.bin_edges.get(name, np.array([]))))
        return result

    def _extract_raw(
        self, snapshot: OrderBookSnapshot, volume: int, side: str
    ) -> List[float]:
        """Compute continuous (un-discretized) feature values."""
        result = []
        for name in self.feature_names:
            if name == "bid_ask_spread":
                # Paper Section 4.2: "Bid-Ask Spread" — 7.97% improvement
                result.append(snapshot.spread())

            elif name == "immediate_market_order_cost":
                # Paper Section 4.2: "Market order cost is a measure of liquidity
                # beyond the bid-ask spread — how much would it cost to submit a
                # market order for the balance of inventory immediately"
                result.append(snapshot.market_order_cost_bps(volume, side))

            elif name == "bid_ask_volume_imbalance":
                # Paper Section 4.2: "signed difference between volumes quoted at bid and ask"
                # ASSUMED: computed at best level only (confidence 0.70)
                # TODO: verify whether imbalance is computed across N levels
                imbalance = snapshot.bid_volume() - snapshot.ask_volume()
                result.append(float(imbalance))

            elif name == "signed_transaction_volume":
                # Paper Section 4.2: "signed volume of all trades within last 15 seconds"
                # Positive = buy-dominated, negative = sell-dominated
                result.append(float(snapshot.recent_trade_volume))

        return result

    def _discretize(self, value: float, edges: np.ndarray) -> int:
        """Map a continuous value to a bin index using provided edges.

        Args:
            value: Continuous feature value.
            edges: Array of (n_bins - 1) bin boundary values.

        Returns:
            Integer bin index in [0, n_bins - 1].
        """
        if not np.isfinite(value):
            return self.n_bins - 1  # extreme values go to highest bin
        return int(np.searchsorted(edges, value, side="right"))

    @property
    def state_dims(self) -> List[int]:
        """Number of discrete values per feature (for state space sizing)."""
        return [self.n_bins] * len(self.feature_names)

    def __repr__(self) -> str:
        return (
            f"MarketFeatureExtractor(features={self.feature_names}, "
            f"n_bins={self.n_bins})"
        )
