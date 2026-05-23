"""
env/order_book.py — Order book data structures and execution simulation.

Implements the INET ECN order book model described in Section 2 of:
  Nevmyvaka, Feng, Kearns — "Reinforcement Learning for Optimized Trade Execution" (ICML 2006)

Key concepts:
  - Limit order book: sorted by price, buy side (bids) and sell side (asks)
  - Execution fills at the prices of resting orders (not the incoming order price)
  - Market impact: consuming deep in the book yields progressively worse prices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PriceLevel:
    """A single price level in the order book with aggregated volume."""
    price: float
    volume: int

    def __repr__(self) -> str:
        return f"PriceLevel(price={self.price:.4f}, volume={self.volume})"


@dataclass
class OrderBookSnapshot:
    """A point-in-time snapshot of the limit order book.

    Models the INET ECN order book as described in Section 2.
    Buy orders are sorted descending by price (best bid first).
    Sell orders are sorted ascending by price (best ask first).

    Paper reference: Section 2 — "Limit Order Trading and Market Simulation"

    Attributes:
        timestamp: Millisecond timestamp of this snapshot.
        bids: List of (price, volume) levels on the buy side, best first.
        asks: List of (price, volume) levels on the sell side, best first.
        recent_trade_volume: Signed volume of recent trades (for market features).
            Positive = buyer-initiated, negative = seller-initiated.
    """

    timestamp: int
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)
    recent_trade_volume: int = 0  # used by signed_transaction_volume feature

    def bid(self) -> float:
        """Best (highest) bid price. Returns 0.0 if book is empty."""
        return self.bids[0].price if self.bids else 0.0

    def ask(self) -> float:
        """Best (lowest) ask price. Returns 0.0 if book is empty."""
        return self.asks[0].price if self.asks else 0.0

    def mid(self) -> float:
        """Mid-spread price: (ask + bid) / 2.

        Used as the baseline for trading cost calculation.
        Paper reference: Section 3 — Rewards: "mid-spread price (ask+bid)/2 at start of episode"
        """
        return (self.ask() + self.bid()) / 2.0

    def spread(self) -> float:
        """Bid-ask spread in absolute price units."""
        return self.ask() - self.bid()

    def bid_volume(self) -> int:
        """Total volume quoted at the best bid."""
        return self.bids[0].volume if self.bids else 0

    def ask_volume(self) -> int:
        """Total volume quoted at the best ask."""
        return self.asks[0].volume if self.asks else 0

    def simulate_sell_execution(
        self, volume: int, limit_price: float
    ) -> Tuple[int, float, int]:
        """Simulate execution of a limit sell order against the buy side.

        A limit sell order at limit_price will execute against buy orders
        at prices >= limit_price, consuming the book from best bid downward.

        Paper reference: Section 2 — "If an arriving limit order can be immediately executed
        with orders on the opposing book, the executions occur."

        Args:
            volume: Number of shares to sell.
            limit_price: Minimum acceptable price (our limit).

        Returns:
            Tuple of:
              - shares_executed: How many shares were filled immediately.
              - avg_price: Volume-weighted average execution price.
              - remaining: Unexecuted shares (placed as resting limit order).
        """
        shares_filled = 0
        total_proceeds = 0.0

        for level in self.bids:
            if level.price < limit_price:
                break  # All remaining bids are below our limit; stop
            fillable = min(volume - shares_filled, level.volume)
            shares_filled += fillable
            total_proceeds += fillable * level.price
            if shares_filled >= volume:
                break

        remaining = volume - shares_filled
        avg_price = (total_proceeds / shares_filled) if shares_filled > 0 else 0.0
        return shares_filled, avg_price, remaining

    def simulate_buy_execution(
        self, volume: int, limit_price: float
    ) -> Tuple[int, float, int]:
        """Simulate execution of a limit buy order against the sell side.

        A limit buy order at limit_price will execute against sell orders
        at prices <= limit_price.

        Args:
            volume: Number of shares to buy.
            limit_price: Maximum acceptable price (our limit).

        Returns:
            Tuple of (shares_executed, avg_price, remaining).
        """
        shares_filled = 0
        total_cost = 0.0

        for level in self.asks:
            if level.price > limit_price:
                break
            fillable = min(volume - shares_filled, level.volume)
            shares_filled += fillable
            total_cost += fillable * level.price
            if shares_filled >= volume:
                break

        remaining = volume - shares_filled
        avg_price = (total_cost / shares_filled) if shares_filled > 0 else 0.0
        return shares_filled, avg_price, remaining

    def market_order_cost_bps(self, volume: int, side: str = "sell") -> float:
        """Cost of immediately submitting a market order for `volume` shares.

        This is the 'immediate_market_order_cost' market feature described in
        Section 4.2 of the paper: "how much would it cost to submit a market order
        for the balance of inventory immediately."

        Measured in basis points relative to mid-spread.

        Args:
            volume: Shares to execute immediately.
            side: "sell" consumes bids; "buy" consumes asks.

        Returns:
            Cost in basis points (positive = worse than mid).
        """
        mid = self.mid()
        if mid == 0:
            return 0.0

        if side == "sell":
            filled, avg_price, _ = self.simulate_sell_execution(volume, 0.0)
        else:
            filled, avg_price, _ = self.simulate_buy_execution(volume, float("inf"))

        if filled == 0:
            return float("inf")

        # For selling: cost = (mid - avg_price) / mid * 10000
        # Paper reference: Section 3 Rewards — "underperfomance compared to mid-spread baseline"
        cost_bps = (mid - avg_price) / mid * 10000.0 if side == "sell" else (avg_price - mid) / mid * 10000.0
        return cost_bps

    def __repr__(self) -> str:
        return (
            f"OrderBookSnapshot(t={self.timestamp}, "
            f"bid={self.bid():.4f}, ask={self.ask():.4f}, "
            f"spread={self.spread():.4f})"
        )
