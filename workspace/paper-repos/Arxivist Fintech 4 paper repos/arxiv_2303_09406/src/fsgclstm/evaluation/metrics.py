"""
evaluation/metrics.py
=====================
Evaluation metrics for FS-GCLSTM: statistical and portfolio-based.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section IV.E

Statistical metrics: MSE, MAE, Directional Correctness
Portfolio metrics: Annualized Return, Sharpe Ratio, Sortino Ratio

Portfolio strategy (Section IV.E):
  Daily rebalanced, equal-weighted, long-only.
  Buy all stocks with positive predicted return.
  Transaction costs: 1 bps. Restricted to current index constituents.
"""
from __future__ import annotations
import numpy as np
from typing import Optional


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where the sign of predicted return matches actual."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct)) * 100.0


def equal_weight_long_only_return(
    returns: np.ndarray,
    predictions: np.ndarray,
    transaction_cost_bps: float = 1.0,
    prev_positions: Optional[np.ndarray] = None,
) -> float:
    """Compute one-day equal-weighted long-only portfolio return.

    Paper Section IV.E: "on each day, all stocks with positive predicted
    returns are equally weighted, with no short selling and transaction costs
    set to 1 bps."

    Args:
        returns: Actual next-day returns [N]
        predictions: Predicted returns [N]
        transaction_cost_bps: Transaction cost in basis points
        prev_positions: Previous day's positions for turnover calculation [N]

    Returns:
        Portfolio return for the day (scalar)
    """
    long_mask = predictions > 0
    if long_mask.sum() == 0:
        return 0.0
    weights = long_mask.astype(float) / long_mask.sum()
    gross_return = float(np.dot(weights, returns))
    # Transaction cost on turnover
    if prev_positions is not None:
        turnover = float(np.sum(np.abs(weights - prev_positions)))
        tc = turnover * transaction_cost_bps * 1e-4
    else:
        tc = transaction_cost_bps * 1e-4  # approximate first-day cost
    return gross_return - tc


def annualized_return(daily_returns: np.ndarray, trading_days: int = 252) -> float:
    """Annualized return from daily portfolio returns."""
    cumulative = np.prod(1.0 + np.array(daily_returns))
    n = len(daily_returns)
    if n == 0:
        return 0.0
    return float(cumulative ** (trading_days / n) - 1.0)


def sharpe_ratio(
    daily_returns: np.ndarray,
    risk_free_daily: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    Paper uses EONIA (Eurostoxx) and USD LIBOR O/N (S&P 500) as risk-free rates.
    """
    excess = np.array(daily_returns) - risk_free_daily
    if len(excess) < 2 or excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(trading_days))


def sortino_ratio(
    daily_returns: np.ndarray,
    risk_free_daily: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Annualized Sortino ratio (uses downside deviation only)."""
    excess = np.array(daily_returns) - risk_free_daily
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    return float(excess.mean() / downside_std * np.sqrt(trading_days))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    portfolio_returns: Optional[np.ndarray] = None,
    risk_free_daily: float = 0.0,
) -> dict:
    """Compute all metrics reported in Tables II-V of the paper."""
    results = {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "Directional_Accuracy_%": directional_accuracy(y_true, y_pred),
    }
    if portfolio_returns is not None:
        pr = np.array(portfolio_returns)
        results["Ann_Return_%"] = annualized_return(pr) * 100.0
        results["Ann_Sharpe"] = sharpe_ratio(pr, risk_free_daily)
        results["Ann_Sortino"] = sortino_ratio(pr, risk_free_daily)
    return results
