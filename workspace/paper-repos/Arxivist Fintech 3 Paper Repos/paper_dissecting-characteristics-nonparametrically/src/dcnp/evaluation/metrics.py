"""
evaluation/metrics.py
=====================
Evaluation metrics for the DCNP replication.

Paper reference: Section V.C
"we find the return forecasts from the nonparametric model have a slope
estimate of 0.78 and explain 3.11% of the ex-post variation in returns
at the firm level"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def compute_firm_level_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """Compute firm-level R² and slope from regressing realized on predicted returns.

    Runs: y_true = alpha + slope * y_pred + epsilon
    A well-calibrated model should have slope ≈ 1.

    Paper result: slope = 0.78, R² = 3.11% (Table 5 discussion)

    Args:
        y_true: Realized excess returns [N]
        y_pred: Predicted expected returns [N]

    Returns:
        Tuple of (r_squared, slope_coefficient)
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) < 10:
        return np.nan, np.nan

    # OLS: regress y_true on [1, y_pred]
    X = np.column_stack([np.ones(len(y_pred)), y_pred])
    try:
        coeffs = np.linalg.lstsq(X, y_true, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    alpha, slope = coeffs
    y_hat = X @ coeffs
    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(r2), float(slope)


def compute_ff3_alpha(
    portfolio_returns: np.ndarray,
    mkt_rf: np.ndarray,
    smb: np.ndarray,
    hml: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute Fama-French 3-factor alpha for a portfolio return series.

    Regresses portfolio returns on [1, MKT-RF, SMB, HML].

    Paper reference: Table 3 — FF3 alphas for characteristic-sorted portfolios

    Args:
        portfolio_returns: Monthly portfolio excess returns [T]
        mkt_rf: Market excess return [T]
        smb: SMB factor return [T]
        hml: HML factor return [T]

    Returns:
        Tuple of (alpha, t_stat_alpha, r_squared)
    """
    valid = (
        np.isfinite(portfolio_returns)
        & np.isfinite(mkt_rf)
        & np.isfinite(smb)
        & np.isfinite(hml)
    )
    y = portfolio_returns[valid]
    X = np.column_stack([
        np.ones(valid.sum()),
        mkt_rf[valid],
        smb[valid],
        hml[valid],
    ])

    if len(y) < 12:
        return np.nan, np.nan, np.nan

    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

    alpha = coeffs[0]
    y_hat = X @ coeffs
    e = y - y_hat
    n, k = X.shape
    sigma2 = np.sum(e ** 2) / (n - k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se_alpha = np.sqrt(cov[0, 0])
    t_stat = alpha / se_alpha if se_alpha > 0 else np.nan

    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(alpha), float(t_stat), float(r2)


def compute_portfolio_stats(
    returns: np.ndarray,
    annualization_factor: int = 12,
) -> dict:
    """Compute summary statistics for a return series.

    Args:
        returns: Monthly portfolio returns [T]
        annualization_factor: Periods per year (default: 12 for monthly)

    Returns:
        Dict with keys: mean, std, sharpe, min, max, t_stat
    """
    valid = returns[np.isfinite(returns)]
    if len(valid) < 2:
        return {k: np.nan for k in ["mean", "std", "sharpe", "min", "max", "t_stat"]}

    mean_r = np.mean(valid)
    std_r = np.std(valid, ddof=1)
    n = len(valid)

    return {
        "mean": float(mean_r * annualization_factor),
        "std": float(std_r * np.sqrt(annualization_factor)),
        "sharpe": float(mean_r / std_r * np.sqrt(annualization_factor)) if std_r > 0 else np.nan,
        "min": float(valid.min()),
        "max": float(valid.max()),
        "t_stat": float(mean_r / (std_r / np.sqrt(n))) if std_r > 0 else np.nan,
    }
