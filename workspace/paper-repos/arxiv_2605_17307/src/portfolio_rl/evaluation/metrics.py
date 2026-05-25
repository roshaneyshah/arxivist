"""Performance metrics (paper §5.8). ARC, ASD, MD, MLD, IR1, IR2, IR3, Sharpe."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(-dd.min())  # positive number


def _max_loss_duration(equity: pd.Series) -> int:
    peak = equity.cummax()
    underwater = equity < peak
    max_run = run = 0
    for u in underwater:
        run = run + 1 if u else 0
        max_run = max(max_run, run)
    return max_run


class PerformanceMetrics:
    @staticmethod
    def compute_all(returns: pd.Series) -> dict:
        r = returns.dropna()
        equity = (1.0 + r).cumprod()
        n = len(r)
        if n == 0:
            return {k: 0.0 for k in ["ARC", "ASD", "MD", "MLD", "Sharpe", "IR1", "IR2", "IR3"]}
        arc = float(equity.iloc[-1] ** (TRADING_DAYS / n) - 1.0)
        asd = float(r.std() * np.sqrt(TRADING_DAYS))
        md = _max_drawdown(equity)
        mld = _max_loss_duration(equity) / TRADING_DAYS
        sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
        ir1 = arc / asd if asd > 0 else 0.0
        # IR2 (Eq. 28) = IR1 * ARC * sign(ARC) / MD
        ir2 = ir1 * arc * np.sign(arc) / md if md > 0 else 0.0
        # IR3 (Eq. 29) = ARC^3 / (ASD * MD * MLD)
        ir3 = (arc ** 3) / (asd * md * mld) if asd > 0 and md > 0 and mld > 0 else 0.0
        return {
            "ARC": arc * 100,            # in %
            "ASD": asd * 100,
            "MD": md * 100,
            "MLD": mld,
            "Sharpe": sharpe,
            "IR1": ir1,
            "IR2": ir2,
            "IR3": ir3,
        }
