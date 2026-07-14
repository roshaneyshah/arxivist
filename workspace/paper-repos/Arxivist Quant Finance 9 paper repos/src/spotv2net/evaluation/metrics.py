"""Aggregate evaluation metrics and statistical tests (Sec. 7.2, Eq. 4/5/7/8)."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


class EvaluationMetrics:
    """MSE / QLIKE aggregation and the Diebold-Mariano test (Sec. 7.2)."""

    @staticmethod
    def mse(pred: np.ndarray, target: np.ndarray) -> float:
        """Aggregate MSE (Eq. 4, single-step; Eq. 7, multi-step averaged over H)."""
        return float(np.mean((pred - target) ** 2))

    @staticmethod
    def qlike(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
        """Aggregate QLIKE (Eq. 5, single-step; Eq. 8, multi-step averaged over H)."""
        pred = np.clip(pred, eps, None)
        target = np.clip(target, eps, None)
        ratio = pred / target
        return float(np.mean(ratio - np.log(ratio) - 1.0))

    @staticmethod
    def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict[str, float]:
        """Diebold & Mariano (2002) test comparing forecast error loss series.

        Positive statistic indicates model B (errors_b) outperforms model A
        (errors_a), matching the sign convention of Tables 3/4/6/7 in the paper.

        Args:
            errors_a: Per-observation squared (or QLIKE) losses for model A.
            errors_b: Per-observation squared (or QLIKE) losses for model B.

        Returns:
            Dict with 'stat' (DM statistic) and 'pvalue' (two-sided).
        """
        d = errors_a - errors_b
        n = len(d)
        d_mean = d.mean()
        d_var = d.var(ddof=1)
        if d_var == 0:
            return {"stat": 0.0, "pvalue": 1.0}
        dm_stat = d_mean / np.sqrt(d_var / n)
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        return {"stat": float(dm_stat), "pvalue": float(pvalue)}
