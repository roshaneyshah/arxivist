"""Preprocessing transforms: jump filtering (Sec. 7.1 footnote 7) and feature scaling.

The StandardScalerTransform is an ASSUMED preprocessing step (SIR
implementation_assumptions, confidence 0.45): the paper does not describe
explicit feature standardization, but it is standard practice for GNN/deep
learning volatility pipelines and is exposed here as an optional, config-driven
step, fit on the training split only.
"""

from __future__ import annotations

import numpy as np


class JumpFilter:
    """Removes price jumps from 1-second log returns (Sec. 7.1, footnote 7).

    theta_n = beta * (T/n)^alpha; any 1-second return whose absolute value exceeds
    theta_n is replaced with zero before Fourier estimation. Paper-reported optimal
    constants: beta=0.5, alpha=0.5.
    """

    def filter(
        self, returns: np.ndarray, T: float, n: int, beta: float = 0.5, alpha: float = 0.5
    ) -> np.ndarray:
        """Args:
        returns: 1-second log-return increments, shape ``[n]``.
        T: Trading day length (paper sets T=1, Sec. 7.1).
        n: Number of intraday observations (paper: n=23400 for a 6.5h day).
        beta: Threshold scale constant (paper-optimal: 0.5).
        alpha: Threshold exponent constant (paper-optimal: 0.5).

        Returns:
            Filtered returns with jumps replaced by 0.0, shape ``[n]``.
        """
        theta_n = beta * (T / n) ** alpha
        filtered = returns.copy()
        filtered[np.abs(filtered) > theta_n] = 0.0
        return filtered


class StandardScalerTransform:
    """Per-feature standardization, fit on the training split only.

    ASSUMED (SIR confidence 0.45): not explicitly described in the paper.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerTransform":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScalerTransform.transform called before fit().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
