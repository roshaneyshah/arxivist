"""
data/transforms.py
==================
Data preprocessing transforms for the DCNP replication.

Primary transform: Cross-sectional rank normalization (Section III.C) of
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227.

The rank transformation maps each characteristic to its cross-sectional
rank percentile:
    C_tilde_{s,it-1} = rank(C_{s,it-1}) / (N_t + 1)

so that C_tilde in (0, 1) and the alpha-quantile of C_tilde equals alpha.
This transformation is key to: (1) robustness to outliers, (2) direct
comparability to portfolio sorts, (3) finite-sample performance.

Paper reference: Section III.C, Equation for F_{s,t}
"Hence, we choose the rank transformation ... such that the cross-sectional
distribution of a given characteristic lies in the unit interval"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


class RankNormalizer:
    """Cross-sectional rank normalization to (0,1) per time period.

    Implements the rank transformation of Section III.C:
        C_tilde_{s,it-1} = rank(C_{s,it-1}) / (N_t + 1)

    where rank() is the cross-sectional rank at time t, and N_t is
    the number of stocks in the cross-section at time t.

    This ensures:
      - C_tilde in (0, 1) (open interval, no boundary issues)
      - The alpha-quantile of C_tilde is exactly alpha
      - Outlier robustness (extreme values get extreme but bounded ranks)
      - Direct correspondence to portfolio sorting

    Paper reference: Section III.C
    "rank(min_i C_{s,it-1}) = 1 and rank(max_i C_{s,it-1}) = N_t"
    "Therefore, the alpha quantile of C_tilde is alpha."

    Args:
        method: Rank averaging method for ties ('average', 'min', 'max', 'first')
    """

    def __init__(self, method: str = "average") -> None:
        self.method = method

    def transform(
        self,
        X: pd.DataFrame,
        date_col: str = "date",
        char_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply cross-sectional rank normalization within each date.

        For each date t and each characteristic s:
            X_tilde_{s,i,t} = rank(X_{s,i,t}) / (N_t + 1)

        Args:
            X: Panel DataFrame with a date column and characteristic columns
            date_col: Name of the date column
            char_cols: Characteristic columns to normalize (default: all non-date columns)

        Returns:
            DataFrame with same structure, characteristics replaced by ranks in (0,1)
        """
        if char_cols is None:
            char_cols = [c for c in X.columns if c != date_col]

        X_out = X.copy()

        for col in char_cols:
            # Cross-sectional rank within each date period
            X_out[col] = X.groupby(date_col)[col].transform(
                lambda x: x.rank(method=self.method, na_option="keep") / (x.notna().sum() + 1)
            )

        return X_out

    def transform_array(
        self,
        X: np.ndarray,
        date_indices: np.ndarray,
    ) -> np.ndarray:
        """Apply rank normalization to a raw array using date group indices.

        Args:
            X: Characteristic matrix [N*T, S] (pooled panel, sorted by date)
            date_indices: Integer array [N*T] identifying the time period of each row

        Returns:
            Rank-normalized array [N*T, S], values in (0, 1)
        """
        assert X.ndim == 2, f"X must be [N, S], got {X.shape}"
        assert X.shape[0] == date_indices.shape[0]

        X_tilde = np.empty_like(X, dtype=float)
        unique_dates = np.unique(date_indices)

        for t in unique_dates:
            mask = date_indices == t
            X_t = X[mask, :]  # [N_t, S]
            N_t = mask.sum()

            for s in range(X.shape[1]):
                col = X_t[:, s]
                # Rank transform: F_{s,t}(C) = rank(C) / (N_t + 1)
                # Paper Eq. in Section III.C
                finite_mask = np.isfinite(col)
                ranks = np.full(N_t, np.nan)
                if finite_mask.sum() > 0:
                    # argsort of argsort gives ranks (1-indexed)
                    finite_ranks = col[finite_mask].argsort().argsort() + 1.0
                    ranks[finite_mask] = finite_ranks / (finite_mask.sum() + 1)
                X_tilde[mask, s] = ranks

        return X_tilde

    def __repr__(self) -> str:
        return f"RankNormalizer(method='{self.method}')"


def winsorize(X: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Winsorize array at given quantiles (not used in primary analysis).

    The paper uses rank normalization rather than winsorization as the
    primary outlier treatment. This function is provided for robustness
    checks (see Table 5 footnote for value-weighted portfolio robustness).

    Args:
        X: Input array (any shape)
        lower: Lower quantile cutoff (default: 1%)
        upper: Upper quantile cutoff (default: 99%)

    Returns:
        Winsorized copy of X
    """
    X_win = X.copy().astype(float)
    finite = np.isfinite(X_win.ravel())
    vals = X_win.ravel()[finite]
    lo = np.quantile(vals, lower)
    hi = np.quantile(vals, upper)
    X_win = np.clip(X_win, lo, hi)
    return X_win
