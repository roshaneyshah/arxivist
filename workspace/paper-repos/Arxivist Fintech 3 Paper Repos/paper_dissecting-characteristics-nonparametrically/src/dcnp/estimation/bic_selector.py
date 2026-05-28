"""
estimation/bic_selector.py
==========================
BIC-based lambda selection for the adaptive group LASSO.

Implements the Bayesian Information Criterion of Yuan & Lin (2006) for
group-structured variable selection models, as used in:
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227, Section III.D

The paper states: "In the application, we choose lambda_1 in a data-dependent
way to minimize a Bayes Information Criterion (BIC) proposed by Yuan and Lin (2006)."

Reference:
  Yuan, M. & Lin, Y. (2006). Model selection and estimation in regression
  with grouped variables. JRSS-B 68(1), 49–67.

Paper reference: Section III.D
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import warnings


class BICSelector:
    """Selects lambda for group LASSO via Yuan-Lin (2006) BIC.

    BIC criterion (Yuan & Lin 2006):
        BIC(lambda) = n * log(RSS/n) + log(n) * df(lambda)

    where:
        RSS = sum_i (y_i - X_i @ beta_hat)^2
        df(lambda) = number of non-zero groups (not individual coefficients)
                     multiplied by (L+2) (the group size)

    Args:
        groups: Column index groups, one per characteristic
        adaptive_weights: Optional adaptive weights for Stage 2 selection
    """

    def __init__(
        self,
        groups: List[List[int]],
        adaptive_weights: Optional[np.ndarray] = None,
    ) -> None:
        self.groups = groups
        self.adaptive_weights = adaptive_weights

    def compute_bic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta: np.ndarray,
    ) -> float:
        """Compute BIC for a given coefficient vector.

        BIC(lambda) = n * log(RSS/n) + log(n) * effective_df

        where effective_df counts non-zero groups (Yuan & Lin 2006 group BIC).

        Args:
            X: Design matrix [N, p]
            y: Response [N]
            beta: Coefficient vector [p]

        Returns:
            BIC value (lower is better)
        """
        n = len(y)
        residuals = y - X @ beta
        rss = np.sum(residuals ** 2)

        # Effective degrees of freedom: count non-zero groups * group_size
        # (Yuan & Lin 2006 approach for group LASSO)
        nonzero_groups = sum(
            1 for cols in self.groups if np.any(beta[cols] != 0.0)
        )
        group_size = len(self.groups[0]) if self.groups else 1
        df = nonzero_groups * group_size

        if rss <= 0:
            return -np.inf

        bic = n * np.log(rss / n) + np.log(n) * df
        return bic

    def select_lambda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_grid: List[float],
    ) -> float:
        """Select lambda by minimizing BIC over provided grid.

        For each lambda in the grid:
          1. Fit group LASSO (Stage 1) or adaptive group LASSO (Stage 2)
          2. Compute BIC
        Return the lambda minimizing BIC.

        Args:
            X: Design matrix [N, p]
            y: Response vector [N]
            lambda_grid: List of candidate lambda values (positive floats)

        Returns:
            Optimal lambda value
        """
        from ..models.group_lasso import AdaptiveGroupLASSO

        best_bic = np.inf
        best_lambda = lambda_grid[0]

        for lam in sorted(lambda_grid):
            try:
                if self.adaptive_weights is None:
                    # Stage 1: standard group LASSO
                    glasso = AdaptiveGroupLASSO(
                        groups=self.groups,
                        lambda1=lam,
                        lambda2=1.0,
                    )
                    beta = glasso.fit_stage1(X, y)
                else:
                    # Stage 2: adaptive group LASSO
                    glasso = AdaptiveGroupLASSO(
                        groups=self.groups,
                        lambda1=1.0,
                        lambda2=lam,
                    )
                    glasso._adaptive_weights = self.adaptive_weights
                    beta = glasso.fit_stage2(X, y)

                bic_val = self.compute_bic(X, y, beta)

                if bic_val < best_bic:
                    best_bic = bic_val
                    best_lambda = lam

            except Exception as e:
                warnings.warn(f"BIC computation failed for lambda={lam}: {e}")
                continue

        return best_lambda

    def __repr__(self) -> str:
        stage = "Stage2" if self.adaptive_weights is not None else "Stage1"
        return f"BICSelector({stage}, n_groups={len(self.groups)})"
