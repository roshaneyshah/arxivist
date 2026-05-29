"""
models/group_lasso.py
=====================
Two-step Adaptive Group LASSO estimator for nonparametric characteristic
selection and estimation.

Implements the procedure of Huang, Horowitz & Wei (2010) as applied in:
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227, Section III.D

Two-step procedure:
  Stage 1 (Eq. 5): Group LASSO with BIC-selected lambda1
  Stage 2 (Eq. 7): Adaptive group LASSO with weights from Stage 1 (Eq. 6)
  Final:            OLS re-estimation on selected characteristics

The group penalty sets ALL coefficients of a characteristic group to zero if
the characteristic has no predictive power, enabling joint model selection
and nonparametric estimation.

Paper reference: Section III.D, Equations 5, 6, 7
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


class AdaptiveGroupLASSO:
    """Two-step Adaptive Group LASSO for nonparametric model selection.

    Implements Equations 5–7 of Freyberger et al. (2017):

    Stage 1 objective (Eq. 5):
        min_b  ||y - X b||^2  +  lambda1 * sum_s ||b_s||_2

    Adaptive weights (Eq. 6):
        w_s = ||b_tilde_s||_2^{-1}  if nonzero, else infinity

    Stage 2 objective (Eq. 7):
        min_b  ||y - X b||^2  +  lambda2 * sum_s w_s ||b_s||_2

    OLS re-estimation on selected groups (for oracle efficiency).

    Args:
        groups: List of lists; groups[s] contains column indices for characteristic s
        lambda1: Penalty parameter for Stage 1 (set via BIC in practice)
        lambda2: Penalty parameter for Stage 2 (set via BIC in practice)
        tol: Convergence tolerance for coordinate descent
        max_iter: Maximum iterations for coordinate descent
    """

    def __init__(
        self,
        groups: List[List[int]],
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        self.groups = groups
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tol = tol
        self.max_iter = max_iter

        self._beta_tilde: Optional[np.ndarray] = None  # Stage 1 coefficients
        self._beta_breve: Optional[np.ndarray] = None  # Stage 2 coefficients
        self._beta_hat: Optional[np.ndarray] = None    # OLS re-estimated
        self._selected_groups: Optional[List[int]] = None
        self._adaptive_weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Stage 1: Group LASSO
    # ------------------------------------------------------------------

    def fit_stage1(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit Stage 1 group LASSO (Eq. 5).

        Uses block coordinate descent (group-wise soft-thresholding).
        Each group is updated by the group soft-threshold operator:
            b_s <- max(1 - lambda1 / ||r_s||_2, 0) * r_s
        where r_s is the OLS residual-based update for group s.

        Paper reference: Section III.D, Eq. (5)

        Args:
            X: Design matrix [N, p] where p = S*(L+2)
            y: Response vector [N]

        Returns:
            Stage 1 coefficient vector [p]
        """
        assert X.ndim == 2, f"X must be [N, p], got {X.shape}"
        assert y.ndim == 1, f"y must be [N], got {y.shape}"
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"

        n, p = X.shape
        beta = np.zeros(p)

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for group_idx, cols in enumerate(self.groups):
                # Compute partial residuals excluding this group
                other_cols = [c for g, grp in enumerate(self.groups)
                              if g != group_idx for c in grp]
                if other_cols:
                    residual = y - X[:, other_cols] @ beta[other_cols]
                else:
                    residual = y.copy()

                X_g = X[:, cols]
                # OLS update for this group (unconstrained)
                # r_s = (X_g' X_g)^{-1} X_g' residual
                XtX = X_g.T @ X_g
                Xtr = X_g.T @ residual
                try:
                    r_s = np.linalg.solve(XtX + 1e-10 * np.eye(len(cols)), Xtr)
                except np.linalg.LinAlgError:
                    r_s = np.linalg.lstsq(XtX, Xtr, rcond=None)[0]

                r_norm = np.linalg.norm(r_s)
                # Group soft-threshold operator (Eq. 5 penalty structure)
                if r_norm > 0:
                    shrinkage = max(1.0 - self.lambda1 / r_norm, 0.0)
                    beta[cols] = shrinkage * r_s
                else:
                    beta[cols] = 0.0

            # Check convergence
            change = np.max(np.abs(beta - beta_old))
            if change < self.tol:
                break
        else:
            warnings.warn(
                f"Stage 1 group LASSO did not converge in {self.max_iter} iterations "
                f"(lambda1={self.lambda1:.4f}). Consider increasing max_iter or adjusting lambda grid."
            )

        self._beta_tilde = beta
        return beta

    # ------------------------------------------------------------------
    # Adaptive weights (Eq. 6)
    # ------------------------------------------------------------------

    def compute_adaptive_weights(
        self, beta_tilde: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute adaptive weights from Stage 1 coefficients (Eq. 6).

        w_s = (sum_k beta_tilde_sk^2)^{-1/2}  if sum != 0
              infinity                           if sum == 0

        Weights of infinity prevent Stage 2 from selecting characteristics
        that Stage 1 zeroed out.

        Paper reference: Section III.D, Eq. (6)

        Args:
            beta_tilde: Stage 1 coefficients (uses self._beta_tilde if None)

        Returns:
            Weight vector [S] with inf for groups that were zeroed in Stage 1
        """
        if beta_tilde is None:
            if self._beta_tilde is None:
                raise RuntimeError("Must call fit_stage1 before computing weights")
            beta_tilde = self._beta_tilde

        S = len(self.groups)
        weights = np.empty(S)

        for s, cols in enumerate(self.groups):
            group_norm_sq = np.sum(beta_tilde[cols] ** 2)
            if group_norm_sq == 0.0:
                weights[s] = np.inf   # Eq. (6): w_s = infinity if zero
            else:
                weights[s] = group_norm_sq ** (-0.5)  # Eq. (6): w_s = ||b_tilde_s||_2^{-1}

        self._adaptive_weights = weights
        return weights

    # ------------------------------------------------------------------
    # Stage 2: Adaptive Group LASSO
    # ------------------------------------------------------------------

    def fit_stage2(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit Stage 2 adaptive group LASSO (Eq. 7).

        Identical to Stage 1 but uses adaptive per-group penalties w_s * lambda2.
        Groups with infinite weights (from Stage 1 zeros) are permanently excluded.

        Paper reference: Section III.D, Eq. (7)
        Theoretical property: model-selection consistent (Huang et al. 2010)

        Args:
            X: Design matrix [N, p]
            y: Response vector [N]
            weights: Adaptive weights [S] (uses self._adaptive_weights if None)

        Returns:
            Stage 2 coefficient vector [p]
        """
        if weights is None:
            if self._adaptive_weights is None:
                raise RuntimeError("Must call compute_adaptive_weights before fit_stage2")
            weights = self._adaptive_weights

        n, p = X.shape
        beta = np.zeros(p)

        # Only iterate over groups with finite weight
        active_groups = [
            (s, cols) for s, cols in enumerate(self.groups)
            if np.isfinite(weights[s])
        ]

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for group_idx, cols in active_groups:
                w_s = weights[group_idx]
                effective_lambda = self.lambda2 * w_s

                # Partial residuals
                other_active_cols = [
                    c for gi, grp in active_groups
                    if gi != group_idx for c in grp
                ]
                if other_active_cols:
                    residual = y - X[:, other_active_cols] @ beta[other_active_cols]
                else:
                    residual = y.copy()

                X_g = X[:, cols]
                XtX = X_g.T @ X_g
                Xtr = X_g.T @ residual
                try:
                    r_s = np.linalg.solve(XtX + 1e-10 * np.eye(len(cols)), Xtr)
                except np.linalg.LinAlgError:
                    r_s = np.linalg.lstsq(XtX, Xtr, rcond=None)[0]

                r_norm = np.linalg.norm(r_s)
                if r_norm > 0:
                    shrinkage = max(1.0 - effective_lambda / r_norm, 0.0)
                    beta[cols] = shrinkage * r_s
                else:
                    beta[cols] = 0.0

            change = np.max(np.abs(beta - beta_old))
            if change < self.tol:
                break
        else:
            warnings.warn(
                f"Stage 2 adaptive group LASSO did not converge in {self.max_iter} iterations."
            )

        self._beta_breve = beta
        self._selected_groups = [
            s for s, cols in enumerate(self.groups)
            if np.any(beta[cols] != 0.0)
        ]
        return beta

    # ------------------------------------------------------------------
    # OLS re-estimation
    # ------------------------------------------------------------------

    def refit_ols(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OLS re-estimation on selected characteristics for oracle efficiency.

        After model selection (Stages 1 & 2), re-estimate by OLS using only
        the selected characteristic columns. This corrects the shrinkage bias
        of the LASSO estimator.

        Paper reference: Section III.D:
        "We re-estimate the model for the selected characteristics with OLS
        to address this problem [non-oracle efficiency]."

        Args:
            X: Full design matrix [N, p]
            y: Response vector [N]

        Returns:
            Full coefficient vector [p] with non-selected groups zeroed
        """
        if self._selected_groups is None:
            raise RuntimeError("Must call fit_stage2 before refit_ols")

        p = X.shape[1]
        beta_hat = np.zeros(p)

        if not self._selected_groups:
            warnings.warn("No characteristics selected — returning zero coefficients.")
            self._beta_hat = beta_hat
            return beta_hat

        # Collect columns for selected groups
        selected_cols = [c for s in self._selected_groups for c in self.groups[s]]
        X_sel = X[:, selected_cols]

        # OLS: beta = (X'X)^{-1} X'y
        try:
            beta_sel = np.linalg.solve(
                X_sel.T @ X_sel + 1e-10 * np.eye(len(selected_cols)),
                X_sel.T @ y,
            )
        except np.linalg.LinAlgError:
            beta_sel = np.linalg.lstsq(X_sel, y, rcond=None)[0]

        beta_hat[selected_cols] = beta_sel
        self._beta_hat = beta_hat
        return beta_hat

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def selected_groups(self) -> List[int]:
        """Indices of selected (non-zero) characteristic groups."""
        if self._selected_groups is None:
            raise RuntimeError("Model has not been fit yet.")
        return self._selected_groups

    def n_selected(self) -> int:
        """Number of selected characteristics."""
        return len(self.selected_groups())

    def __repr__(self) -> str:
        n = self.n_selected() if self._selected_groups is not None else "?"
        return (
            f"AdaptiveGroupLASSO(n_groups={len(self.groups)}, "
            f"lambda1={self.lambda1:.4f}, lambda2={self.lambda2:.4f}, "
            f"n_selected={n})"
        )
