"""
models/nonparametric.py
=======================
Main AdaptiveGroupLASSOModel class orchestrating the full estimation pipeline.

Implements the complete nonparametric model of:
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227

Pipeline:
  1. Rank-normalize characteristics (Section III.C)
  2. Expand to quadratic spline basis (Section III.D, Eq. 4)
  3. Two-step adaptive group LASSO for model selection (Eqs. 5–7)
  4. OLS re-estimation on selected characteristics
  5. Compute estimated conditional mean functions m_ts(c)

Paper reference: Sections III.C and III.D
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .spline_basis import QuadraticSplineBasis
from .group_lasso import AdaptiveGroupLASSO
from ..estimation.bic_selector import BICSelector


class AdaptiveGroupLASSOModel:
    """Full nonparametric model for cross-sectional return prediction.

    Combines rank normalization (Section III.C), quadratic spline basis
    expansion (Section III.D, Eq. 4), and two-step adaptive group LASSO
    (Section III.D, Eqs. 5–7) with OLS re-estimation.

    The additive model structure (Eq. 2):
        R_it = sum_s m_ts(C_tilde_{s,it-1}) + epsilon_it

    allows the procedure to set individual m_ts = 0 for characteristics
    without independent predictive power, achieving model selection while
    simultaneously estimating the functional form for selected characteristics.

    Args:
        n_knots: Number of interior knots L (paper: 4, 9, 14, or 19)
        char_names: Optional list of characteristic names for labeling
        lambda1_grid: Grid of lambda1 values for BIC selection
        lambda2_grid: Grid of lambda2 values for BIC selection
    """

    def __init__(
        self,
        n_knots: int = 14,
        char_names: Optional[List[str]] = None,
        lambda1_grid: Optional[List[float]] = None,
        lambda2_grid: Optional[List[float]] = None,
    ) -> None:
        self.n_knots = n_knots
        self.char_names = char_names
        self.lambda1_grid = lambda1_grid or [0.001, 0.01, 0.1, 1.0, 10.0]
        self.lambda2_grid = lambda2_grid or [0.001, 0.01, 0.1, 1.0, 10.0]

        self.spline_basis = QuadraticSplineBasis(n_knots=n_knots)
        self._beta_hat: Optional[np.ndarray] = None
        self._selected_char_indices: Optional[List[int]] = None
        self._groups: Optional[List[List[int]]] = None
        self._n_chars: Optional[int] = None
        self._lasso: Optional[AdaptiveGroupLASSO] = None

    def fit(
        self,
        X_tilde: np.ndarray,
        y: np.ndarray,
    ) -> "AdaptiveGroupLASSOModel":
        """Fit the full two-step adaptive group LASSO model.

        Steps:
          1. Expand X_tilde [N, S] → X_spline [N, S*(L+2)] via spline basis
          2. BIC-select lambda1; fit Stage 1 group LASSO
          3. Compute adaptive weights (Eq. 6)
          4. BIC-select lambda2; fit Stage 2 adaptive group LASSO (Eq. 7)
          5. OLS re-estimation on selected characteristics

        Paper reference: Section III.D

        Args:
            X_tilde: Rank-normalized characteristics [N, S], values in [0,1]
            y: Excess returns [N]

        Returns:
            self (fitted)
        """
        assert X_tilde.ndim == 2, f"X_tilde must be [N, S], got {X_tilde.shape}"
        assert y.ndim == 1, f"y must be [N], got {y.shape}"
        assert X_tilde.shape[0] == y.shape[0], "X_tilde and y must have same N"

        N, S = X_tilde.shape
        self._n_chars = S

        # ---- Step 1: Spline basis expansion ----
        self.spline_basis.fit()
        X_spline = self.spline_basis.transform(X_tilde)   # [N, S*(L+2)]
        self._groups = self.spline_basis.group_indices(S)  # groups[s] = col indices for char s

        # ---- Step 2: BIC-select lambda1, fit Stage 1 ----
        bic_s1 = BICSelector(self._groups)
        lambda1_opt = bic_s1.select_lambda(X_spline, y, self.lambda1_grid)

        lasso = AdaptiveGroupLASSO(
            groups=self._groups,
            lambda1=lambda1_opt,
            lambda2=1.0,  # placeholder; set in Stage 2
        )
        lasso.fit_stage1(X_spline, y)

        # ---- Step 3: Adaptive weights (Eq. 6) ----
        lasso.compute_adaptive_weights()

        # ---- Step 4: BIC-select lambda2, fit Stage 2 ----
        bic_s2 = BICSelector(self._groups, adaptive_weights=lasso._adaptive_weights)
        lambda2_opt = bic_s2.select_lambda(X_spline, y, self.lambda2_grid)
        lasso.lambda2 = lambda2_opt
        lasso.fit_stage2(X_spline, y)

        # ---- Step 5: OLS re-estimation ----
        lasso.refit_ols(X_spline, y)

        self._beta_hat = lasso._beta_hat
        self._selected_char_indices = lasso.selected_groups()
        self._lasso = lasso
        return self

    def predict(self, X_tilde: np.ndarray) -> np.ndarray:
        """Predict expected returns from rank-normalized characteristics.

        Computes: R_hat_i = sum_s m_hat_ts(C_tilde_{s,i})
        using the estimated spline coefficients.

        Paper reference: Section III.D (final estimator)

        Args:
            X_tilde: Rank-normalized characteristics [N, S]

        Returns:
            Predicted expected returns [N]
        """
        if self._beta_hat is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        assert X_tilde.ndim == 2, f"X_tilde must be [N, S], got {X_tilde.shape}"

        X_spline = self.spline_basis.transform(X_tilde)
        return X_spline @ self._beta_hat

    def get_conditional_mean_function(
        self,
        char_idx: int,
        grid: Optional[np.ndarray] = None,
        other_chars_at_median: bool = True,
        X_tilde_ref: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute estimated conditional mean function m_hat_ts(c) for one characteristic.

        Returns the function on a grid of characteristic values [0,1],
        fixing all other characteristics at their median (0.5) unless a
        reference dataset is provided.

        Paper reference: Section III.F (normalization and interpretation)
        "m_t(c_tilde_1, c_bar_2, ..., c_bar_S) = m_t1(c_tilde_1) + sum_{s>=2} m_ts(c_bar_s)"

        Args:
            char_idx: Index of the characteristic (0-indexed)
            grid: Evaluation grid in [0,1]. Default: 100 equally-spaced points.
            other_chars_at_median: If True, fix other characteristics at 0.5
            X_tilde_ref: Optional reference matrix to use for other characteristics

        Returns:
            Tuple of (grid, function_values) both shape [n_grid]
        """
        if self._beta_hat is None:
            raise RuntimeError("Model must be fitted before getting conditional mean function.")
        if self._n_chars is None:
            raise RuntimeError("Model must be fitted.")

        if grid is None:
            grid = np.linspace(0.0, 1.0, 100)

        n_grid = len(grid)
        S = self._n_chars

        # Build a synthetic dataset where char_idx varies over grid,
        # others are fixed at median (0.5) — Section III.F normalization
        X_eval = np.full((n_grid, S), 0.5)
        X_eval[:, char_idx] = grid

        # Predict; this gives m_hat(c1, 0.5, ..., 0.5) as function of c1
        predicted = self.predict(X_eval)
        return grid, predicted

    def selected_characteristics(self) -> List[str]:
        """Return names of selected characteristics.

        Returns:
            List of selected characteristic names (or indices if no names given)
        """
        if self._selected_char_indices is None:
            raise RuntimeError("Model must be fitted first.")
        if self.char_names is not None:
            return [self.char_names[i] for i in self._selected_char_indices]
        return [f"char_{i}" for i in self._selected_char_indices]

    def n_selected(self) -> int:
        """Number of selected characteristics after model selection."""
        if self._selected_char_indices is None:
            raise RuntimeError("Model must be fitted first.")
        return len(self._selected_char_indices)

    def __repr__(self) -> str:
        fitted_str = f", n_selected={self.n_selected()}" if self._beta_hat is not None else ""
        return f"AdaptiveGroupLASSOModel(n_knots={self.n_knots}{fitted_str})"


# ---------------------------------------------------------------------------
# Characteristic names constant (Table 1, paper)
# ---------------------------------------------------------------------------

CHARACTERISTIC_NAMES: List[str] = [
    "A2ME", "AT", "ATO", "BEME", "Beta", "C", "CTO", "D2A", "DPI2A", "E2P",
    "FC2Y", "Free_CF", "Idio_vol", "Investment", "Lev", "LME", "Lturnover",
    "NOA", "OA", "OL", "PCM", "PM", "Prof", "Q", "Rel_to_High", "RNA",
    "ROA", "ROE", "r12_2", "r12_7", "r2_1", "r36_13", "S2P", "SGA2M",
    "Spread", "SUV",
]
