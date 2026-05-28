"""
estimation/confidence_bands.py
================================
Uniform confidence bands for estimated conditional mean functions.

Implements the simulation-based uniform confidence bands of Section III.E in:
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227

The uniform band for characteristic s has the form:
    [m_hat_ts(c) ± d_ts * sigma_hat_ts(c)]   for all c in [0,1]

where:
    sigma_hat_ts(c) = sqrt(p(c)' Sigma_hat_ts p(c))
    Sigma_hat_ts    = HC covariance estimate of sqrt(n)(beta_hat_ts - beta_ts)
    d_ts            = critical value from:
                      P(sup_c |Z'p(c)/sqrt(p(c)'Sigma_hat p(c))| <= d_ts) = 1-alpha
                      with Z ~ N(0, Sigma_hat_ts)

Paper reference: Section III.E, Equations for d_ts and sigma_hat_ts
"we can calculate the probability on the left-hand side using simulations"

WARNING (SIR ambiguity): Number of simulation draws not specified in paper.
ASSUMED: n_sims=10,000 (standard practice). Confidence: 0.65.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class UniformConfidenceBand:
    """Simulation-based uniform confidence bands for nonparametric estimates.

    Computes pointwise standard errors and simulates the supremum of the
    studentized process to determine critical values d_ts.

    Paper reference: Section III.E

    Args:
        n_sims: Number of simulation draws for critical value computation.
                ASSUMED: 10,000 — paper does not specify. (conf: 0.65)
        alpha: Confidence level (default: 0.05 for 95% bands)
    """

    def __init__(self, n_sims: int = 10000, alpha: float = 0.05) -> None:
        # ASSUMED: n_sims=10000 — not specified in paper (SIR confidence: 0.65)
        # TODO: verify n_sims from paper or supplementary materials
        self.n_sims = n_sims
        self.alpha = alpha
        self._Sigma_hat: Optional[np.ndarray] = None
        self._beta_hat: Optional[np.ndarray] = None
        self._n: Optional[int] = None

    def fit(
        self,
        X_selected: np.ndarray,
        y: np.ndarray,
        beta_hat: np.ndarray,
    ) -> "UniformConfidenceBand":
        """Compute heteroscedasticity-consistent covariance matrix.

        HC covariance (White 1980):
            Sigma_hat = (X'X)^{-1} (X' diag(e^2) X) (X'X)^{-1}

        where e = y - X @ beta_hat are the OLS residuals.

        Paper reference: Section III.E:
        "We define Sigma_hat_ts as the heteroscedasticity-consistent estimator of Sigma_ts"

        Args:
            X_selected: Spline design matrix for selected characteristics [N, p_sel]
            y: Response vector [N]
            beta_hat: OLS coefficient vector [p_sel]
        """
        n, p = X_selected.shape
        self._n = n
        self._beta_hat = beta_hat

        residuals = y - X_selected @ beta_hat  # [N]
        XtX = X_selected.T @ X_selected        # [p, p]

        try:
            XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(p))
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        # Meat: X' diag(e^2) X
        meat = X_selected.T @ (X_selected * (residuals ** 2)[:, None])  # [p, p]

        # HC covariance estimate
        self._Sigma_hat = n * XtX_inv @ meat @ XtX_inv  # [p, p]
        return self

    def pointwise_se(
        self,
        p_grid: np.ndarray,
    ) -> np.ndarray:
        """Compute pointwise standard errors sigma_hat_ts(c) on a grid.

        sigma_hat_ts(c) = sqrt(p(c)' Sigma_hat_ts p(c))

        Paper reference: Section III.E, sigma_hat_ts definition

        Args:
            p_grid: Basis function matrix on evaluation grid [n_grid, p_sel]

        Returns:
            Pointwise SEs [n_grid]
        """
        if self._Sigma_hat is None:
            raise RuntimeError("Must call fit() before computing standard errors.")

        # [n_grid]: sqrt(diag(p_grid @ Sigma_hat @ p_grid.T))
        variances = np.sum(p_grid @ self._Sigma_hat * p_grid, axis=1)
        variances = np.maximum(variances, 0.0)  # numerical safety
        return np.sqrt(variances)

    def critical_value(
        self,
        p_grid: np.ndarray,
    ) -> float:
        """Simulate critical value d_ts for the uniform band.

        Draws Z ~ N(0, Sigma_hat) and computes:
            sup_{c in grid} |Z'p(c) / sqrt(p(c)'Sigma_hat p(c))|

        The (1-alpha) quantile of this supremum is d_ts.

        Paper reference: Section III.E:
        "let d_ts be such that P(sup_c |Z'p(c)/sqrt(p(c)'Sigma_hat p(c))| <= d_ts) = 1-alpha"
        "We can calculate the probability using simulations."

        Args:
            p_grid: Basis function matrix [n_grid, p_sel]

        Returns:
            Critical value d_ts
        """
        if self._Sigma_hat is None:
            raise RuntimeError("Must call fit() before computing critical value.")

        p_sel = self._Sigma_hat.shape[0]
        pointwise_se = self.pointwise_se(p_grid)  # [n_grid]

        # Cholesky decomposition for sampling Z ~ N(0, Sigma_hat)
        try:
            L = np.linalg.cholesky(self._Sigma_hat + 1e-10 * np.eye(p_sel))
        except np.linalg.LinAlgError:
            # Fallback: use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(self._Sigma_hat)
            eigvals = np.maximum(eigvals, 0.0)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # Simulate sup statistics
        sup_stats = np.empty(self.n_sims)
        standard_normals = np.random.randn(self.n_sims, p_sel)  # [n_sims, p_sel]
        Z_draws = standard_normals @ L.T  # [n_sims, p_sel]; Z ~ N(0, Sigma_hat)

        # Z_draws @ p_grid.T: [n_sims, n_grid]
        numerator = Z_draws @ p_grid.T  # [n_sims, n_grid]
        # Divide by pointwise SE (broadcast)
        studentized = numerator / (pointwise_se[None, :] + 1e-15)  # [n_sims, n_grid]
        sup_stats = np.max(np.abs(studentized), axis=1)  # [n_sims]

        d_ts = float(np.quantile(sup_stats, 1.0 - self.alpha))
        return d_ts

    def band(
        self,
        p_grid: np.ndarray,
        m_hat_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute uniform confidence band around estimated function.

        Returns lower band, point estimate, and upper band on the grid.

        Args:
            p_grid: Basis function matrix on evaluation grid [n_grid, p_sel]
            m_hat_grid: Point estimate m_hat_ts(c) on grid [n_grid]

        Returns:
            Tuple of (lower_band, m_hat_grid, upper_band), each [n_grid]
        """
        se = self.pointwise_se(p_grid)
        d_ts = self.critical_value(p_grid)

        lower = m_hat_grid - d_ts * se
        upper = m_hat_grid + d_ts * se
        return lower, m_hat_grid, upper

    def __repr__(self) -> str:
        return f"UniformConfidenceBand(n_sims={self.n_sims}, alpha={self.alpha})"
