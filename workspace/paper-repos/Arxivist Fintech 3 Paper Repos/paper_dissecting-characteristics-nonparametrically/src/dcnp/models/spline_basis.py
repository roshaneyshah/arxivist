"""
models/spline_basis.py
======================
Quadratic spline basis expansion for the nonparametric conditional mean
function estimator.

Implements the basis functions from Section III.D of:
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227

Key equation implemented:
  p_1(c) = 1
  p_2(c) = c
  p_3(c) = c^2
  p_k(c) = max{c - t_{k-3}, 0}^2   for k = 4, ..., L+2

where t_l = l/L are the interior knots (placed at equally-spaced quantiles
of the rank-normalized characteristic distribution).

Paper Section: III.D
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class QuadraticSplineBasis:
    """Quadratic spline basis expander for rank-normalized characteristics.

    Given rank-normalized characteristics in [0,1], expands each to L+2
    quadratic spline basis functions with L-1 interior knots at quantiles
    t_l = l/L. The resulting basis supports C^1 (differentiable) approximation
    of the conditional mean function.

    Paper reference: Section III.D, Eq. (4)
    "We propose to estimate quadratic functions over parts of the normalized
    characteristic distribution ... approximated by a series expansion"

    Args:
        n_knots: Number of interior knots L. Paper uses 4, 9, 14, or 19.
                 Nine knots corresponds to 10 portfolios in portfolio sorts.
    """

    def __init__(self, n_knots: int = 14) -> None:
        if n_knots < 1:
            raise ValueError(f"n_knots must be >= 1, got {n_knots}")
        self.n_knots = n_knots          # L in paper notation
        self.n_basis = n_knots + 2      # L+2 basis functions per characteristic
        self._knot_locations: Optional[np.ndarray] = None

    def fit(self, X_tilde: Optional[np.ndarray] = None) -> "QuadraticSplineBasis":
        """Set interior knot locations at equally-spaced quantiles t_l = l/L.

        Paper Section III.D: "we choose t_l = l/L for all l = 1, ..., L"
        Because the rank transformation places characteristics on [0,1],
        the knots are simply t_l = l/L.

        Args:
            X_tilde: Rank-normalized characteristics (optional; knots are
                     set analytically from n_knots, not from data quantiles)

        Returns:
            self (for method chaining)
        """
        L = self.n_knots
        # t_l = l/L for l = 1, ..., L-1 (interior knots only)
        self._knot_locations = np.array([l / L for l in range(1, L)])
        return self

    @property
    def knot_locations(self) -> np.ndarray:
        """Interior knot locations t_1, ..., t_{L-1} as 1-D array."""
        if self._knot_locations is None:
            self.fit()
        return self._knot_locations

    def basis_vector(self, c: np.ndarray) -> np.ndarray:
        """Compute basis function vector p(c) for scalar or 1-D input.

        Implements Eq. (4):
          p_1(c) = 1
          p_2(c) = c
          p_3(c) = c^2
          p_k(c) = max{c - t_{k-3}, 0}^2  for k = 4, ..., L+2

        Args:
            c: Array of shape [N] with values in [0,1]

        Returns:
            Array of shape [N, L+2]
        """
        if self._knot_locations is None:
            self.fit()

        c = np.asarray(c, dtype=float).ravel()
        N = c.shape[0]
        L = self.n_knots
        B = np.empty((N, L + 2), dtype=float)

        # p_1(c) = 1
        B[:, 0] = 1.0
        # p_2(c) = c
        B[:, 1] = c
        # p_3(c) = c^2
        B[:, 2] = c ** 2
        # p_k(c) = max{c - t_{k-3}, 0}^2  for k=4..L+2  (0-indexed: cols 3..L+1)
        for idx, t in enumerate(self._knot_locations):
            # k = idx + 4 (1-indexed), column = idx + 3 (0-indexed)
            B[:, idx + 3] = np.maximum(c - t, 0.0) ** 2  # Eq. (4), truncated power basis

        return B

    def transform(self, X_tilde: np.ndarray) -> np.ndarray:
        """Expand rank-normalized characteristics into spline design matrix.

        Stacks spline basis expansions for each characteristic side by side,
        producing the full design matrix used in the LASSO objective (Eq. 5).

        Args:
            X_tilde: Rank-normalized characteristics, shape [N, S]

        Returns:
            Design matrix of shape [N, S * (L+2)], where the first L+2
            columns correspond to characteristic 0, next L+2 to characteristic 1, etc.
        """
        assert X_tilde.ndim == 2, f"Expected [N, S], got shape {X_tilde.shape}"
        if self._knot_locations is None:
            self.fit()

        N, S = X_tilde.shape
        X_expanded = np.empty((N, S * self.n_basis), dtype=float)

        for s in range(S):
            start = s * self.n_basis
            end = start + self.n_basis
            X_expanded[:, start:end] = self.basis_vector(X_tilde[:, s])

        return X_expanded

    def group_indices(self, S: int) -> list[list[int]]:
        """Return column index groups corresponding to each characteristic.

        Used by the group LASSO to penalize entire characteristic blocks.

        Args:
            S: Number of characteristics

        Returns:
            List of S lists, each containing L+2 column indices
        """
        return [
            list(range(s * self.n_basis, (s + 1) * self.n_basis))
            for s in range(S)
        ]

    def __repr__(self) -> str:
        return f"QuadraticSplineBasis(n_knots={self.n_knots}, n_basis={self.n_basis})"
