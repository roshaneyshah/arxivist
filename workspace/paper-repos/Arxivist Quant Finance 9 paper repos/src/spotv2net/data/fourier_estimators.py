"""Non-parametric Fourier spot estimators (Sec. 6, arXiv:2401.06249).

Implements the Malliavin-Mancino Fourier-Fejer methodology for estimating:
  - spot volatility / co-volatility (Sec. 6.1)
  - spot volatility-of-volatility / co-volatility-of-volatility (Sec. 6.2)

WARNING (SIR ambiguity, confidence 0.35-0.4): the paper defers the exact cutting
frequencies (Nc, Mc, Sc, Lc and per-asset Nvi, Mvi, Svi, Lvi) used in the DJIA
empirical study to an external MATLAB library (Sanfelici & Toscano, 2024, FMVol).
The defaults exposed here are configurable placeholders — calibrate them against
that library, or the asymptotic MSE-minimizing formulas of Mancino & Recchioni
(2015), before treating results as a faithful reproduction. See data/README_data.md.
"""

from __future__ import annotations

import numpy as np


class FourierSpotEstimator:
    """Fourier-Fejer estimator of spot (co-)volatility and (co-)volatility-of-volatility.

    All methods operate on a single trading day's log-return increments sampled on
    a synchronous, equally-spaced grid (Sec. 6.1, footnote 6).
    """

    def __init__(self, trading_day_length: float = 1.0) -> None:
        """Args:
        trading_day_length: T in the paper's notation; time measured in days (Sec. 7.1
            sets T=1).
        """
        self.T = trading_day_length

    # ------------------------------------------------------------------ #
    # Sec. 6.1 — volatility and co-volatility
    # ------------------------------------------------------------------ #
    def _fourier_coeffs_returns(self, dp: np.ndarray, max_k: int) -> np.ndarray:
        """Discrete Fourier coefficients c_k(dp) for |k| <= max_k (Sec. 6.1)."""
        n = dp.shape[0]
        t = np.arange(n) / n * self.T
        ks = np.arange(-max_k, max_k + 1)
        # c_k(dp) = (1/T) sum_l exp(-i*k*2*pi/T*t_l) * dp_l
        phase = np.exp(-1j * np.outer(ks, t) * 2 * np.pi / self.T)
        return (phase @ dp) / self.T

    def estimate_covolatility(
        self, log_returns_i: np.ndarray, log_returns_j: np.ndarray, Nc: int, Mc: int, grid: np.ndarray
    ) -> np.ndarray:
        """Fourier spot co-volatility estimate C_hat_ij(tau_b) for tau_b in ``grid`` (Sec. 6.1).

        Args:
            log_returns_i: 1-second log-return increments for asset i, shape ``[n]``.
            log_returns_j: 1-second log-return increments for asset j, shape ``[n]``.
            Nc: Convolution cutting frequency (SIR ambiguity — see module docstring).
            Mc: Fourier-Fejer inversion cutoff, ``Mc < Nc``.
            grid: Target timestamps tau_b in ``[0, T]``, shape ``[B]``.

        Returns:
            Spot co-volatility path, shape ``[B]``.
        """
        if Mc >= Nc:
            raise ValueError(f"Mc ({Mc}) must be < Nc ({Nc}) per Sec. 6.1")

        c_i = self._fourier_coeffs_returns(log_returns_i, 2 * Nc)
        c_j = self._fourier_coeffs_returns(log_returns_j, 2 * Nc)

        # Eq. (3): c_k(C_{n,Nc}) = T/(2Nc+1) * sum_{|l|<=Nc} c_l(dp1) c_{k-l}(dp2)
        ks = np.arange(-Mc, Mc)
        ls = np.arange(-Nc, Nc + 1)
        offset = 2 * Nc  # index offset into c_i / c_j arrays (which span -2Nc..2Nc)
        c_C = np.zeros(len(ks), dtype=complex)
        for idx, k in enumerate(ks):
            kl = k - ls
            valid = np.abs(kl) <= 2 * Nc
            c_C[idx] = (self.T / (2 * Nc + 1)) * np.sum(
                c_i[ls[valid] + offset] * c_j[kl[valid] + offset]
            )

        return self._fejer_inversion(c_C, ks, Mc, grid)

    def estimate_volatility(self, log_returns: np.ndarray, Nv: int, Mv: int, grid: np.ndarray) -> np.ndarray:
        """Fourier spot volatility estimate V_hat_i(tau_b) (Sec. 6.1, special case i=j)."""
        return self.estimate_covolatility(log_returns, log_returns, Nv, Mv, grid).real

    def _fejer_inversion(
        self, coeffs: np.ndarray, ks: np.ndarray, M: int, grid: np.ndarray
    ) -> np.ndarray:
        """Fourier-Fejer inversion: sum_k (1 - |k|/M) c_k exp(i*k*2*pi/T*tau_b)."""
        weights = 1.0 - np.abs(ks) / M
        phase = np.exp(1j * np.outer(grid, ks) * 2 * np.pi / self.T)  # [B, len(ks)]
        return (phase @ (weights * coeffs)).real

    # ------------------------------------------------------------------ #
    # Sec. 6.2 — volatility-of-volatility and co-volatility-of-volatility
    # ------------------------------------------------------------------ #
    def estimate_covol_of_vol(
        self,
        log_returns_i: np.ndarray,
        log_returns_j: np.ndarray,
        Nc: int,
        Sc: int,
        Lc: int,
        grid: np.ndarray,
    ) -> np.ndarray:
        """Fourier spot co-volatility-of-volatility estimate (Sec. 6.2).

        Iterates the Fourier procedure on the co-volatility coefficients themselves,
        treating them as observable (Sec. 6.2): does NOT require pre-estimating the
        spot volatility path, per the paper's stated efficiency advantage.
        """
        if Sc >= Nc or Lc >= Sc:
            raise ValueError(f"Require Lc < Sc < Nc, got Lc={Lc}, Sc={Sc}, Nc={Nc}")

        c_i = self._fourier_coeffs_returns(log_returns_i, 2 * Nc)
        c_j = self._fourier_coeffs_returns(log_returns_j, 2 * Nc)
        offset = 2 * Nc

        def c_C(k: int, l_range: np.ndarray) -> complex:
            kl = k - l_range
            valid = np.abs(kl) <= 2 * Nc
            return (self.T / (2 * Nc + 1)) * np.sum(
                c_i[l_range[valid] + offset] * c_j[kl[valid] + offset]
            )

        ls = np.arange(-Nc, Nc + 1)
        ks = np.arange(-Lc, Lc)
        c_Ctilde = np.zeros(len(ks), dtype=complex)
        ss = np.arange(-Sc, Sc + 1)
        for idx, k in enumerate(ks):
            acc = 0j
            for s in ss:
                kl = k - s
                if abs(kl) > 2 * Nc or abs(s) > 2 * Nc:
                    continue
                acc += s * (s - k) * c_C(int(s), ls) * c_C(int(kl), ls)
            c_Ctilde[idx] = (self.T / (2 * Sc + 1)) * acc

        return self._fejer_inversion(c_Ctilde, ks, Lc, grid)

    def estimate_vol_of_vol(
        self, log_returns: np.ndarray, Nv: int, Sv: int, Lv: int, grid: np.ndarray
    ) -> np.ndarray:
        """Fourier spot volatility-of-volatility estimate (Sec. 6.2, special case i=j)."""
        return self.estimate_covol_of_vol(log_returns, log_returns, Nv, Sv, Lv, grid).real
