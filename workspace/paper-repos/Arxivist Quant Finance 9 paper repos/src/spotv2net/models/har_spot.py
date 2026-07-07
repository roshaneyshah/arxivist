"""Panel Heterogeneous Auto-Regressive Spot volatility model (Appendix A.1).

Implements the HAR-Spot panel regression:
    V_hat_{i,b+1} = mu + phi1*V_{i,b} + phi2*mean(V_{i,b-1..b-7}) + phi3*mean(V_{i,b-8..b-13})
        + sum_{k!=i} [theta1*V_{k,b} + theta2*mean(V_{k,b-1..b-7}) + theta3*mean(V_{k,b-8..b-13})]
        + epsilon_{i,b}
"""

from __future__ import annotations

import numpy as np


class HARSpotModel:
    """Ordinary-least-squares panel HAR-Spot model (App. A.1, arXiv:2401.06249).

    The paper adapts Corsi's (2009) HAR model to intraday spot volatilities. This
    implementation performs the panel regression via closed-form OLS on the
    engineered feature matrix produced by :func:`build_har_features`.
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None  # [mu, phi1, phi2, phi3, theta1, theta2, theta3]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HARSpotModel":
        """Fit the panel HAR-Spot regression via OLS.

        Args:
            X: Feature matrix ``[num_obs, 6]`` = [own_current, own_avg7, own_avg13,
                others_current, others_avg7, others_avg13] (see ``build_har_features``).
            y: Target vector ``[num_obs]`` — next-step spot volatility.

        Returns:
            self, fitted.
        """
        design = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept mu
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict next-step spot volatility.

        Args:
            X: Feature matrix ``[num_obs, 6]``, same layout as in ``fit``.

        Returns:
            Predictions, shape ``[num_obs]``.
        """
        if self.coef_ is None:
            raise RuntimeError("HARSpotModel.predict called before fit().")
        design = np.hstack([np.ones((X.shape[0], 1)), X])
        return design @ self.coef_


def build_har_features(vol_panel: np.ndarray, asset_idx: int, t: int) -> np.ndarray:
    """Build the 6-column HAR-Spot feature row for asset ``asset_idx`` at time ``t``.

    Args:
        vol_panel: Spot volatility panel, shape ``[num_timestamps, num_assets]``.
        asset_idx: Index i of the target asset.
        t: Current timestamp index b (feature uses lags b, b-1..b-7, b-8..b-13).

    Returns:
        Feature row ``[own_current, own_avg_1_7, own_avg_8_13, others_current,
        others_avg_1_7, others_avg_8_13]``.
    """
    assert t >= 13, "Need at least 13 lags of history to build HAR-Spot features"
    own = vol_panel[:, asset_idx]
    others = np.delete(vol_panel, asset_idx, axis=1).mean(axis=1)

    own_current = own[t]
    own_avg_1_7 = own[t - 7 : t].mean()
    own_avg_8_13 = own[t - 13 : t - 7].mean()

    others_current = others[t]
    others_avg_1_7 = others[t - 7 : t].mean()
    others_avg_8_13 = others[t - 13 : t - 7].mean()

    return np.array(
        [own_current, own_avg_1_7, own_avg_8_13, others_current, others_avg_1_7, others_avg_8_13]
    )
