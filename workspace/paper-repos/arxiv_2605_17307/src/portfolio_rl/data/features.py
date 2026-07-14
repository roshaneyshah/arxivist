"""Per-asset and global feature engineering (paper §5.1).

ASSUMED defaults:
    RSI(14), MACD(12, 26, 9), Bollinger(20, 2σ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_HORIZONS = (1, 5, 20, 60)
VOLATILITY_HORIZONS = (5, 20)
BETA_WINDOW = 60


def _log_returns(prices: pd.Series, horizon: int) -> pd.Series:
    return np.log(prices / prices.shift(horizon))


def _rolling_std(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_histogram(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def _bollinger_pct_b(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mu = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    upper = mu + n_std * sd
    lower = mu - n_std * sd
    return (prices - lower) / (upper - lower).replace(0.0, np.nan)


def _distance_from_high(prices: pd.Series, window: int = 20) -> pd.Series:
    return prices / prices.rolling(window).max() - 1.0


def _mean_reversion(prices: pd.Series, window: int = 20) -> pd.Series:
    return prices / prices.rolling(window).mean() - 1.0


def _rolling_beta(asset_ret: pd.Series, market_ret: pd.Series, window: int = BETA_WINDOW) -> pd.Series:
    cov = asset_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    return cov / var.replace(0.0, np.nan)


class FeatureBuilder:
    """Build per-asset feature panel (N, T_full, F_asset) and global features (T_full, F_global)."""

    def asset_features(self, prices: pd.DataFrame, market_proxy: pd.Series) -> np.ndarray:
        """Returns ndarray of shape (n_tickers, T_full, F_asset)."""
        proxy_ret = np.log(market_proxy / market_proxy.shift(1))
        feats_per_ticker: list[np.ndarray] = []
        for ticker in prices.columns:
            p = prices[ticker]
            cols: list[pd.Series] = []
            for h in MOMENTUM_HORIZONS:
                cols.append(_log_returns(p, h))
            r1 = _log_returns(p, 1)
            for w in VOLATILITY_HORIZONS:
                cols.append(_rolling_std(r1, w))
            cols.append(_rsi(p))
            cols.append(_macd_histogram(p))
            cols.append(_bollinger_pct_b(p))
            cols.append(_distance_from_high(p))
            cols.append(_mean_reversion(p))
            cols.append(_rolling_beta(r1, proxy_ret))
            cols.append(_log_returns(p, 20))
            df = pd.concat(cols, axis=1)
            feats_per_ticker.append(df.values)  # (T_full, F_asset)
        return np.stack(feats_per_ticker, axis=0)  # (N, T_full, F_asset)

    def global_features(
        self, prices: pd.DataFrame, market_proxy: pd.Series, vix: pd.Series
    ) -> np.ndarray:
        """Returns ndarray of shape (T_full, F_global)."""
        r1 = np.log(prices / prices.shift(1))
        proxy_r1 = np.log(market_proxy / market_proxy.shift(1))
        proxy_r5 = np.log(market_proxy / market_proxy.shift(5))
        proxy_r20 = np.log(market_proxy / market_proxy.shift(20))

        cross_mean = r1.mean(axis=1)
        cross_vol5 = r1.std(axis=1).rolling(5).mean()
        breadth = (r1 > 0).mean(axis=1)
        vix_chg = vix.pct_change(5)

        out = pd.concat(
            [vix.rename("vix"), vix_chg.rename("vix_chg5"),
             cross_mean.rename("xs_mean"), cross_vol5.rename("xs_vol5"),
             breadth.rename("breadth"),
             proxy_r5.rename("proxy_r5"), proxy_r20.rename("proxy_r20")],
            axis=1,
        )
        return out.values  # (T_full, F_global)
