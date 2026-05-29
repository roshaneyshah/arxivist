"""
evaluation/portfolio.py
=======================
Hedge portfolio construction and out-of-sample evaluation.

Implements the portfolio evaluation framework of:
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227, Sections I and V.C

Primary evaluation: equally-weighted hedge portfolio going long the top
decile and short the bottom decile of predicted expected returns.

Paper reference: Section V.C
"we take the selected characteristics and predict one-month-ahead returns,
and construct a hedge portfolio going long stocks with the 10% highest
expected returns and shorting stocks with the 10% lowest predicted returns"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class HedgePortfolioEvaluator:
    """Constructs and evaluates hedge portfolios from predicted returns.

    Forms equally-weighted or value-weighted long-short portfolios based
    on model-predicted expected returns, then computes Sharpe ratios.

    Args:
        decile: Fraction of stocks in each long/short leg (default: 0.10)
        weighting: 'equal' or 'value' (default: 'equal')
        annualization_factor: Periods per year for Sharpe ratio (default: 12)
    """

    def __init__(
        self,
        decile: float = 0.10,
        weighting: str = "equal",
        annualization_factor: int = 12,
    ) -> None:
        assert 0 < decile <= 0.5, f"decile must be in (0, 0.5], got {decile}"
        assert weighting in ("equal", "value"), f"weighting must be 'equal' or 'value'"
        self.decile = decile
        self.weighting = weighting
        self.annualization_factor = annualization_factor

    def form_portfolio(
        self,
        returns: np.ndarray,
        predicted: np.ndarray,
        market_caps: Optional[np.ndarray] = None,
    ) -> float:
        """Form hedge portfolio and return its realized return for one period.

        Sorts stocks by predicted return, goes long top decile and short
        bottom decile.

        Paper reference: Section I
        "an equally-weighted hedge portfolio going long the stocks with the
        10% highest expected returns and shorting the 10% of stocks with
        the lowest predicted returns"

        Args:
            returns: Realized excess returns for this period [N]
            predicted: Predicted expected returns [N]
            market_caps: Market caps for value-weighting [N] (required if weighting='value')

        Returns:
            Hedge portfolio return (long - short) for this period
        """
        N = len(returns)
        cutoff = max(int(np.floor(N * self.decile)), 1)

        sorted_idx = np.argsort(predicted)
        short_idx = sorted_idx[:cutoff]   # bottom decile — lowest predicted return
        long_idx = sorted_idx[-cutoff:]   # top decile — highest predicted return

        if self.weighting == "equal":
            long_ret = np.mean(returns[long_idx])
            short_ret = np.mean(returns[short_idx])
        else:
            # Value-weighted
            if market_caps is None:
                raise ValueError("market_caps required for value weighting")
            long_w = market_caps[long_idx] / market_caps[long_idx].sum()
            short_w = market_caps[short_idx] / market_caps[short_idx].sum()
            long_ret = np.dot(long_w, returns[long_idx])
            short_ret = np.dot(short_w, returns[short_idx])

        return float(long_ret - short_ret)

    def compute_sharpe(
        self,
        portfolio_returns: np.ndarray,
        annualize: bool = True,
    ) -> float:
        """Compute annualized Sharpe ratio.

        Sharpe ratio = mean(r) / std(r) * sqrt(annualization_factor)

        Note: paper reports annualized Sharpe ratios of hedge portfolios.
        Portfolio is already excess return (long - short), so no risk-free
        rate subtraction needed.

        Args:
            portfolio_returns: Time series of hedge portfolio returns [T]
            annualize: If True, multiply by sqrt(annualization_factor)

        Returns:
            Sharpe ratio
        """
        if len(portfolio_returns) < 2:
            return np.nan
        mean_r = np.mean(portfolio_returns)
        std_r = np.std(portfolio_returns, ddof=1)
        if std_r == 0:
            return np.nan
        sr = mean_r / std_r
        if annualize:
            sr *= np.sqrt(self.annualization_factor)
        return float(sr)

    def rolling_oos_evaluation(
        self,
        panel: pd.DataFrame,
        model_class,
        model_kwargs: dict,
        char_cols: List[str],
        return_col: str = "ret",
        date_col: str = "date",
        estimation_window: int = 120,
        oos_start: str = "1991-01",
        market_cap_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run full rolling out-of-sample evaluation (Section V.C).

        Estimation procedure (Table 5):
          1. Model selection on data from sample start to oos_selection_end
          2. First estimation window: 120 months before first OOS prediction
          3. For each OOS month: estimate on 120-month window, predict next month
          4. Roll forward by 1 month

        Paper reference: Section V.C
        "We then use 10 years of data to estimate the model on the selected
        characteristics. In the first month after the end of our estimation
        period, we take the selected characteristics, predict one-month-ahead
        returns..."

        Args:
            panel: Pooled panel with date, characteristics, returns
            model_class: Model class (e.g., AdaptiveGroupLASSOModel)
            model_kwargs: Kwargs passed to model_class constructor
            char_cols: List of characteristic column names
            return_col: Column name for excess returns
            date_col: Column name for dates
            estimation_window: Number of months in rolling window (default: 120)
            oos_start: First OOS prediction date (default: '1991-01')
            market_cap_col: Column for value-weighting (optional)

        Returns:
            DataFrame with columns: date, hedge_return, n_long, n_short
        """
        from ..data.transforms import RankNormalizer

        panel = panel.sort_values(date_col).copy()
        dates = pd.to_datetime(panel[date_col])
        unique_dates = sorted(dates.unique())

        oos_start_dt = pd.Timestamp(oos_start)
        oos_dates = [d for d in unique_dates if d >= oos_start_dt]

        normalizer = RankNormalizer()
        results = []

        for pred_date in oos_dates:
            # Estimation window: estimation_window months ending one month before pred_date
            pred_idx = unique_dates.index(pred_date)
            est_end_idx = pred_idx - 1
            est_start_idx = est_end_idx - estimation_window + 1

            if est_start_idx < 0:
                continue

            est_dates = unique_dates[est_start_idx: est_end_idx + 1]
            est_mask = dates.isin(est_dates)
            est_data = panel[est_mask].copy()

            pred_mask = dates == pred_date
            pred_data = panel[pred_mask].copy()

            if len(est_data) < 100 or len(pred_data) < 10:
                continue

            # Rank-normalize estimation data
            est_norm = normalizer.transform(est_data, date_col=date_col, char_cols=char_cols)
            X_est = est_norm[char_cols].values
            y_est = est_data[return_col].values

            # Remove rows with any NaN
            valid = np.isfinite(X_est).all(axis=1) & np.isfinite(y_est)
            X_est, y_est = X_est[valid], y_est[valid]

            if len(y_est) < 50:
                continue

            # Fit model
            try:
                model = model_class(**model_kwargs)
                model.fit(X_est, y_est)
            except Exception as e:
                continue

            # Rank-normalize prediction month characteristics
            pred_norm = normalizer.transform(pred_data, date_col=date_col, char_cols=char_cols)
            X_pred = pred_norm[char_cols].values
            valid_pred = np.isfinite(X_pred).all(axis=1)

            if valid_pred.sum() < 5:
                continue

            X_pred_clean = X_pred[valid_pred]
            ret_pred = pred_data[return_col].values[valid_pred]

            # Predict and form portfolio
            try:
                predicted = model.predict(X_pred_clean)
                mcap = pred_data[market_cap_col].values[valid_pred] if market_cap_col else None
                hedge_ret = self.form_portfolio(ret_pred, predicted, mcap)
            except Exception:
                continue

            results.append({
                date_col: pred_date,
                "hedge_return": hedge_ret,
                "n_stocks": valid_pred.sum(),
                "n_selected": model.n_selected() if hasattr(model, "n_selected") else np.nan,
            })

        return pd.DataFrame(results)

    def __repr__(self) -> str:
        return (
            f"HedgePortfolioEvaluator(decile={self.decile}, "
            f"weighting='{self.weighting}', annualization={self.annualization_factor})"
        )
