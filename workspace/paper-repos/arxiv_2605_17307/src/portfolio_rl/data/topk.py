"""Top-k pre-selection via 120-day momentum (paper §5.1.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class TopKSelector:
    def __init__(self, k: int, horizon: int = 120):
        self.k = k
        self.horizon = horizon

    def select(self, prices: pd.DataFrame, date: pd.Timestamp) -> list[str]:
        """Return the top-k tickers by 120d momentum as of ``date``."""
        if date not in prices.index:
            date = prices.index[prices.index.get_indexer([date], method="ffill")[0]]
        end = prices.loc[date]
        start_idx = max(0, prices.index.get_loc(date) - self.horizon)
        start = prices.iloc[start_idx]
        mom = (end / start) - 1.0
        mom = mom.dropna()
        return mom.sort_values(ascending=False).head(self.k).index.tolist()
