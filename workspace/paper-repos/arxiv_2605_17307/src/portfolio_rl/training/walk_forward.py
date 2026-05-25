"""Walk-forward optimisation + adaptive retraining (paper §5.6)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass


class AdaptiveRetrainPolicy:
    """Eq. 18: retrain if S_k < 0 OR S_k < median(last m) - 0.5*std(last m)
    OR >max_folds_without_retraining folds since last retrain. Always retrain if k<3."""

    def __init__(self, m: int = 5, max_folds_without_retraining: int = 3):
        self.m = m
        self.max_folds = max_folds_without_retraining

    def should_retrain(self, sharpe_history: list[float], folds_since_last: int, fold_idx: int) -> bool:
        if fold_idx < 3:
            return True
        if folds_since_last >= self.max_folds:
            return True
        if not sharpe_history:
            return True
        s_k = sharpe_history[-1]
        if s_k < 0:
            return True
        window = sharpe_history[-self.m :]
        if len(window) < 2:
            return False
        med = statistics.median(window)
        sd = statistics.stdev(window)
        return s_k < med - 0.5 * sd


@dataclass
class FoldSpec:
    fold_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


class WalkForwardRunner:
    """Non-anchored rolling WFO. Generates 16 folds (paper §5.6.1)."""

    def __init__(self, start_date: str, train_years: int, val_years: int, test_years: int, num_folds: int):
        import pandas as pd
        self.pd = pd
        self.start = pd.Timestamp(start_date)
        self.train_y = train_years
        self.val_y = val_years
        self.test_y = test_years
        self.num_folds = num_folds

    def folds(self) -> list[FoldSpec]:
        out: list[FoldSpec] = []
        for k in range(self.num_folds):
            ts = self.start + self.pd.DateOffset(years=k)
            vs = ts + self.pd.DateOffset(years=self.train_y)
            es = vs + self.pd.DateOffset(years=self.val_y)
            ee = es + self.pd.DateOffset(years=self.test_y)
            out.append(FoldSpec(
                fold_idx=k,
                train_start=ts.strftime("%Y-%m-%d"),
                train_end=vs.strftime("%Y-%m-%d"),
                val_start=vs.strftime("%Y-%m-%d"),
                val_end=es.strftime("%Y-%m-%d"),
                test_start=es.strftime("%Y-%m-%d"),
                test_end=ee.strftime("%Y-%m-%d"),
            ))
        return out
