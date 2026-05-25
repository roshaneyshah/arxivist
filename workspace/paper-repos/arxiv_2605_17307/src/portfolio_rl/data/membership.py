"""Index membership loader.

ASSUMED: paper uses Bloomberg historical membership. We fall back to current
membership if no user CSV is provided. This introduces survivorship bias.
See ``data/membership/README.md``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMBERSHIP_DIR = REPO_ROOT / "data" / "membership"
CACHE_DIR = REPO_ROOT / "data" / "cache"


class IndexMembership:
    def __init__(self, market: str):
        self.market = market.lower()
        self._csv_path = MEMBERSHIP_DIR / self.market / "membership.csv"
        self._df: pd.DataFrame | None = None
        self._current: list[str] | None = None
        self._load()

    def _load(self) -> None:
        if self._csv_path.exists():
            self._df = pd.read_csv(self._csv_path, parse_dates=["start", "end"])
            return
        warnings.warn(
            f"[membership] No historical CSV at {self._csv_path}. "
            "Falling back to CURRENT MEMBERSHIP ONLY — results will be survivorship-biased.",
            stacklevel=2,
        )
        universes_path = CACHE_DIR / "universes.json"
        if not universes_path.exists():
            raise FileNotFoundError(
                f"{universes_path} not found. Run `bash data/download.sh` first."
            )
        universes = json.loads(universes_path.read_text())
        self._current = universes[self.market]

    def get_active_tickers(self, date: pd.Timestamp) -> list[str]:
        if self._df is not None:
            mask = (self._df["start"] <= date) & (
                self._df["end"].isna() | (self._df["end"] > date)
            )
            return self._df.loc[mask, "ticker"].tolist()
        assert self._current is not None
        return list(self._current)
