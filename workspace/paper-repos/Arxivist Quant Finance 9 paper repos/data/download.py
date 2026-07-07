"""Data acquisition helper for SpotV2Net (arXiv:2401.06249).

The paper's raw data is 1-second TAQ Millisecond Consolidated Trades data for the
30 DJIA constituents (Sec. 7.1), accessed via Wharton Research Data Services (WRDS)
— a proprietary, licensed source that cannot be redistributed by this repository.

This script:
  1. Checks whether a user-supplied raw data extract already exists.
  2. If not, prints WRDS access instructions.
  3. Optionally generates a small synthetic dataset so the rest of the pipeline
     (Fourier estimation -> graph construction -> training) can be smoke-tested
     without real data.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


DJIA_30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",
]


def check_existing_data(raw_dir: str) -> bool:
    """Returns True if raw TAQ data already exists under ``raw_dir``."""
    return os.path.isdir(raw_dir) and len(os.listdir(raw_dir)) > 0


def print_wrds_instructions() -> None:
    print(
        "Raw data not found.\n\n"
        "SpotV2Net's raw data (Sec. 7.1) is the TAQ Millisecond Consolidated Trades\n"
        "database for the 30 DJIA constituents, 2020-06-01 to 2023-05-10, accessed via\n"
        "Wharton Research Data Services (WRDS). This requires an institutional WRDS\n"
        "subscription and cannot be redistributed here.\n\n"
        "To obtain it:\n"
        "  1. Register for WRDS access at https://wrds-www.wharton.upenn.edu/\n"
        "  2. Query the TAQ Millisecond Consolidated Trades table for the DJIA-30\n"
        "     tickers over the study period, filtered to NYSE, 9:30-16:00.\n"
        "  3. Place the resulting per-ticker tick files under data/raw/{TICKER}.csv\n"
        "     with columns ['timestamp', 'price'].\n\n"
        "Alternatively, run with --synthetic to generate a small synthetic dataset\n"
        "for smoke-testing the pipeline without real data.\n"
    )


def generate_synthetic_data(raw_dir: str, n_days: int = 5, seed: int = 42) -> None:
    """Generate a tiny synthetic 1-second tick dataset for pipeline smoke tests.

    NOT representative of real market microstructure — for code-path validation only.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(raw_dir, exist_ok=True)
    seconds_per_day = 23400  # Sec. 7.1: n=23400 one-second observations/day

    for ticker in DJIA_30:
        prices = []
        p = 100.0 + rng.normal(0, 5)
        for _ in range(n_days * seconds_per_day):
            p *= np.exp(rng.normal(0, 0.0002))
            prices.append(p)
        timestamps = pd.date_range("2023-01-02 09:30:00", periods=len(prices), freq="1s")
        df = pd.DataFrame({"timestamp": timestamps, "price": prices})
        df.to_csv(os.path.join(raw_dir, f"{ticker}.csv"), index=False)

    print(f"[download.py] wrote synthetic tick data for {len(DJIA_30)} tickers to {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SpotV2Net data acquisition helper")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Destination for raw tick data")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic smoke-test data")
    args = parser.parse_args()

    if check_existing_data(args.raw_dir):
        print(f"[download.py] raw data already present at {args.raw_dir}, skipping.")
        return

    if args.synthetic:
        generate_synthetic_data(args.raw_dir)
    else:
        print_wrds_instructions()


if __name__ == "__main__":
    main()
