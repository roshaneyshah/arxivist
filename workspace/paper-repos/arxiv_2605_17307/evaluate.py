"""Evaluation entrypoint: compute performance metrics + statistical tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from portfolio_rl.evaluation import PerformanceMetrics, StatisticalTests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str, help="Run directory containing oos_returns.csv")
    ap.add_argument("--benchmark", type=str, default=None, help="Benchmark ticker (e.g. QQQ)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    returns_path = run / "oos_returns.csv"
    if not returns_path.exists():
        raise FileNotFoundError(
            f"{returns_path} not found. Run train.py first (full training, not --quick-test)."
        )
    df = pd.read_csv(returns_path, parse_dates=["date"], index_col="date")

    strat = df["strategy"]
    metrics = PerformanceMetrics.compute_all(strat)
    print("\n=== Performance metrics (strategy) ===")
    for k, v in metrics.items():
        print(f"  {k:7s}: {v:.4f}")

    if args.benchmark and args.benchmark in df.columns:
        bench = df[args.benchmark]
        bm = PerformanceMetrics.compute_all(bench)
        print(f"\n=== Performance metrics ({args.benchmark}) ===")
        for k, v in bm.items():
            print(f"  {k:7s}: {v:.4f}")

        hac_p = StatisticalTests.hac_pvalue(strat, bench)
        boot_p = StatisticalTests.block_bootstrap_pvalue(strat, bench, n_boot=2000)
        print(f"\nHAC p-value (strategy vs {args.benchmark}): {hac_p:.4f}")
        print(f"Block-bootstrap p-value:                    {boot_p:.4f}")

        (run / "evaluation.json").write_text(json.dumps({
            "strategy": metrics, "benchmark": bm,
            "hac_pvalue": hac_p, "bootstrap_pvalue": boot_p,
        }, indent=2))
    else:
        (run / "evaluation.json").write_text(json.dumps({"strategy": metrics}, indent=2))


if __name__ == "__main__":
    main()
