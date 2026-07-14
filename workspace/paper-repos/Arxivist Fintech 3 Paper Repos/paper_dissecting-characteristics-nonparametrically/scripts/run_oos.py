"""
scripts/run_oos.py
==================
Out-of-sample rolling evaluation: replicates Table 5 of Freyberger et al. (2017).

Compares nonparametric and linear models on equally- and value-weighted
hedge portfolios for the OOS period 1991–2014.

Expected results (Table 5, col 1 vs col 3):
  NP model EW Sharpe  = 3.42
  Linear model EW Sharpe = 2.26

Usage:
    python scripts/run_oos.py --config configs/config.yaml --use-synthetic
    python scripts/run_oos.py --config configs/config.yaml --data-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dcnp.utils.config import load_config, set_seed
from dcnp.models.nonparametric import AdaptiveGroupLASSOModel
from dcnp.data.transforms import RankNormalizer
from dcnp.evaluation.portfolio import HedgePortfolioEvaluator
from dcnp.evaluation.metrics import compute_portfolio_stats, compute_firm_level_r2


def parse_args():
    p = argparse.ArgumentParser(description="DCNP OOS Evaluation (Table 5)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--use-synthetic", action="store_true")
    p.add_argument("--n-knots", type=int, default=9,
                   help="Knots for OOS (paper baseline: 9)")
    p.add_argument("--oos-start", default="1991-01",
                   help="OOS start date (paper: 1991-01)")
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def run_oos(args):
    cfg = load_config(args.config)
    set_seed(cfg.reproducibility.seed)
    n_knots = args.n_knots or 9

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DCNP Out-of-Sample Evaluation")
    print(f"Paper: Freyberger, Neuhierl & Weber (2017), Table 5")
    print(f"n_knots={n_knots}, OOS start={args.oos_start}")
    print(f"{'='*60}\n")

    # ---- Load data ----
    if args.use_synthetic:
        print("[Data] Using synthetic nonlinear DGP")
        from dcnp.data.synthetic_generator import generate_synthetic_panel
        panel = generate_synthetic_panel(
            n_stocks=500, n_periods=360,
            n_chars=cfg.model.n_characteristics,
            dgp="nonlinear", seed=cfg.reproducibility.seed,
        )
        char_cols = [f"char_{s}" for s in range(cfg.model.n_characteristics)]
    else:
        from dcnp.data.loader import PanelDataLoader, CHARACTERISTIC_COLS
        loader = PanelDataLoader(
            crsp_path=Path(args.data_dir) / "crsp_monthly.parquet",
            compustat_path=Path(args.data_dir) / "compustat_annual.parquet",
            ff3_factors_path=Path(args.data_dir) / "ff3_factors.csv",
        )
        crsp = loader.load_crsp()
        compustat = loader.load_compustat()
        panel = loader.merge_crsp_compustat(crsp, compustat)
        char_cols = CHARACTERISTIC_COLS

    panel["date"] = pd.to_datetime(panel["date"])
    print(f"[Data] Panel: {len(panel):,} obs, {len(char_cols)} chars")

    # ---- Run rolling OOS evaluation ----
    evaluator = HedgePortfolioEvaluator(
        decile=cfg.estimation.hedge_decile,
        weighting="equal",
        annualization_factor=cfg.estimation.annualization_factor,
    )

    print(f"[OOS] Running rolling {cfg.estimation.rolling_window_months}-month window...")
    oos_results = evaluator.rolling_oos_evaluation(
        panel=panel,
        model_class=AdaptiveGroupLASSOModel,
        model_kwargs={
            "n_knots": n_knots,
            "char_names": char_cols,
            "lambda1_grid": cfg.lasso.lambda1_grid,
            "lambda2_grid": cfg.lasso.lambda2_grid,
        },
        char_cols=char_cols,
        return_col="ret",
        date_col="date",
        estimation_window=cfg.estimation.rolling_window_months,
        oos_start=args.oos_start,
    )

    if oos_results.empty:
        print("[Warning] No OOS results produced — check data coverage")
        return

    hedge_returns = oos_results["hedge_return"].values
    sr_np = evaluator.compute_sharpe(hedge_returns)
    stats = compute_portfolio_stats(hedge_returns)

    print(f"\n{'='*60}")
    print(f"[Results] NP Model OOS Sharpe (EW): {sr_np:.2f}")
    print(f"[Results] Mean monthly return: {np.mean(hedge_returns)*100:.3f}%")
    print(f"[Results] Std monthly return:  {np.std(hedge_returns)*100:.3f}%")
    print(f"[Results] OOS periods:          {len(hedge_returns)}")
    print(f"[Paper target] 3.42 (Table 5, col 1, EW, 9 knots, 1991–2014)")
    print(f"{'='*60}\n")

    # ---- Save ----
    oos_results.to_csv(output_dir / "oos_hedge_returns.csv", index=False)
    summary = pd.DataFrame([{
        "model": "NP",
        "n_knots": n_knots,
        "oos_start": args.oos_start,
        "sharpe_ew": sr_np,
        "mean_annual_ret": stats["mean"],
        "std_annual": stats["std"],
        "n_oos_months": len(hedge_returns),
    }])
    summary.to_csv(output_dir / "oos_summary.csv", index=False)
    print(f"[Output] Results saved to: {output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    run_oos(args)
