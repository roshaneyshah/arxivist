"""
scripts/run_insample.py
=======================
In-sample estimation: replicates Table 4 of Freyberger, Neuhierl & Weber (2017).

Runs the adaptive group LASSO on the full sample (1963–2014) for varying
numbers of knots and size filters, reporting selected characteristics and
in-sample Sharpe ratios.

Usage:
    python scripts/run_insample.py --config configs/config.yaml --use-synthetic
    python scripts/run_insample.py --config configs/config.yaml --data-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is on path when run from project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dcnp.utils.config import load_config, set_seed
from dcnp.models.nonparametric import AdaptiveGroupLASSOModel, CHARACTERISTIC_NAMES
from dcnp.data.transforms import RankNormalizer
from dcnp.evaluation.portfolio import HedgePortfolioEvaluator
from dcnp.evaluation.metrics import compute_portfolio_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCNP in-sample estimation (Table 4)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    p.add_argument("--data-dir", default=None, help="Directory with CRSP/Compustat parquet files")
    p.add_argument("--use-synthetic", action="store_true",
                   help="Use synthetic data (no CRSP required)")
    p.add_argument("--n-knots", type=int, default=None,
                   help="Override n_knots from config (paper uses 4, 9, 14, 19)")
    p.add_argument("--size-filter", type=int, default=None,
                   help="NYSE size percentile filter (10, 20, or 50)")
    p.add_argument("--output-dir", default="results", help="Output directory")
    return p.parse_args()


def load_data(args, cfg) -> pd.DataFrame:
    """Load real or synthetic panel data."""
    if args.use_synthetic:
        print("[Data] Using synthetic nonlinear DGP (Section III.B, Figure 1)")
        from dcnp.data.synthetic_generator import generate_synthetic_panel
        panel = generate_synthetic_panel(
            n_stocks=500,
            n_periods=240,
            n_chars=cfg.model.n_characteristics,
            dgp="nonlinear",
            seed=cfg.reproducibility.seed,
        )
        char_cols = [f"char_{s}" for s in range(cfg.model.n_characteristics)]
    else:
        from dcnp.data.loader import PanelDataLoader, CHARACTERISTIC_COLS
        loader = PanelDataLoader(
            crsp_path=Path(args.data_dir) / "crsp_monthly.parquet",
            compustat_path=Path(args.data_dir) / "compustat_annual.parquet",
            ff3_factors_path=Path(args.data_dir) / "ff3_factors.csv",
            min_price=cfg.data.min_price,
            exchanges=cfg.data.exchanges,
            share_codes=cfg.data.share_codes,
            min_compustat_years=cfg.data.min_compustat_years,
        )
        crsp = loader.load_crsp()
        compustat = loader.load_compustat()
        panel = loader.merge_crsp_compustat(crsp, compustat)
        panel = loader.apply_filters(panel)
        char_cols = CHARACTERISTIC_COLS

    return panel, char_cols


def run_insample(args):
    cfg = load_config(args.config)
    set_seed(cfg.reproducibility.seed)

    if args.n_knots is not None:
        cfg.model.n_knots = args.n_knots
    n_knots = cfg.model.n_knots

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DCNP In-Sample Estimation")
    print(f"Paper: Freyberger, Neuhierl & Weber (2017), Table 4")
    print(f"n_knots={n_knots}, size_filter={args.size_filter}")
    print(f"{'='*60}\n")

    # ---- Load data ----
    panel, char_cols = load_data(args, cfg)
    print(f"[Data] Panel shape: {panel.shape}, chars: {len(char_cols)}")

    # ---- Sample period filter ----
    panel["date"] = pd.to_datetime(panel["date"])
    mask = (
        (panel["date"] >= pd.Timestamp(cfg.data.sample_start))
        & (panel["date"] <= pd.Timestamp(cfg.data.sample_end))
    )
    panel = panel[mask].copy()
    print(f"[Data] After date filter: {len(panel):,} observations")

    # ---- Rank-normalize characteristics ----
    normalizer = RankNormalizer()
    panel_norm = normalizer.transform(panel, date_col="date", char_cols=char_cols)

    # ---- Prepare arrays ----
    X = panel_norm[char_cols].values
    y = panel["ret"].values

    # Remove NaN rows
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    print(f"[Data] After NaN removal: {len(y):,} observations")

    # ---- Fit model ----
    print(f"\n[Model] Fitting AdaptiveGroupLASSOModel (n_knots={n_knots})...")
    char_names = char_cols if not args.use_synthetic else None
    model = AdaptiveGroupLASSOModel(
        n_knots=n_knots,
        char_names=char_names,
        lambda1_grid=cfg.lasso.lambda1_grid,
        lambda2_grid=cfg.lasso.lambda2_grid,
    )
    model.fit(X, y)

    selected = model.selected_characteristics()
    n_selected = model.n_selected()
    print(f"\n[Results] Selected {n_selected} characteristics:")
    for name in selected:
        print(f"    ✓ {name}")

    # ---- Evaluate hedge portfolio ----
    predicted = model.predict(X)
    evaluator = HedgePortfolioEvaluator(
        decile=cfg.estimation.hedge_decile,
        weighting=cfg.estimation.weighting,
        annualization_factor=cfg.estimation.annualization_factor,
    )

    dates_valid = panel["date"].values[valid]
    unique_dates = sorted(set(dates_valid))

    monthly_hedge_returns = []
    panel_norm_valid = panel_norm[valid].copy() if hasattr(panel_norm, 'copy') else panel_norm
    y_valid = y
    pred_valid = predicted

    # Reconstruct per-date portfolios
    panel_idx = np.where(valid)[0]
    all_dates = panel["date"].values
    for d in unique_dates:
        mask_d = all_dates[panel_idx] == d
        if mask_d.sum() < 10:
            continue
        ret_d = y_valid[mask_d]
        pred_d = pred_valid[mask_d]
        hr = evaluator.form_portfolio(ret_d, pred_d)
        monthly_hedge_returns.append(hr)

    monthly_hedge_returns = np.array(monthly_hedge_returns)
    sr = evaluator.compute_sharpe(monthly_hedge_returns)
    stats = compute_portfolio_stats(monthly_hedge_returns)

    print(f"\n[Results] In-sample Sharpe ratio: {sr:.2f}")
    print(f"[Results] Mean monthly hedge return: {stats['mean']/12:.4f} ({stats['mean']:.2f}% annualized)")
    print(f"[Paper target] ~2.98–3.02 (Table 4, columns 1–3)")

    # ---- Save results ----
    results = {
        "n_knots": n_knots,
        "n_selected": n_selected,
        "selected_characteristics": selected,
        "sharpe_ratio": sr,
        "mean_annual_return": stats["mean"],
        "std_annual": stats["std"],
    }

    out_path = output_dir / f"insample_results_knots{n_knots}.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"\n[Output] Results saved to: {out_path}")
    return results


if __name__ == "__main__":
    args = parse_args()
    run_insample(args)
