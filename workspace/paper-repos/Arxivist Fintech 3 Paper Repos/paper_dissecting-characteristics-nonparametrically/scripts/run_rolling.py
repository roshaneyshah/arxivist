"""
scripts/run_rolling.py
======================
Rolling conditional mean function estimation: replicates Figures 12–15.

Estimates the conditional mean function on rolling 10-year windows and
plots time variation in the shape of m_ts(c) over the sample period.

Usage:
    python scripts/run_rolling.py --config configs/config.yaml --use-synthetic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dcnp.utils.config import load_config, set_seed
from dcnp.models.nonparametric import AdaptiveGroupLASSOModel
from dcnp.data.transforms import RankNormalizer


def parse_args():
    p = argparse.ArgumentParser(description="DCNP Rolling Conditional Mean Functions (Figs 12–15)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--use-synthetic", action="store_true")
    p.add_argument("--char-indices", type=int, nargs="+", default=[0, 1],
                   help="Characteristic indices to plot (default: 0 and 1)")
    p.add_argument("--n-knots", type=int, default=9)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


def run_rolling(args):
    cfg = load_config(args.config)
    set_seed(cfg.reproducibility.seed)

    output_dir = Path(args.output_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    if args.use_synthetic:
        from dcnp.data.synthetic_generator import generate_synthetic_panel
        panel = generate_synthetic_panel(
            n_stocks=300, n_periods=360,
            n_chars=cfg.model.n_characteristics,
            dgp="nonlinear", seed=cfg.reproducibility.seed,
        )
        char_cols = [f"char_{s}" for s in range(cfg.model.n_characteristics)]
        char_labels = char_cols
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
        char_labels = CHARACTERISTIC_COLS

    panel["date"] = pd.to_datetime(panel["date"])
    unique_dates = sorted(panel["date"].unique())
    normalizer = RankNormalizer()
    window = cfg.estimation.rolling_window_months
    grid = np.linspace(0, 1, 50)

    # Store: {char_idx: {date: function_values}}
    rolling_functions = {ci: {} for ci in args.char_indices}

    print(f"[Rolling] Estimating over {len(unique_dates) - window} periods...")
    for t_idx in range(window, len(unique_dates)):
        pred_date = unique_dates[t_idx]
        est_dates = unique_dates[t_idx - window: t_idx]
        est_mask = panel["date"].isin(est_dates)
        est_data = panel[est_mask].copy()

        est_norm = normalizer.transform(est_data, date_col="date", char_cols=char_cols)
        X_est = est_norm[char_cols].values
        y_est = est_data["ret"].values
        valid = np.isfinite(X_est).all(axis=1) & np.isfinite(y_est)
        X_est, y_est = X_est[valid], y_est[valid]

        if len(y_est) < 100:
            continue

        try:
            model = AdaptiveGroupLASSOModel(
                n_knots=args.n_knots,
                char_names=char_cols,
                lambda1_grid=cfg.lasso.lambda1_grid,
                lambda2_grid=cfg.lasso.lambda2_grid,
            )
            model.fit(X_est, y_est)
        except Exception:
            continue

        for ci in args.char_indices:
            try:
                _, m_vals = model.get_conditional_mean_function(ci, grid=grid)
                # Normalize: m_ts(0.5) = 0 (Section III.F second normalization)
                m_vals = m_vals - m_vals[len(grid) // 2]
                rolling_functions[ci][pred_date] = m_vals
            except Exception:
                continue

        if t_idx % 12 == 0:
            print(f"  {pred_date.strftime('%Y-%m')} — done")

    # ---- Plot ----
    for ci in args.char_indices:
        func_dict = rolling_functions[ci]
        if not func_dict:
            continue

        plot_dates = sorted(func_dict.keys())
        years = np.array([d.year + d.month / 12 for d in plot_dates])
        Z = np.array([func_dict[d] for d in plot_dates])  # [T, n_grid]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for t_i, t_val in enumerate(years):
            alpha = 0.3 + 0.7 * (t_i / len(years))
            ax.plot(grid, Z[t_i], alpha=alpha,
                    color=plt.cm.viridis(t_i / len(years)), linewidth=0.8)

        cname = char_labels[ci] if ci < len(char_labels) else f"char_{ci}"
        ax.set_xlabel(f"Normalized {cname}", fontsize=12)
        ax.set_ylabel("Expected Return (normalized)", fontsize=12)
        ax.set_title(f"Time-varying Conditional Mean Function: {cname}\n"
                     f"(Paper: Figures 12–15)", fontsize=13)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.grid(alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(years.min(), years.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Year")

        out_path = output_dir / f"rolling_cmf_char{ci}_{cname}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Output] Saved: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    run_rolling(args)
