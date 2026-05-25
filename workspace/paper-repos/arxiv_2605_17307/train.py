"""Training entrypoint for arXiv:2605.17307 reproduction.

Usage:
    python train.py --config configs/lstm_2_ndx.yaml [--quick-test] [--seed 42]

`--quick-test` runs a single fold with 2 epochs on synthetic data, so it can be
executed CPU-only in a few minutes (smoke test).

Full reproduction (without --quick-test) requires downloaded data
(see ``data/download.sh``) and a GPU (NVIDIA L4 or similar).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from portfolio_rl.utils import load_config, resolve_device, seed_everything


def _synthetic_fold(k: int, lookback: int, n_days: int = 400, f_asset: int = 15, f_global: int = 7):
    rng = np.random.default_rng(0)
    asset_feats = rng.standard_normal((n_days, k, f_asset)).astype(np.float32) * 0.5
    global_feats = rng.standard_normal((n_days, f_global)).astype(np.float32) * 0.3
    asset_returns = rng.normal(0.0005, 0.012, size=(n_days, k)).astype(np.float32)
    benchmark = asset_returns.mean(axis=1)
    return asset_feats, global_feats, asset_returns, benchmark


def quick_test(cfg: dict) -> dict:
    """Run a tiny SAC training loop on synthetic data to verify wiring."""
    from portfolio_rl.envs import PortfolioEnv
    from portfolio_rl.agents import SACAgent

    k = cfg["data"]["topk"]
    lookback = cfg["data"]["lookback_window"]
    af, gf, ar, br = _synthetic_fold(k, lookback)

    env = PortfolioEnv(
        asset_features=af, global_features=gf,
        asset_returns=ar, benchmark_returns=br,
        lookback_window=lookback,
        transaction_cost_bps=cfg["data"]["transaction_cost_bps"],
        lambda_turnover=cfg["evaluation"]["lambda_turnover"],
        lambda_concentration=cfg["evaluation"]["lambda_concentration"],
        reward_type=cfg["evaluation"]["reward_type"],
        allow_cash=cfg["model"]["allow_cash"],
    )

    device = resolve_device(cfg.get("device", "cpu"))
    agent = SACAgent(cfg, env, device)

    obs, _ = env.reset()
    total_reward = 0.0
    n_steps = 200
    for step in range(n_steps):
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, term, trunc, info = env.step(action)
        agent.buffer.push(obs, action, reward, next_obs, term or trunc)
        losses = agent.update()
        total_reward += reward
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs

    print(f"[quick-test] {n_steps} steps complete, mean_reward={total_reward / n_steps:.4f}")
    return {"steps": n_steps, "mean_reward": total_reward / n_steps}


def full_training(cfg: dict, output_dir: Path) -> None:
    """Full WFO across all 16 folds + adaptive retraining.

    NOTE: this requires real data (run ``bash data/download.sh`` first) and is
    extremely expensive (~14-23h per fold on NVIDIA L4). The implementation below
    constructs the fold schedule and per-fold env, but **does not** wire up the
    real-data feature pipeline end-to-end — that requires the user to drop
    historical membership CSVs (see ``data/membership/README.md``). Without
    those, only the synthetic ``--quick-test`` path is meaningful.
    """
    from portfolio_rl.training import WalkForwardRunner, AdaptiveRetrainPolicy

    runner = WalkForwardRunner(
        start_date=cfg["data"]["start_date"],
        train_years=cfg["evaluation"]["train_years"],
        val_years=cfg["evaluation"]["val_years"],
        test_years=cfg["evaluation"]["test_years"],
        num_folds=cfg["evaluation"]["num_folds"],
    )
    policy = AdaptiveRetrainPolicy(
        m=cfg["evaluation"]["adaptive_retrain_window"],
        max_folds_without_retraining=cfg["evaluation"]["max_folds_without_retraining"],
    )
    folds = runner.folds()
    print(f"[wfo] Generated {len(folds)} folds:")
    for f in folds[:3]:
        print(f"  fold {f.fold_idx}: train {f.train_start}->{f.train_end} | "
              f"val {f.val_start}->{f.val_end} | test {f.test_start}->{f.test_end}")
    print("  ...")

    raise NotImplementedError(
        "Full WFO requires real data + historical index membership. "
        "See README and data/membership/README.md. "
        "Use --quick-test for a synthetic smoke test."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quick-test", action="store_true")
    ap.add_argument("--output-dir", type=str, default="runs/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(cfg.get("seed", 42))

    out = Path(args.output_dir) / Path(args.config).stem / f"seed{cfg.get('seed', 42)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.json").write_text(json.dumps(cfg, indent=2))

    if args.quick_test:
        result = quick_test(cfg)
        (out / "quick_test_result.json").write_text(json.dumps(result, indent=2))
        return
    full_training(cfg, out)


if __name__ == "__main__":
    main()
