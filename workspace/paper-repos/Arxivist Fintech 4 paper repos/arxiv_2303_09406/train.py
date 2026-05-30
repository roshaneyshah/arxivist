"""
train.py
========
Main training entrypoint for FS-GCLSTM.

Paper: Liu (2023/2025) — arXiv:2303.09406

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug
    python train.py --config configs/config.yaml --dry-run
    python train.py --config configs/config.yaml --resume checkpoints/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fsgclstm.utils.config import load_config, set_seed
from fsgclstm.models.fsgclstm_model import FSGCLSTMModel
from fsgclstm.data.dataset import StockReturnDataset, RollingWindowSplitter, generate_synthetic_returns
from fsgclstm.data.graph_builder import generate_synthetic_graph
from fsgclstm.training.trainer import Trainer
from fsgclstm.evaluation.metrics import compute_all_metrics, equal_weight_long_only_return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FS-GCLSTM (arXiv:2303.09406)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=None, help="Random seed override")
    p.add_argument("--debug", action="store_true", help="Reduce data/steps for quick local test")
    p.add_argument("--dry-run", action="store_true", help="Build all components, skip training")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (no LSEG/market data needed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.hardware.seed
    set_seed(seed, cfg.hardware.deterministic)

    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"FS-GCLSTM Training — arXiv:2303.09406")
    print(f"Device: {device} | Seed: {seed}")
    print(f"{'='*60}\n")

    # --- Load or generate data ---
    if args.synthetic or not Path(cfg.data.price_data_path).exists():
        print("[Data] Using synthetic data (set --config data.price_data_path for real data)")
        n_nodes = 50 if args.debug else 100
        n_days = 500 if args.debug else cfg.training.initial_window_days + 600
        returns = generate_synthetic_returns(n_nodes=n_nodes, n_days=n_days, seed=seed)
        adj, pred_indices = generate_synthetic_graph(n_nodes=n_nodes, n_pred=n_nodes // 3, seed=seed)
    else:
        raise NotImplementedError(
            "Real data loading not yet implemented. "
            "See data/README_data.md for LSEG and market data instructions. "
            "Use --synthetic flag for testing."
        )

    n_pred = len(pred_indices)
    adj = adj.to(device)

    # --- Build model ---
    model = FSGCLSTMModel(
        input_dim=cfg.model.input_seq_len,
        hidden_dim=cfg.model.hidden_dim,
        n_lstm_layers=cfg.model.n_lstm_layers,
        n_pred=n_pred,
        mlp_hidden=cfg.model.mlp_hidden,
        dropout=cfg.model.dropout,
        pred_node_indices=pred_indices,
    )
    print(f"[Model] {model}")
    print(f"[Model] Parameters: {model.count_parameters():,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[Model] Resumed from {args.resume}")

    if args.dry_run:
        print("[Dry-run] All components built successfully. Exiting.")
        return

    # --- Rolling window training ---
    initial_window = 200 if args.debug else cfg.training.initial_window_days
    advance = 50 if args.debug else cfg.training.advance_days

    splitter = RollingWindowSplitter(
        returns=returns,
        initial_window=initial_window,
        advance_days=advance,
        train_frac=cfg.training.train_frac,
        val_frac=cfg.training.val_frac,
        test_frac=cfg.training.test_frac,
    )

    all_portfolio_returns = []
    all_y_true, all_y_pred = [], []

    for window_idx, (train_ret, val_ret, test_ret) in enumerate(splitter.splits()):
        print(f"\n[Window {window_idx+1}] train={len(train_ret)}, val={len(val_ret)}, test={len(test_ret)} days")

        train_ds = StockReturnDataset(train_ret, adj, pred_indices, cfg.model.input_seq_len)
        val_ds = StockReturnDataset(val_ret, adj, pred_indices, cfg.model.input_seq_len)
        test_ds = StockReturnDataset(test_ret, adj, pred_indices, cfg.model.input_seq_len)

        if len(train_ds) == 0 or len(val_ds) == 0:
            print("  Skipping — insufficient data")
            continue

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

        trainer = Trainer(
            model=model,
            device=device,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            max_epochs=5 if args.debug else cfg.training.max_epochs,
            early_stop_patience=cfg.training.early_stop_patience,
            checkpoint_dir=f"checkpoints/window_{window_idx}",
            log_every=cfg.training.log_every_n_steps,
        )
        result = trainer.fit(train_loader, val_loader)
        print(f"  Best val loss: {result['best_val_loss']:.6f} at epoch {result['best_epoch']}")

        # --- Test / portfolio evaluation ---
        model.eval()
        prev_positions = None
        with torch.no_grad():
            for x_seq, adj_b, y in DataLoader(test_ds, batch_size=1, shuffle=False):
                x_s = x_seq.squeeze(0).to(device)
                adj_s = adj_b.squeeze(0).to(device)
                y_s = y.squeeze(0).cpu().numpy()
                pred = model(x_s, adj_s).cpu().numpy()
                port_ret = equal_weight_long_only_return(
                    y_s, pred, cfg.data.transaction_cost_bps, prev_positions
                )
                all_portfolio_returns.append(port_ret)
                all_y_true.append(y_s)
                all_y_pred.append(pred)

    # --- Final metrics ---
    if all_y_true:
        y_t = np.concatenate(all_y_true)
        y_p = np.concatenate(all_y_pred)
        metrics = compute_all_metrics(y_t, y_p, np.array(all_portfolio_returns))
        print(f"\n{'='*60}")
        print("Final Evaluation Metrics")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\nPaper targets (Table II, Eurostoxx 600):")
        print(f"  Ann_Return_%: 7.41  |  Ann_Sharpe: 0.462  |  Ann_Sortino: 0.592")


if __name__ == "__main__":
    main()
