"""
evaluate.py
===========
Evaluation entrypoint for a trained FS-GCLSTM checkpoint.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --synthetic
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
from fsgclstm.data.dataset import StockReturnDataset, generate_synthetic_returns
from fsgclstm.data.graph_builder import generate_synthetic_graph
from fsgclstm.evaluation.metrics import compute_all_metrics, equal_weight_long_only_return


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FS-GCLSTM checkpoint")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed or cfg.hardware.seed
    set_seed(seed)
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")

    # Data
    if args.synthetic:
        returns = generate_synthetic_returns(n_nodes=100, n_days=4000, seed=seed)
        adj, pred_indices = generate_synthetic_graph(n_nodes=100, n_pred=33, seed=seed)
    else:
        raise NotImplementedError("Real data loading: see data/README_data.md")

    adj = adj.to(device)
    n_pred = len(pred_indices)

    # Model
    model = FSGCLSTMModel(
        input_dim=cfg.model.input_seq_len,
        hidden_dim=cfg.model.hidden_dim,
        n_lstm_layers=cfg.model.n_lstm_layers,
        n_pred=n_pred,
        mlp_hidden=cfg.model.mlp_hidden,
        dropout=0.0,
        pred_node_indices=pred_indices,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_ds = StockReturnDataset(returns[-500:], adj, pred_indices, cfg.model.input_seq_len)
    all_y_true, all_y_pred, all_pr = [], [], []
    with torch.no_grad():
        for x_seq, adj_b, y in DataLoader(test_ds, batch_size=1, shuffle=False):
            x_s = x_seq.squeeze(0).to(device)
            adj_s = adj_b.squeeze(0).to(device)
            y_np = y.squeeze(0).cpu().numpy()
            pred = model(x_s, adj_s).cpu().numpy()
            all_y_true.append(y_np)
            all_y_pred.append(pred)
            all_pr.append(equal_weight_long_only_return(y_np, pred, cfg.data.transaction_cost_bps))

    metrics = compute_all_metrics(np.concatenate(all_y_true), np.concatenate(all_y_pred), np.array(all_pr))
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
