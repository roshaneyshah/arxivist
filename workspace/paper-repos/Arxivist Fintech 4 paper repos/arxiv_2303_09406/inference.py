"""
inference.py
============
Single-step inference with a trained FS-GCLSTM checkpoint.

Usage:
    python inference.py --config configs/config.yaml --checkpoint checkpoints/best.pt --synthetic
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fsgclstm.utils.config import load_config, set_seed
from fsgclstm.models.fsgclstm_model import FSGCLSTMModel
from fsgclstm.data.graph_builder import generate_synthetic_graph
from fsgclstm.data.dataset import generate_synthetic_returns


def parse_args():
    p = argparse.ArgumentParser(description="Run one FS-GCLSTM inference step")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--synthetic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.hardware.seed)
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        returns = generate_synthetic_returns(n_nodes=100, n_days=200, seed=42)
        adj, pred_indices = generate_synthetic_graph(n_nodes=100, n_pred=33, seed=42)
    else:
        raise NotImplementedError("Real data: see data/README_data.md")

    n_pred = len(pred_indices)
    adj = adj.to(device)

    model = FSGCLSTMModel(
        input_dim=cfg.model.input_seq_len,
        hidden_dim=cfg.model.hidden_dim,
        n_lstm_layers=cfg.model.n_lstm_layers,
        n_pred=n_pred,
        mlp_hidden=cfg.model.mlp_hidden,
        pred_node_indices=pred_indices,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Use last input_seq_len days as input
    x_window = returns[-cfg.model.input_seq_len:]       # [seq_len, N]
    x_t = torch.from_numpy(x_window).float().to(device) # [seq_len, N]

    with torch.no_grad():
        predictions = model(x_t, adj).cpu().numpy()

    long_stocks = np.where(predictions > 0)[0]
    print(f"\nPredicted next-day returns for {n_pred} target stocks:")
    print(f"  Positive (buy): {len(long_stocks)} stocks")
    print(f"  Top 5 predictions: {sorted(predictions, reverse=True)[:5]}")
    print(f"\nEqual-weight portfolio: buy top {len(long_stocks)} stocks with positive predicted return")


if __name__ == "__main__":
    main()
