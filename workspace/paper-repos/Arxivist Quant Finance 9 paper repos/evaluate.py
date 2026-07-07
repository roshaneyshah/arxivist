"""Evaluation entrypoint for SpotV2Net (arXiv:2401.06249), Sec. 7.2/7.4.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt --split test
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spotv2net.utils.config import load_config  # noqa: E402
from spotv2net.data.dataset import SpotVolGraphDataset  # noqa: E402
from spotv2net.models.spotv2net import SpotV2Net  # noqa: E402
from spotv2net.evaluation.metrics import EvaluationMetrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SpotV2Net")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"], help="Split to evaluate on")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset = SpotVolGraphDataset(
        data_dir=config["data"]["raw_data_dir"].replace("raw", "").rstrip("/") or "data",
        split=args.split,
        num_lags=config["model"]["num_lags"],
        horizon=config["model"]["output_dim"],
        use_edge_features=config["model"]["use_edge_features"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    node_in_dim = (config["model"]["num_lags"] + 1) * config["model"]["num_assets"]
    edge_in_dim = 3 * (config["model"]["num_lags"] + 1)
    model = SpotV2Net(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dims=config["model"]["hidden_dims"],
        heads=config["model"]["heads"],
        output_dim=config["model"]["output_dim"],
        use_edge_features=config["model"]["use_edge_features"],
        dropout=config["model"]["dropout_architecture"],
        attn_dropout=config["model"]["dropout_attention"],
        negative_slope=config["model"]["negative_slope"],
        activation=config["model"]["activation"],
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = EvaluationMetrics()
    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in loader:
            x, edge_index, edge_attr, y = (
                batch["x"].squeeze(0),
                batch["edge_index"].squeeze(0),
                batch["edge_attr"].squeeze(0),
                batch["y"].squeeze(0),
            )
            pred = model(x, edge_index, edge_attr)
            all_pred.append(pred.numpy())
            all_target.append(y.numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)

    mse = metrics.mse(pred, target)
    qlike = metrics.qlike(pred, target)
    print(f"[evaluate.py] split={args.split} MSE={mse:.6e} QLIKE={qlike:.4f}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"eval_{args.split}.npz")
    np.savez(out_path, pred=pred, target=target, mse=mse, qlike=qlike)
    print(f"[evaluate.py] saved predictions/targets to {out_path}")


if __name__ == "__main__":
    main()
