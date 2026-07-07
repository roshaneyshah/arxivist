"""Single-snapshot inference entrypoint for SpotV2Net (arXiv:2401.06249).

Usage:
    python inference.py --config configs/config.yaml --checkpoint checkpoints/best.pt --input snapshot.npz

The input .npz file must contain arrays 'x' [N, node_in_dim], 'edge_index' [2, E],
'edge_attr' [E, edge_in_dim].
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spotv2net.utils.config import load_config  # noqa: E402
from spotv2net.models.spotv2net import SpotV2Net  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-snapshot SpotV2Net inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to a single graph snapshot (.npz)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input snapshot not found: {args.input}")
    snapshot = np.load(args.input)

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

    x = torch.from_numpy(snapshot["x"]).float()
    edge_index = torch.from_numpy(snapshot["edge_index"]).long()
    edge_attr = torch.from_numpy(snapshot["edge_attr"]).float()

    with torch.no_grad():
        pred = model(x, edge_index, edge_attr)

    print(f"[inference.py] prediction shape: {tuple(pred.shape)}")
    print(pred.numpy())


if __name__ == "__main__":
    main()
