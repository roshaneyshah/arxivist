"""Training entrypoint for SpotV2Net (arXiv:2401.06249).

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --debug
    python train.py --config configs/config.yaml --resume checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spotv2net.utils.config import load_config, set_seed  # noqa: E402
from spotv2net.data.dataset import SpotVolGraphDataset  # noqa: E402
from spotv2net.models.spotv2net import SpotV2Net  # noqa: E402
from spotv2net.training.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SpotV2Net")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--debug", action="store_true", help="Reduce dataset/steps for quick local testing")
    parser.add_argument("--dry-run", action="store_true", help="Build components without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config["training"]["seed"]
    set_seed(seed, deterministic=config["hardware"].get("deterministic", False))

    if args.debug:
        config["training"]["epochs"] = min(2, config["training"]["epochs"])
        print("[train.py] --debug enabled: epochs capped at 2")

    train_ds = SpotVolGraphDataset(
        data_dir=config["data"]["raw_data_dir"].replace("raw", "").rstrip("/") or "data",
        split="train",
        num_lags=config["model"]["num_lags"],
        horizon=config["model"]["output_dim"],
        use_edge_features=config["model"]["use_edge_features"],
    )
    val_ds = SpotVolGraphDataset(
        data_dir=config["data"]["raw_data_dir"].replace("raw", "").rstrip("/") or "data",
        split="validation",
        num_lags=config["model"]["num_lags"],
        horizon=config["model"]["output_dim"],
        use_edge_features=config["model"]["use_edge_features"],
    )

    if args.debug:
        train_ds.valid_indices = train_ds.valid_indices[:20]
        val_ds.valid_indices = val_ds.valid_indices[:10]

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=config["hardware"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config["hardware"]["num_workers"])

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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[train.py] resumed model weights from {args.resume}")

    if args.dry_run:
        print("[train.py] --dry-run: components built successfully, skipping training.")
        print(model)
        return

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.fit(epochs=config["training"]["epochs"])


if __name__ == "__main__":
    main()
