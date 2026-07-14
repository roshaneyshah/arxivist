"""Inference: generate weights for a single date from a trained checkpoint."""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--date", required=True, type=str, help="YYYY-MM-DD")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(f"Loaded checkpoint with keys: {list(ckpt.keys())[:5]}")
    print(f"Date requested: {args.date}")
    raise NotImplementedError(
        "Inference requires the feature pipeline (membership + price data) — "
        "complete data setup per README first."
    )


if __name__ == "__main__":
    main()
