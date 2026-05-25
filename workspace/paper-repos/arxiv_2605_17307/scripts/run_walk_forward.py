"""Multi-fold orchestration helper. Wrapper around train.py invoked per fold."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-folds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for k in range(args.num_folds):
        print(f"\n========== Fold {k} ==========")
        cmd = [
            sys.executable, str(repo_root / "train.py"),
            "--config", args.config,
            "--seed", str(args.seed + k),
            "--output-dir", f"runs/fold_{k:02d}",
        ]
        subprocess.run(cmd, check=False, cwd=repo_root)


if __name__ == "__main__":
    main()
