"""
evaluate.py
===========
Evaluation entrypoint for the Dropout reproduction.

Evaluates a trained DropoutNet checkpoint and optionally computes
hidden-layer sparsity statistics (Section 7.2, Figure 8).

Usage:
    # Evaluate on test set:
    python evaluate.py --checkpoint checkpoints/dropout_repro/best.pt \\
                       --config configs/mnist_3layer_1024.yaml

    # Evaluate on validation set:
    python evaluate.py --checkpoint checkpoints/dropout_repro/best.pt \\
                       --config configs/mnist_3layer_1024.yaml --split val

    # Include sparsity analysis (Section 7.2):
    python evaluate.py --checkpoint checkpoints/dropout_repro/best.pt \\
                       --config configs/mnist_3layer_1024.yaml --report-sparsity

Paper: Srivastava et al. (2014) JMLR 15:1929-1958.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from dropout_repro.data.dataset import MNISTDataModule
from dropout_repro.evaluation.metrics import (
    compute_error_rate,
    compute_sparsity_stats,
    print_result_vs_paper,
)
from dropout_repro.models.dropout_net import DropoutNet
from dropout_repro.utils.config import DropoutConfig, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Dropout checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved .pt checkpoint file")
    parser.add_argument("--config", type=str, default="configs/mnist_3layer_1024.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--report-sparsity", action="store_true",
                        help="Compute hidden unit sparsity statistics (Section 7.2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DropoutConfig.from_yaml(args.config)
    if args.device:
        config.hardware.device = args.device

    set_seed(config.training.seed)
    device = get_device(config.hardware.device)

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # --- Rebuild model from checkpoint config (if available) or YAML config ---
    if "config" in ckpt:
        saved_cfg = ckpt["config"]
        mc = saved_cfg.get("model", {})
    else:
        mc = {
            "input_dim": config.model.input_dim,
            "hidden_dims": config.model.hidden_dims,
            "num_classes": config.model.num_classes,
            "p_hidden": config.model.p_hidden,
            "p_input": config.model.p_input,
            "activation": config.model.activation,
            "use_dropout": config.model.use_dropout,
        }

    model = DropoutNet(
        input_dim=mc.get("input_dim", 784),
        hidden_dims=mc.get("hidden_dims", [1024, 1024, 1024]),
        num_classes=mc.get("num_classes", 10),
        p_hidden=mc.get("p_hidden", 0.5),
        p_input=mc.get("p_input", 0.8),
        activation=mc.get("activation", "relu"),
        use_dropout=mc.get("use_dropout", True),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    step = ckpt.get("step", "?")
    print(f"Model loaded from training step {step:,}" if isinstance(step, int) else f"Step: {step}")
    print(model)

    # --- Data ---
    data_module = MNISTDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        val_size=config.data.val_size,
        num_workers=config.data.num_workers,
        seed=config.training.seed,
    )
    data_module.setup()

    loader = {
        "train": data_module.train_dataloader,
        "val": data_module.val_dataloader,
        "test": data_module.test_dataloader,
    }[args.split]()

    # --- Evaluate ---
    print(f"\nEvaluating on {args.split} split ({len(loader.dataset):,} samples)...")
    results = compute_error_rate(model, loader, device)

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS — {args.split.upper()} SET")
    print(f"{'='*50}")
    print(f"  Error rate:  {results['error_rate']:.4f}%")
    print(f"  Accuracy:    {results['accuracy']:.4f}%")
    print(f"  Loss:        {results['loss']:.6f}")
    print(f"  N samples:   {results['n_samples']:,}")

    # Compare to paper target
    if args.split == "test" and config.experiment.expected_test_error_pct:
        print_result_vs_paper(
            method_name="Dropout NN + max-norm, ReLU, 3×1024",
            repro_error=results["error_rate"],
            paper_error=config.experiment.expected_test_error_pct,
        )

    # --- Sparsity analysis (Section 7.2) ---
    if args.report_sparsity:
        print(f"\nComputing sparsity statistics (Section 7.2)...")
        sparsity = compute_sparsity_stats(model, loader, device, n_batches=10)
        print(f"\n{'='*50}")
        print("SPARSITY STATISTICS")
        print(f"{'='*50}")
        print("Paper reference (Section 7.2 / Figure 8):")
        print("  Without dropout: mean activation ≈ 2.0")
        print("  With dropout:    mean activation ≈ 0.7\n")
        for key, val in sparsity.items():
            if "mean_activation" in key and "per_unit" not in key:
                print(f"  {key}: {val:.4f}")
            elif "pct_near_zero" in key:
                print(f"  {key}: {val:.2f}%")
        results["sparsity"] = {
            k: v for k, v in sparsity.items()
            if "sample" not in k and "per_unit" not in k
        }

    # --- Save results ---
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
