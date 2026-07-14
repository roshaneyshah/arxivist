"""
train.py
========
Training entrypoint for the Dropout reproduction.

Usage:
    # Primary repro target (Table 2, ~1.06% MNIST error):
    python train.py --config configs/mnist_3layer_1024.yaml

    # Ablation: regularizer comparison (Table 9):
    python train.py --config configs/mnist_regularizer_comparison.yaml

    # Resume from checkpoint:
    python train.py --config configs/mnist_3layer_1024.yaml --resume checkpoints/dropout_repro/step_0500000.pt

    # Quick debug run (1000 updates):
    python train.py --config configs/mnist_3layer_1024.yaml --debug

    # Dry run (validate setup only):
    python train.py --config configs/mnist_3layer_1024.yaml --dry-run

Paper: Srivastava et al. (2014) JMLR 15:1929-1958.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without pip install
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from dropout_repro.data.dataset import MNISTDataModule
from dropout_repro.models.dropout_net import DropoutNet
from dropout_repro.training.trainer import Trainer
from dropout_repro.utils.config import DropoutConfig, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dropout reproduction — Srivastava et al. (2014)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mnist_3layer_1024.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override (overrides config seed)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cuda', 'cpu', or 'cuda:N' (overrides config)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Experiment name override (overrides config experiment.run_name)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: 1000 weight updates, smaller dataset. For quick iteration only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: build all components and validate setup, but do not train.",
    )
    parser.add_argument(
        "--no-dropout",
        action="store_true",
        help="Disable dropout (for baseline comparison). Overrides config.",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help=(
            "Phase 2 training: train on full 60K (train+val) after hyperparameter tuning. "
            "See Appendix B.1. Requires --resume pointing to best Phase 1 checkpoint."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    print(f"Loading config: {args.config}")
    config = DropoutConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        config.training.seed = args.seed
    if args.device is not None:
        config.hardware.device = args.device
    if args.run_name is not None:
        config.experiment.run_name = args.run_name
    if args.no_dropout:
        config.model.use_dropout = False
        config.model.use_max_norm = False  # typically disabled together for fair comparison
        print("WARNING: Dropout disabled (--no-dropout). Running baseline comparison.")
    if args.debug:
        config.training.n_weight_updates = 1_000
        config.training.log_interval = 100
        config.training.checkpoint_interval = 500
        config.data.val_size = 1_000
        config.experiment.run_name = config.experiment.run_name + "_debug"
        print("DEBUG MODE: 1,000 weight updates, reduced val set.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(config.training.seed, deterministic=config.hardware.deterministic)
    device = get_device(config.hardware.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_module = MNISTDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        val_size=config.data.val_size,
        num_workers=config.data.num_workers,
        seed=config.training.seed,
        mean=config.data.normalize_mean,
        std=config.data.normalize_std,
    )
    data_module.setup()

    if args.phase2:
        # Appendix B.1: "The validation set was then combined with the training set
        # and training was done for 1 million weight updates."
        train_loader = data_module.combined_dataloader()
        val_loader = None
        print("Phase 2 training: using full 60K (train+val) with no validation split.")
    else:
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    mc = config.model
    model = DropoutNet(
        input_dim=mc.input_dim,
        hidden_dims=mc.hidden_dims,
        num_classes=mc.num_classes,
        p_hidden=mc.p_hidden,
        p_input=mc.p_input,
        activation=mc.activation,
        use_dropout=mc.use_dropout,
    )
    print(model)

    # ------------------------------------------------------------------
    # Dry run: validate setup only
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\nDRY RUN: Validating setup...")
        model.eval()
        x_sample = next(iter(train_loader))[0][:4].to(device)
        model = model.to(device)
        with torch.no_grad():
            out = model(x_sample)
        print(f"  Input shape:  {tuple(x_sample.shape)}")
        print(f"  Output shape: {tuple(out.shape)}")
        print(f"  Output (logits sample): {out[0].cpu().numpy().round(3)}")
        print("\nDRY RUN PASSED — all components functional. Exiting without training.")
        return

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(model=model, config=config, device=device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    print("\nRunning final evaluation on test set...")
    test_loader = data_module.test_dataloader()
    test_result = trainer.evaluate(test_loader)

    print(f"\n{'='*50}")
    print("FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Test error rate: {test_result['error_rate']:.4f}%")
    print(f"  Test accuracy:   {100.0 - test_result['error_rate']:.4f}%")
    print(f"  Test loss:       {test_result['loss']:.6f}")

    if config.experiment.expected_test_error_pct is not None:
        expected = config.experiment.expected_test_error_pct
        delta = test_result["error_rate"] - expected
        status = "✓ PASS" if abs(delta) <= 0.3 else "✗ NEEDS REVIEW"
        print(f"\n  Paper target:    {expected:.2f}%")
        print(f"  Delta:           {delta:+.4f}%")
        print(f"  Status:          {status}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
