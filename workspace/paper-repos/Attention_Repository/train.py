"""
train.py
========
Training entrypoint for the Transformer model.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Usage:
    python train.py --config configs/base.yaml
    python train.py --config configs/base.yaml --resume checkpoints/checkpoint_step50000.pt
    python train.py --config configs/base.yaml --debug
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformer.data.dataset import TokenBatchSampler, TranslationDataset
from transformer.data.tokenizer import BPETokenizer
from transformer.models.transformer import Transformer
from transformer.training.trainer import TransformerTrainer
from transformer.utils.config import TransformerConfig, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer (Vaswani et al. 2017) from scratch."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--output-dir", type=str, default="checkpoints/", help="Checkpoint output dir.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--debug", action="store_true", help="Reduce data/steps for quick local test.")
    parser.add_argument("--dry-run", action="store_true", help="Build all components, skip training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    return parser.parse_args()


def build_data_loader(
    config: TransformerConfig,
    split: str,
    debug: bool = False,
) -> DataLoader:
    """Build a DataLoader for the given split."""
    dc = config.data

    src_file = f"{dc.data_dir}/{split}.{dc.src_lang}"
    tgt_file = f"{dc.data_dir}/{split}.{dc.tgt_lang}"

    tokenizer = BPETokenizer(dc.sp_model_path)

    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        tokenizer=tokenizer,
        max_len=dc.max_seq_len,
    )

    if debug:
        # Use only first 200 examples in debug mode
        dataset.examples = dataset.examples[:200]

    lengths = [(len(ex["src_ids"]), len(ex["tgt_ids"])) for ex in dataset.examples]
    max_tokens = 1000 if debug else config.training.max_tokens_per_batch

    sampler = TokenBatchSampler(lengths=lengths, max_tokens=max_tokens, shuffle=(split == dc.train_prefix))

    collate_fn = functools.partial(TranslationDataset.collate_fn, pad_idx=dc.pad_idx)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config.hardware.dataloader_num_workers,
        pin_memory=(config.hardware.device == "cuda"),
    )


def main() -> None:
    args = parse_args()

    # Load config
    config = TransformerConfig.from_yaml(args.config)

    # Seed override
    seed = args.seed if args.seed is not None else config.training.seed
    set_seed(seed, deterministic=config.hardware.deterministic)
    print(f"Seed: {seed}")

    # Debug: reduce steps
    if args.debug:
        config.training.max_steps = 200
        config.training.log_every_steps = 10
        config.training.checkpoint_every_steps = 100
        print("DEBUG mode: max_steps=200, reduced batch size.")

    device = get_device(config.hardware)
    print(f"Device: {device}")

    # Build model
    model = Transformer(config)
    print(model)

    if args.dry_run:
        print("DRY RUN: all components built successfully. Exiting without training.")
        return

    # Build data loaders
    print("\nLoading training data...")
    train_loader = build_data_loader(config, config.data.train_prefix, debug=args.debug)
    print("Loading validation data...")
    val_loader = build_data_loader(config, config.data.val_prefix, debug=args.debug)

    # Initialize trainer
    trainer = TransformerTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )

    # Resume from checkpoint if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # WandB (optional)
    if args.wandb:
        try:
            import wandb
            wandb.init(project="attention-is-all-you-need", config=config.__dict__)
        except ImportError:
            print("WARNING: wandb not installed. Skipping.")

    # Train
    trainer.train(train_loader, val_loader)

    # After training: average last N checkpoints (Section 6.1)
    if len(trainer._checkpoint_paths) >= 2:
        n = config.training.avg_last_n_checkpoints
        avg_paths = [str(p) for p in trainer._checkpoint_paths[-n:]]
        print(f"\nAveraging last {len(avg_paths)} checkpoints...")
        avg_state = trainer.average_checkpoints(avg_paths)
        avg_path = Path(args.output_dir) / "checkpoint_averaged.pt"
        torch.save({"model_state_dict": avg_state, "config": config}, avg_path)
        print(f"Averaged checkpoint saved to {avg_path}")


if __name__ == "__main__":
    main()
