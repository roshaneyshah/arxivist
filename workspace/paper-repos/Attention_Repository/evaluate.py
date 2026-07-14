"""
evaluate.py
===========
Evaluation entrypoint: run beam search on a test set and report BLEU.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 6.1 — beam_size=4, length_penalty_alpha=0.6

Usage:
    python evaluate.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_averaged.pt
    python evaluate.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_averaged.pt --split val
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformer.data.dataset import TranslationDataset, TokenBatchSampler
from transformer.data.tokenizer import BPETokenizer
from transformer.evaluation.metrics import BLEUEvaluator
from transformer.models.transformer import Transformer
from transformer.utils.config import TransformerConfig, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer (BLEU + PPL).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output-file", type=str, default="results/translations.txt")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit evaluation batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TransformerConfig.from_yaml(args.config)
    set_seed(config.training.seed)
    device = get_device(config.hardware)

    dc = config.data
    split_prefix = dc.test_prefix if args.split == "test" else dc.val_prefix

    # Load tokenizer and dataset
    tokenizer = BPETokenizer(dc.sp_model_path)
    src_file = f"{dc.data_dir}/{split_prefix}.{dc.src_lang}"
    tgt_file = f"{dc.data_dir}/{split_prefix}.{dc.tgt_lang}"

    dataset = TranslationDataset(src_file, tgt_file, tokenizer, max_len=dc.max_seq_len)
    lengths = [(len(ex["src_ids"]), len(ex["tgt_ids"])) for ex in dataset.examples]
    sampler = TokenBatchSampler(lengths, max_tokens=config.training.max_tokens_per_batch, shuffle=False)
    collate_fn = functools.partial(TranslationDataset.collate_fn, pad_idx=dc.pad_idx)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn,
                        num_workers=config.hardware.dataloader_num_workers)

    # Load model
    model = Transformer(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(model)

    # Evaluate
    evaluator = BLEUEvaluator(config.evaluation)
    print(f"\nRunning evaluation on {args.split} split...")
    results = evaluator.evaluate_dataset(
        model, loader, tokenizer, device, max_batches=args.max_batches
    )

    print(f"\n{'='*40}")
    print(f"BLEU:        {results['bleu']:.2f}")
    print(f"Perplexity:  {results['ppl']:.4f}")
    print(f"{'='*40}")

    # Paper's reported results for reference:
    print("\nPaper reported results (Table 2):")
    print("  Transformer (base): EN-DE BLEU = 27.3")
    print("  Transformer (big):  EN-DE BLEU = 28.4")
    print("  Transformer (base): EN-FR BLEU = 38.1")
    print("  Transformer (big):  EN-FR BLEU = 41.8")

    # Save results
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    results_file = Path(args.output_file).with_suffix(".json")
    with open(results_file, "w") as f:
        json.dump({"bleu": results["bleu"], "ppl": results["ppl"],
                   "checkpoint": args.checkpoint, "split": args.split}, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
