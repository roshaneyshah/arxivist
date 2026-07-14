"""
translate.py
============
Interactive single-sentence translation using beam search.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 6.1 — inference via beam search.

Usage:
    python translate.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_averaged.pt
    python translate.py --config configs/base.yaml --checkpoint ... --src "The cat sat on the mat."
    echo "Hello world" | python translate.py --config configs/base.yaml --checkpoint ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformer.data.tokenizer import BPETokenizer
from transformer.evaluation.metrics import beam_search_decode
from transformer.models.transformer import Transformer
from transformer.utils.config import TransformerConfig, get_device, set_seed
from transformer.utils.masking import MaskFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a sentence with a trained Transformer.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--src", type=str, default=None, help="Source sentence. Reads stdin if omitted.")
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Length penalty alpha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TransformerConfig.from_yaml(args.config)
    set_seed(config.training.seed)
    device = get_device(config.hardware)

    beam_size = args.beam_size or config.evaluation.beam_size
    alpha = args.alpha or config.evaluation.length_penalty_alpha

    # Load tokenizer
    tokenizer = BPETokenizer(config.data.sp_model_path)

    # Load model
    model = Transformer(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Read input
    if args.src:
        sentences = [args.src]
    else:
        print("Enter source sentences (Ctrl+D to finish):", file=sys.stderr)
        sentences = [line.strip() for line in sys.stdin if line.strip()]

    for sentence in sentences:
        src_ids = tokenizer.encode(sentence, add_eos=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = MaskFactory.make_padding_mask(src, tokenizer.pad_id)

        pred_ids = beam_search_decode(
            model, src, src_mask, tokenizer,
            beam_size=beam_size,
            max_len_offset=config.evaluation.max_decode_len_offset,
            length_penalty_alpha=alpha,
            device=device,
        )
        translation = tokenizer.decode(pred_ids)
        print(translation)


if __name__ == "__main__":
    main()
