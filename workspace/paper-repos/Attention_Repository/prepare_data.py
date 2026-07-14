"""
prepare_data.py
===============
Download WMT14 EN-DE data and train a shared SentencePiece BPE tokenizer.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 5.1 — WMT 2014 EN-DE, shared BPE vocab of ~37000 tokens.

Usage:
    python prepare_data.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformer.utils.config import TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess WMT14 EN-DE data.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def check_exists(path: Path, description: str) -> bool:
    if path.exists():
        print(f"  [skip] {description} already exists at {path}")
        return True
    return False


def main() -> None:
    args = parse_args()
    config = TransformerConfig.from_yaml(args.config)
    dc = config.data

    data_dir = Path(args.output_dir or dc.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    sp_model = Path(dc.sp_model_path)

    print("=" * 60)
    print("WMT14 EN-DE Data Preparation")
    print(f"  Output dir:    {data_dir}")
    print(f"  Vocab size:    {dc.vocab_size}")
    print(f"  SP model path: {sp_model}")
    print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Download via HuggingFace datasets
    # ----------------------------------------------------------------
    raw_dir = data_dir / "raw"
    if not check_exists(raw_dir / "train.en", "raw training data"):
        print("\nDownloading WMT14 EN-DE via HuggingFace datasets...")
        print("  Estimated download: ~1.6 GB compressed, ~4.5M sentence pairs.")
        try:
            import datasets as hf_datasets
            dataset = hf_datasets.load_dataset("wmt14", "de-en", trust_remote_code=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            for split_name, split_key in [
                (dc.train_prefix, "train"),
                (dc.val_prefix, "validation"),
                (dc.test_prefix, "test"),
            ]:
                src_out = raw_dir / f"{split_name}.{dc.src_lang}"
                tgt_out = raw_dir / f"{split_name}.{dc.tgt_lang}"
                if check_exists(src_out, f"{split_key} src"):
                    continue
                print(f"  Writing {split_key} split...")
                with open(src_out, "w", encoding="utf-8") as fsrc, \
                     open(tgt_out, "w", encoding="utf-8") as ftgt:
                    for ex in dataset[split_key]:
                        fsrc.write(ex["translation"][dc.src_lang] + "\n")
                        ftgt.write(ex["translation"][dc.tgt_lang] + "\n")
                print(f"    {split_key}: {src_out.name}, {tgt_out.name}")
        except Exception as e:
            print(f"\nERROR downloading data: {e}")
            print("Please manually place data files at:")
            print(f"  {raw_dir}/train.en, train.de")
            print(f"  {raw_dir}/{dc.val_prefix}.en, {dc.val_prefix}.de")
            print(f"  {raw_dir}/{dc.test_prefix}.en, {dc.test_prefix}.de")
            sys.exit(1)

    # ----------------------------------------------------------------
    # 2. Train SentencePiece BPE model on training corpus
    # ----------------------------------------------------------------
    if not check_exists(sp_model, "SentencePiece model"):
        print(f"\nTraining SentencePiece BPE (vocab_size={dc.vocab_size})...")
        sp_model.parent.mkdir(parents=True, exist_ok=True)
        try:
            import sentencepiece as spm
            # Combine src and tgt for shared vocab training
            combined = data_dir / "train_combined.txt"
            with open(combined, "w", encoding="utf-8") as fout:
                for lang in [dc.src_lang, dc.tgt_lang]:
                    with open(raw_dir / f"{dc.train_prefix}.{lang}", encoding="utf-8") as fin:
                        fout.write(fin.read())
            spm.SentencePieceTrainer.train(
                input=str(combined),
                model_prefix=str(sp_model.with_suffix("")),
                vocab_size=dc.vocab_size,
                model_type="bpe",
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3,
                pad_piece="<pad>",
                bos_piece="<s>",
                eos_piece="</s>",
                unk_piece="<unk>",
                character_coverage=0.9995,
            )
            combined.unlink()
            print(f"  SentencePiece model saved: {sp_model}")
        except Exception as e:
            print(f"ERROR training SentencePiece: {e}")
            sys.exit(1)

    # ----------------------------------------------------------------
    # 3. Copy processed files to final data_dir
    # ----------------------------------------------------------------
    print("\nCopying processed files to data dir...")
    import shutil
    for split_name in [dc.train_prefix, dc.val_prefix, dc.test_prefix]:
        for lang in [dc.src_lang, dc.tgt_lang]:
            src = raw_dir / f"{split_name}.{lang}"
            dst = data_dir / f"{split_name}.{lang}"
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)

    print("\nData preparation complete.")
    print(f"  Data dir: {data_dir}")
    print(f"  SP model: {sp_model}")
    print("\nReady to train:")
    print(f"  python train.py --config {args.config}")


if __name__ == "__main__":
    main()
