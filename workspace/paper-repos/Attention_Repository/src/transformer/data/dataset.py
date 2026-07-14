"""
data/dataset.py
===============
Translation dataset and token-based batch sampler.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 5.1 — "sentence pairs batched by approximate sequence length;
each training batch contained ~25000 source tokens and ~25000 target tokens."
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from transformer.data.tokenizer import BPETokenizer


class TranslationDataset(Dataset):
    """
    Parallel translation dataset loading pre-tokenized sentence pairs.

    Expects two parallel text files (one sentence per line):
      - {data_dir}/{split}.{src_lang}
      - {data_dir}/{split}.{tgt_lang}

    Args:
        src_file:  Path to source language text file.
        tgt_file:  Path to target language text file.
        tokenizer: BPETokenizer instance.
        max_len:   Maximum sequence length; longer examples are filtered out.
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        tokenizer: BPETokenizer,
        max_len: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

        src_path, tgt_path = Path(src_file), Path(tgt_file)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        if not tgt_path.exists():
            raise FileNotFoundError(f"Target file not found: {tgt_path}")

        print(f"Loading dataset from {src_path.name} / {tgt_path.name} ...")
        with open(src_path, encoding="utf-8") as f:
            src_lines = [l.strip() for l in f]
        with open(tgt_path, encoding="utf-8") as f:
            tgt_lines = [l.strip() for l in f]

        assert len(src_lines) == len(tgt_lines), (
            f"Source ({len(src_lines)}) and target ({len(tgt_lines)}) have different line counts."
        )

        # Tokenize and filter by max_len
        self.examples: List[Dict[str, List[int]]] = []
        filtered = 0
        for src, tgt in zip(src_lines, tgt_lines):
            src_ids = tokenizer.encode(src, add_eos=True)
            tgt_ids = tokenizer.encode(tgt, add_bos=True, add_eos=True)
            if len(src_ids) > max_len or len(tgt_ids) > max_len:
                filtered += 1
                continue
            self.examples.append({"src_ids": src_ids, "tgt_ids": tgt_ids})

        print(
            f"  {len(self.examples):,} examples loaded "
            f"({filtered:,} filtered for length > {max_len})."
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, List[int]]],
        pad_idx: int = 0,
    ) -> Dict[str, Tensor]:
        """
        Collate a list of examples into padded tensors.

        Args:
            batch:   List of dicts with 'src_ids' and 'tgt_ids'.
            pad_idx: Padding token id.

        Returns:
            Dict with 'src' [B, T_src], 'tgt_in' [B, T_tgt-1], 'tgt_out' [B, T_tgt-1].
        """
        src_ids = [ex["src_ids"] for ex in batch]
        tgt_ids = [ex["tgt_ids"] for ex in batch]

        src_lens = [len(s) for s in src_ids]
        tgt_lens = [len(t) for t in tgt_ids]

        max_src = max(src_lens)
        max_tgt = max(tgt_lens)

        # Pad src
        src_padded = torch.full((len(batch), max_src), pad_idx, dtype=torch.long)
        for i, ids in enumerate(src_ids):
            src_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        # Pad tgt — split into decoder input (drop last) and labels (drop first)
        tgt_padded = torch.full((len(batch), max_tgt), pad_idx, dtype=torch.long)
        for i, ids in enumerate(tgt_ids):
            tgt_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        tgt_in = tgt_padded[:, :-1]   # BOS ... (decoder input)
        tgt_out = tgt_padded[:, 1:]   # ... EOS (labels)

        return {"src": src_padded, "tgt_in": tgt_in, "tgt_out": tgt_out}


class TokenBatchSampler(Sampler):
    """
    Token-based batch sampler matching the paper's batching strategy.

    Paper: Section 5.1 — "sentence pairs batched together by approximate
    sequence length; each training batch contained ~25000 source tokens
    and ~25000 target tokens."

    Groups examples into buckets by length, then greedily fills each batch
    up to max_tokens (total source + target tokens).

    Args:
        lengths:    List of (src_len, tgt_len) tuples.
        max_tokens: Maximum total token count per batch.
        shuffle:    Shuffle within buckets each epoch.
        seed:       Random seed for reproducibility.
    """

    def __init__(
        self,
        lengths: List[tuple],
        max_tokens: int = 25000,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        # Sort by source length (reduces padding)
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda i: self.lengths[i][0])

        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            # Shuffle within buckets of 1000 to preserve approximate sorting
            bucket_size = 1000
            buckets = [indices[i : i + bucket_size] for i in range(0, len(indices), bucket_size)]
            for bucket in buckets:
                rng.shuffle(bucket)
            indices = [idx for bucket in buckets for idx in bucket]

        # Greedily fill batches up to max_tokens
        batch: List[int] = []
        max_len_in_batch = 0

        for idx in indices:
            src_len, tgt_len = self.lengths[idx]
            candidate_max = max(max_len_in_batch, src_len + tgt_len)
            # Estimate tokens: candidate_max * (len(batch) + 1)
            if batch and candidate_max * (len(batch) + 1) > self.max_tokens:
                yield batch
                batch = []
                max_len_in_batch = 0
            batch.append(idx)
            max_len_in_batch = max(max_len_in_batch, src_len + tgt_len)

        if batch:
            yield batch

    def __len__(self) -> int:
        # Approximate — actual count depends on lengths
        total_tokens = sum(s + t for s, t in self.lengths)
        return max(1, total_tokens // self.max_tokens)
