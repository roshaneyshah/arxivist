"""
utils/masking.py
================
Mask construction utilities for the Transformer.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 3.1 — masking in decoder self-attention to preserve auto-regressive property.
"""

from __future__ import annotations

import torch
from torch import Tensor


class MaskFactory:
    """
    Factory for creating boolean attention masks.

    Masks use convention: True (1) = attend, False (0) = ignore.
    ScaledDotProductAttention fills positions where mask==0 with -1e9.
    """

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """
        Causal (lower-triangular) mask for decoder self-attention.

        Prevents position i from attending to positions j > i.
        Implemented by masking out illegal positions before softmax — Section 3.1.

        Args:
            seq_len: Target sequence length.
            device:  Torch device.

        Returns:
            [1, 1, T, T] bool tensor — lower-triangular (True = attend).
        """
        # torch.tril: retain lower triangle (including diagonal)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    @staticmethod
    def make_padding_mask(token_ids: Tensor, pad_idx: int) -> Tensor:
        """
        Source padding mask — prevents attention to <pad> tokens.

        Args:
            token_ids: [B, T] integer token ids.
            pad_idx:   Integer id of the padding token.

        Returns:
            [B, 1, 1, T] bool tensor — True where token is NOT padding.

        Note: shape [B, 1, 1, T] broadcasts correctly across heads and query positions.
        # TODO: padding mask shape not explicitly specified in paper — inferred from
        # standard practice (SIR confidence 0.75).
        """
        return (token_ids != pad_idx).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]

    @staticmethod
    def make_tgt_mask(tgt_ids: Tensor, pad_idx: int) -> Tensor:
        """
        Combined causal + padding mask for decoder self-attention.

        Args:
            tgt_ids: [B, T_tgt] target token ids.
            pad_idx: Padding token id.

        Returns:
            [B, 1, T_tgt, T_tgt] bool tensor.
        """
        T = tgt_ids.size(1)
        device = tgt_ids.device
        causal = MaskFactory.make_causal_mask(T, device)                    # [1, 1, T, T]
        padding = MaskFactory.make_padding_mask(tgt_ids, pad_idx)           # [B, 1, 1, T]
        return causal & padding                                             # [B, 1, T, T]
