"""
training/losses.py
==================
Label-smoothed cross-entropy loss.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 5.4 — Label Smoothing:
    "employed label smoothing of value ε_ls = 0.1"
    "This hurts perplexity, as the model learns to be more unsure,
     but improves accuracy and BLEU score."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Paper: Section 5.4, ε_ls = 0.1.
    Instead of one-hot targets, smoothed targets distribute ε_ls / (V-1)
    probability mass uniformly across all non-target tokens.

    Args:
        vocab_size:   Vocabulary size V.
        smoothing:    Label smoothing factor ε (default 0.1 per paper).
        ignore_index: Token id to ignore in loss (padding). Default -100.
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing  # probability mass on correct token

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits:  [B, T, vocab_size]  raw model output (pre-softmax)
            targets: [B, T]              integer token ids

        Returns:
            Scalar mean loss over non-padding positions.
        """
        assert logits.dim() == 3, f"Expected [B, T, V], got {logits.shape}"
        assert targets.dim() == 2, f"Expected [B, T], got {targets.shape}"

        B, T, V = logits.shape

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, V)    # [B*T, V]
        targets_flat = targets.reshape(-1)     # [B*T]

        # Create smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(logits_flat, self.smoothing / (V - 1))
            smooth_targets.scatter_(1, targets_flat.unsqueeze(1).clamp(min=0), self.confidence)

            # Zero out padding positions entirely (they won't count toward loss)
            pad_mask = targets_flat.eq(self.ignore_index)
            smooth_targets[pad_mask] = 0.0

        # Log-softmax + KL divergence = cross-entropy with smoothed targets
        log_probs = F.log_softmax(logits_flat, dim=-1)  # [B*T, V]
        loss = -(smooth_targets * log_probs).sum(dim=-1)  # [B*T]

        # Mask out padding positions
        non_pad = (~pad_mask).float()
        loss = (loss * non_pad).sum() / non_pad.sum().clamp(min=1)

        return loss

    def __repr__(self) -> str:
        return (
            f"LabelSmoothedCrossEntropy(smoothing={self.smoothing}, "
            f"ignore_index={self.ignore_index})"
        )
