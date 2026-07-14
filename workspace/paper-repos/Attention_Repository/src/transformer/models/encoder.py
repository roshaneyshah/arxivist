"""
models/encoder.py
=================
Transformer Encoder: EncoderLayer and Encoder stack.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 3.1 — Encoder and Decoder Stacks

Each EncoderLayer has two sub-layers:
  1. Multi-Head Self-Attention
  2. Position-wise Feed-Forward Network
Each sub-layer uses a residual connection and layer normalization:
  output = LayerNorm(x + Dropout(Sublayer(x)))
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from transformer.models.attention import MultiHeadAttention
from transformer.models.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    Paper: Section 3.1 — two sub-layers with residual + LayerNorm.
    LayerNorm(x + Sublayer(x))

    Args:
        d_model: Model dimensionality.
        h:       Number of attention heads.
        d_ff:    Feed-forward inner dimensionality.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:        [B, T, d_model]
            src_mask: Optional [B, 1, 1, T] padding mask.

        Returns:
            [B, T, d_model]
        """
        assert x.dim() == 3, f"Expected [B, T, d_model], got {x.shape}"

        # Sub-layer 1: Multi-Head Self-Attention + residual + LayerNorm
        attn_out, _ = self.self_attn(q=x, k=x, v=x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Sub-layer 2: Position-wise FFN + residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

    def __repr__(self) -> str:
        return (
            f"EncoderLayer(d_model={self.self_attn.d_model}, "
            f"h={self.self_attn.h}, d_ff={self.ffn.d_ff})"
        )


class Encoder(nn.Module):
    """
    Transformer Encoder Stack of N identical EncoderLayers.

    Paper: Section 3.1 — stack of N=6 identical layers.

    Args:
        d_model: Model dimensionality (default 512).
        N:       Number of encoder layers (default 6).
        h:       Attention heads (default 8).
        d_ff:    FFN inner dim (default 2048).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)  # final normalization after stack

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:    [B, T, d_model]  (embeddings + positional encoding already applied)
            mask: Optional [B, 1, 1, T] padding mask.

        Returns:
            encoder memory [B, T, d_model]
        """
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x)

    def __repr__(self) -> str:
        N = len(self.layers)
        d = self.layers[0].self_attn.d_model
        return f"Encoder(N={N}, d_model={d})"
