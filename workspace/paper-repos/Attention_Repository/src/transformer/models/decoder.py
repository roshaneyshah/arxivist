"""
models/decoder.py
=================
Transformer Decoder: DecoderLayer and Decoder stack.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 3.1 — Encoder and Decoder Stacks

Each DecoderLayer has three sub-layers:
  1. Masked Multi-Head Self-Attention (causal mask)
  2. Multi-Head Cross-Attention over encoder output
  3. Position-wise Feed-Forward Network
Each sub-layer: LayerNorm(x + Dropout(Sublayer(x)))
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from transformer.models.attention import MultiHeadAttention
from transformer.models.feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.

    Paper: Section 3.1 — three sub-layers with residual + LayerNorm.

    Args:
        d_model: Model dimensionality.
        h:       Number of attention heads.
        d_ff:    Feed-forward inner dimensionality.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # Sub-layer 1: Masked self-attention (Section 3.1 — prevent leftward info flow)
        self.self_attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        # Sub-layer 2: Cross-attention over encoder memory
        self.cross_attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        # Sub-layer 3: FFN
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:        [B, T_tgt, d_model] — decoder input (shifted right)
            memory:   [B, T_src, d_model] — encoder output
            src_mask: Optional [B, 1, 1, T_src] — source padding mask for cross-attn
            tgt_mask: Optional [B, 1, T_tgt, T_tgt] — causal + target padding mask

        Returns:
            [B, T_tgt, d_model]
        """
        assert x.dim() == 3, f"Expected [B, T_tgt, d_model], got {x.shape}"
        assert memory.dim() == 3, f"Expected [B, T_src, d_model], got {memory.shape}"

        # Sub-layer 1: Masked Multi-Head Self-Attention
        # Q=K=V=x, masked to prevent attending to future positions — Section 3.1
        self_attn_out, _ = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        # Sub-layer 2: Multi-Head Cross-Attention
        # Q from decoder, K/V from encoder memory — Section 3.1, Section 3.2.3
        cross_attn_out, _ = self.cross_attn(q=x, k=memory, v=memory, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Sub-layer 3: Position-wise FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x

    def __repr__(self) -> str:
        return (
            f"DecoderLayer(d_model={self.self_attn.d_model}, "
            f"h={self.self_attn.h}, d_ff={self.ffn.d_ff})"
        )


class Decoder(nn.Module):
    """
    Transformer Decoder Stack of N identical DecoderLayers.

    Paper: Section 3.1 — stack of N=6 identical layers.

    Args:
        d_model: Model dimensionality (default 512).
        N:       Number of decoder layers (default 6).
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
            [DecoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:        [B, T_tgt, d_model]
            memory:   [B, T_src, d_model]
            src_mask: Optional [B, 1, 1, T_src]
            tgt_mask: Optional [B, 1, T_tgt, T_tgt]

        Returns:
            [B, T_tgt, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)

    def __repr__(self) -> str:
        N = len(self.layers)
        d = self.layers[0].self_attn.d_model
        return f"Decoder(N={N}, d_model={d})"
