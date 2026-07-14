"""
models/attention.py
===================
Scaled Dot-Product Attention and Multi-Head Attention.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Sections 3.2.1 and 3.2.2  |  Equations 1 and MultiHead formula
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.

    Paper: Section 3.2.1, Equation 1:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        dropout: Dropout rate on attention weights.
                 # TODO: verify — paper explicitly states residual+embedding dropout (Section 5.4)
                 # but attention dropout is assumed standard practice (SIR confidence 0.78)
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q:    Query tensor  [B, h, T_q, d_k]
            k:    Key tensor    [B, h, T_k, d_k]
            v:    Value tensor  [B, h, T_v, d_v]  (T_v == T_k)
            mask: Optional boolean mask [B, 1, T_q, T_k] or [B, 1, 1, T_k].
                  Positions where mask==False are set to -1e9 (effectively -inf).

        Returns:
            output:       [B, h, T_q, d_v]
            attn_weights: [B, h, T_q, T_k]
        """
        assert q.dim() == 4, f"Expected q: [B, h, T_q, d_k], got {q.shape}"

        d_k = q.size(-1)

        # Eq. 1: score = Q K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B, h, T_q, T_k]

        # Causal or padding mask: fill illegal positions with -inf before softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)  # [B, h, T_q, T_k]

        # Attention dropout (see SIR ambiguity note in docstring above)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum over values
        output = torch.matmul(attn_weights, v)  # [B, h, T_q, d_v]

        return output, attn_weights

    def __repr__(self) -> str:
        return f"ScaledDotProductAttention(dropout={self.dropout.p})"


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.

    Paper: Section 3.2.2:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)

    Projection dimensions (base model):
        W^Q_i, W^K_i: [d_model, d_k] = [512, 64]
        W^V_i:        [d_model, d_v] = [512, 64]
        W^O:          [h*d_v, d_model] = [512, 512]

    Args:
        d_model: Model dimensionality (default 512).
        h:       Number of attention heads (default 8).
        dropout: Dropout rate applied in attention and after projection.
    """

    def __init__(self, d_model: int = 512, h: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # 64 for base model
        self.d_v = d_model // h  # 64 for base model

        # Projection matrices W^Q, W^K, W^V, W^O — Section 3.2.2
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # h * d_k = d_model
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # W^O

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization — ASSUMED (Section not specified, SIR confidence 0.70)."""
        for layer in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q:    [B, T_q, d_model]
            k:    [B, T_k, d_model]
            v:    [B, T_v, d_model]
            mask: Optional [B, 1, T_q, T_k] or [B, 1, 1, T_k]

        Returns:
            output:       [B, T_q, d_model]
            attn_weights: [B, h, T_q, T_k]
        """
        assert q.dim() == 3, f"Expected q: [B, T, D], got {q.shape}"
        B, T_q, _ = q.size()
        T_k = k.size(1)

        # Linear projections then split into h heads
        # [B, T, d_model] → [B, T, h, d_k] → [B, h, T, d_k]
        q_proj = self.w_q(q).view(B, T_q, self.h, self.d_k).transpose(1, 2)  # [B, h, T_q, d_k]
        k_proj = self.w_k(k).view(B, T_k, self.h, self.d_k).transpose(1, 2)  # [B, h, T_k, d_k]
        v_proj = self.w_v(v).view(B, T_k, self.h, self.d_v).transpose(1, 2)  # [B, h, T_k, d_v]

        # Scaled dot-product attention across all heads in parallel
        context, attn_weights = self.attention(q_proj, k_proj, v_proj, mask)  # [B, h, T_q, d_v]

        # Concat heads: [B, h, T_q, d_v] → [B, T_q, h*d_v == d_model]
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # Output projection W^O
        output = self.w_o(context)  # [B, T_q, d_model]

        return output, attn_weights

    def __repr__(self) -> str:
        return f"MultiHeadAttention(d_model={self.d_model}, h={self.h}, d_k={self.d_k})"
