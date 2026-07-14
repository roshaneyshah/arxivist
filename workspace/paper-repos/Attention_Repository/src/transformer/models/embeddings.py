"""
models/embeddings.py
====================
Token embeddings and sinusoidal positional encoding.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
  Section 3.4 — Embeddings and Softmax
  Section 3.5 — Positional Encoding (sinusoidal)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """
    Learned token embedding.

    Paper: Section 3.4 — embedding weights are multiplied by sqrt(d_model).
    Weight matrix is shared with OutputEmbedding and the pre-softmax linear
    transformation (3-way weight tying; ASSUMED: SIR confidence 0.82).

    Args:
        vocab_size: Vocabulary size.
        d_model:    Model dimensionality (default 512).
    """

    def __init__(self, vocab_size: int, d_model: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: token ids [B, T]  (int64)

        Returns:
            [B, T, d_model]  — scaled by sqrt(d_model) per Section 3.4
        """
        assert x.dim() == 2, f"Expected [B, T], got {x.shape}"
        return self.embedding(x) * math.sqrt(self.d_model)

    def __repr__(self) -> str:
        return (
            f"TokenEmbedding(vocab_size={self.embedding.num_embeddings}, "
            f"d_model={self.d_model})"
        )


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.

    Paper: Section 3.5, formulas:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The PE is added to the input embeddings at the bottom of both encoder and
    decoder stacks.  Dropout is applied to the sum.

    Ordering: PE added to embeddings, then dropout applied to the sum.
    # TODO: verify ordering — SIR ambiguity #4, confidence 0.88

    Args:
        d_model: Model dimensionality (default 512).
        max_len: Maximum sequence length to pre-compute PE for (default 5000).
        dropout: Dropout applied after adding PE.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the sinusoidal table once
        pe = torch.zeros(max_len, d_model)          # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        # 10000^(2i/d_model) — computed in log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices → sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices  → cosine
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]  with positional encoding added, then dropout applied
        """
        assert x.dim() == 3, f"Expected [B, T, d_model], got {x.shape}"
        x = x + self.pe[:, : x.size(1), :]  # broadcast over batch
        return self.dropout(x)

    def __repr__(self) -> str:
        return f"PositionalEncoding(d_model={self.pe.size(-1)}, dropout={self.dropout.p})"
