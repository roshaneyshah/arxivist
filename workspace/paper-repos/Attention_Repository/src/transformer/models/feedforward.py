"""
models/feedforward.py
=====================
Position-wise Feed-Forward Network.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 3.3, Equation 2:
    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    Applied identically and independently to each position in the sequence.
    Consists of two linear transformations with a ReLU activation between them.

    Paper: Section 3.3, Eq. 2:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    Dimensions (base model):
        d_model = 512  (input/output)
        d_ff    = 2048 (inner layer)

    This can be interpreted as two 1D convolutions with kernel size 1.

    Args:
        d_model: Input and output dimensionality (default 512).
        d_ff:    Inner layer dimensionality (default 2048).
        dropout: Dropout rate applied after first linear + ReLU.
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w_1 = nn.Linear(d_model, d_ff)   # W_1, b_1
        self.w_2 = nn.Linear(d_ff, d_model)   # W_2, b_2
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization — ASSUMED (SIR confidence 0.70)."""
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            [B, T, d_model]
        """
        assert x.dim() == 3, f"Expected [B, T, d_model], got {x.shape}"
        # Eq. 2: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

    def __repr__(self) -> str:
        return f"PositionwiseFeedForward(d_model={self.d_model}, d_ff={self.d_ff})"
