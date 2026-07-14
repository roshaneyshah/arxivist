"""LSTM baseline (Appendix A.3, arXiv:2401.06249).

Per the paper: only spot volatility and co-volatility time series are used as
input (not volatility-of-volatility), since the authors report that including
the latter worsens LSTM performance due to dimensionality.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Stacked LSTM baseline for spot volatility forecasting (App. A.3).

    Args:
        input_dim: Per-timestep input feature dimension (volatilities + co-volatilities
            for all assets, no lag stacking — LSTM consumes the lag dimension as sequence).
        hidden_dims: Hidden sizes per LSTM layer, e.g. ``[400, 200]`` (Table 10).
        output_dim: 1 (single-step) or 14 (multi-step functional forecast).
        dropout: Dropout between LSTM layers (Table 10 "Dropout (Architecture)").
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers = []
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=h_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                )
            )
            in_dim = h_dim
        self.lstm_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"LSTMBaseline(layers={len(self.lstm_layers)}, output_dim={self.output_layer.out_features})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sequence, shape ``[B, T, input_dim]`` where T is the lag window.

        Returns:
            Predictions, shape ``[B, output_dim]``, taken from the last timestep.
        """
        assert x.dim() == 3, f"Expected [B, T, input_dim], got {x.shape}"
        h = x
        for lstm in self.lstm_layers:
            h, _ = lstm(h)
            h = self.dropout(h)
        last_hidden = h[:, -1, :]  # [B, hidden_dim]
        return self.output_layer(last_hidden)
