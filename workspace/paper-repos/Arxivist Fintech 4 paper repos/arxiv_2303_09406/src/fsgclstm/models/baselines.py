"""
models/baselines.py
===================
Baseline models for comparison against FS-GCLSTM.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section IV.D

Baselines:
  - FCL: Four-layer fully connected network
  - LSTM: Two-layer LSTM with MLP head
  - GConvGRU: Chebyshev spectral GRU (reference: Seo et al. 2018)
  - ARIMA: Implemented separately via pmdarima (not a PyTorch model)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class FCLBaseline(nn.Module):
    """Fully Connected Layer baseline.

    Paper Section IV.D: "Four-layer fully connected network with layer sizes
    (10*nin, 5*nin, 10*nin, 10*nout). Input is a flattened (10, nin) return tensor."

    Args:
        n_in: Number of input stocks
        n_out: Number of output stocks
        seq_len: Input sequence length (paper uses 10 for FCL)
    """

    def __init__(self, n_in: int, n_out: int, seq_len: int = 10) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        flat_in = seq_len * n_in
        self.net = nn.Sequential(
            nn.Linear(flat_in, 10 * n_in),
            nn.ReLU(),
            nn.Linear(10 * n_in, 5 * n_in),
            nn.ReLU(),
            nn.Linear(5 * n_in, 10 * n_in),
            nn.ReLU(),
            nn.Linear(10 * n_in, 10 * n_out),
            nn.ReLU(),
            nn.Linear(10 * n_out, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, n_in] recent returns, will be flattened

        Returns:
            Predicted returns [n_out]
        """
        assert x.dim() == 2, f"Expected [seq_len, n_in], got {x.shape}"
        flat = x.reshape(1, -1)   # [1, seq_len*n_in]
        return self.net(flat).squeeze(0)

    def __repr__(self) -> str:
        return f"FCLBaseline(n_in={self.n_in}, n_out={self.n_out})"


class LSTMBaseline(nn.Module):
    """Two-layer LSTM baseline.

    Paper Section IV.D: "Two-layer LSTM with hidden sizes (60, 60, 6),
    followed by an MLP with layers (6*nin, 10*nin, 1*nout). Cell states are
    carried forward within the rolling window."

    Args:
        n_in: Number of input stocks
        n_out: Number of output stocks
        input_size: Feature dimension per stock per time step (default: 1 for returns)
    """

    def __init__(self, n_in: int, n_out: int, input_size: int = 1) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        # Hidden sizes from paper: (60, 60, 6) — interpreted as two LSTM layers (60→60) + projection (60→6)
        self.lstm = nn.LSTM(
            input_size=n_in * input_size,
            hidden_size=60,
            num_layers=2,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(60, 6 * n_in),
            nn.ReLU(),
            nn.Linear(6 * n_in, 10 * n_in),
            nn.ReLU(),
            nn.Linear(10 * n_in, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, n_in] recent returns

        Returns:
            Predicted returns [n_out]
        """
        assert x.dim() == 2, f"Expected [seq_len, n_in], got {x.shape}"
        x_in = x.unsqueeze(0)                              # [1, seq_len, n_in]
        out, _ = self.lstm(x_in)                           # [1, seq_len, 60]
        last = out[:, -1, :]                               # [1, 60] — last time step
        return self.mlp(last).squeeze(0)                   # [n_out]

    def __repr__(self) -> str:
        return f"LSTMBaseline(n_in={self.n_in}, n_out={self.n_out})"
