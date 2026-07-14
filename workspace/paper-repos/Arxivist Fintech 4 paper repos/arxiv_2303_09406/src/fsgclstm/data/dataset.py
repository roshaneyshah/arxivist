"""
data/dataset.py
===============
Rolling-window graph dataset for FS-GCLSTM training.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section IV.C

Rolling-window strategy:
  - Initial window: 3,000 trading days (70% train / 20% val / 10% test)
  - After each iteration: advance by 300 days
  - Temporal ordering preserved (no shuffling)
  - Node features: d-day rolling window of daily returns (d=60 default)
"""
from __future__ import annotations
from typing import Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class StockReturnDataset(Dataset):
    """Dataset for a single rolling window (train, val, or test split).

    Each sample is (x_seq, adj, y) where:
      x_seq: [input_seq_len, N, 1] node feature sequences (daily returns)
      adj:   [N, N] adjacency matrix (fixed for this window)
      y:     [N_pred] next-day returns for target stocks

    Args:
        returns: DataFrame [dates x stocks] of daily log-returns
        adj: Adjacency matrix [N, N]
        pred_indices: Indices of prediction target stocks
        input_seq_len: Rolling window length d (paper default: 60)
    """

    def __init__(
        self,
        returns: np.ndarray,      # [T, N] daily returns
        adj: torch.Tensor,        # [N, N]
        pred_indices: List[int],
        input_seq_len: int = 60,
    ) -> None:
        self.returns = returns
        self.adj = adj
        self.pred_indices = pred_indices
        self.input_seq_len = input_seq_len
        self.n_samples = max(0, len(returns) - input_seq_len - 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_seq: returns over [idx, idx+input_seq_len)
        x = self.returns[idx: idx + self.input_seq_len]         # [input_seq_len, N]
        # y: next-day returns for prediction targets
        y_all = self.returns[idx + self.input_seq_len]          # [N]
        y = y_all[self.pred_indices]                            # [N_pred]

        x_t = torch.from_numpy(x).float()                      # [input_seq_len, N]
        y_t = torch.from_numpy(y).float()                      # [N_pred]
        return x_t, self.adj, y_t


class RollingWindowSplitter:
    """Implements the rolling-window strategy from Section IV.C.

    "The initial window covers 3,000 trading days, split into 70% training,
    20% validation, and 10% testing. After each iteration, the window advances
    by 300 days until the dataset is exhausted."

    Args:
        returns: Full return matrix [T, N]
        initial_window: Initial window size in days (paper: 3000)
        advance_days: Days to advance per iteration (paper: 300)
        train_frac / val_frac / test_frac: Split fractions
    """

    def __init__(
        self,
        returns: np.ndarray,
        initial_window: int = 3000,
        advance_days: int = 300,
        train_frac: float = 0.70,
        val_frac: float = 0.20,
        test_frac: float = 0.10,
    ) -> None:
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
        self.returns = returns
        self.T = len(returns)
        self.initial_window = initial_window
        self.advance_days = advance_days
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

    def splits(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (train_returns, val_returns, test_returns) for each rolling window."""
        start = 0
        window = self.initial_window
        while start + window <= self.T:
            chunk = self.returns[start: start + window]
            n_train = int(len(chunk) * self.train_frac)
            n_val = int(len(chunk) * self.val_frac)
            train = chunk[:n_train]
            val = chunk[n_train: n_train + n_val]
            test = chunk[n_train + n_val:]
            yield train, val, test
            start += self.advance_days


def generate_synthetic_returns(
    n_nodes: int = 100,
    n_days: int = 4000,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic daily log-returns for testing without market data.

    Uses a simple factor model: R_it = beta_i * F_t + epsilon_it

    Args:
        n_nodes: Number of stocks
        n_days: Number of trading days
        seed: Random seed

    Returns:
        returns: [n_days, n_nodes] array of synthetic log-returns
    """
    rng = np.random.default_rng(seed)
    # Market factor
    market_factor = rng.normal(0.0003, 0.01, size=(n_days, 1))
    # Stock-specific loadings and idiosyncratic noise
    betas = rng.uniform(0.5, 1.5, size=(1, n_nodes))
    idio = rng.normal(0, 0.015, size=(n_days, n_nodes))
    returns = betas * market_factor + idio
    return returns.astype(np.float32)
