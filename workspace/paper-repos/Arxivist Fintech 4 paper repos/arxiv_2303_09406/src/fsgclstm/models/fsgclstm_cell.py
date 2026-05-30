"""
models/fsgclstm_cell.py
=======================
Full-State Graph Convolutional LSTM Cell — the core contribution of the paper.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section III.b

Key innovation: GCN (H_tilde) is applied to ALL three LSTM inputs:
  - X_t (current node features)       ← also done by baseline GCLSTM
  - h_{t-1} (previous hidden state)   ← SAME as baseline
  - c_{t-1} (previous cell state)     ← NEW: not done in baseline

Gate equations (Section III.b):
  f_t = σ(W_f [H_tilde(h_{t-1}), H_tilde(X_t)] + b_f)
  i_t = σ(W_i [H_tilde(h_{t-1}), H_tilde(X_t)] + b_i)
  c_t = f_t ⊙ H_tilde(c_{t-1}) + i_t ⊙ tanh(W_c [H_tilde(h_{t-1}), H_tilde(X_t)] + b_c)
  o_t = σ(W_o [H_tilde(h_{t-1}), H_tilde(X_t)] + b_o)
  h_t = o_t ⊙ tanh(c_t)
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from .gcn_layer import TwoLayerGCN


class FSGCLSTMCell(nn.Module):
    """Full-State Graph Convolutional LSTM Cell.

    Applies 2-layer GCN to h_{t-1}, c_{t-1}, and X_t before LSTM gates,
    ensuring spatial (graph) information propagates through all cell components.

    Args:
        input_dim: Dimension of input node features (d or hidden_dim for stacked cells)
        hidden_dim: LSTM hidden state dimension
            WARNING: not stated in paper — ASSUMED = 64 (conf: 0.45)
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GCN transformers — one per tensor type (H_tilde in paper notation)
        self.gcn_x = TwoLayerGCN(input_dim, hidden_dim, hidden_dim)
        self.gcn_h = TwoLayerGCN(hidden_dim, hidden_dim, hidden_dim)
        self.gcn_c = TwoLayerGCN(hidden_dim, hidden_dim, hidden_dim)

        # LSTM gate weight matrices (operate on concatenated GCN outputs: [H_tilde(h), H_tilde(X)])
        gate_input_dim = 2 * hidden_dim  # concat of H_tilde(h_{t-1}) and H_tilde(X_t)
        self.W_f = nn.Linear(gate_input_dim, hidden_dim)  # forget gate
        self.W_i = nn.Linear(gate_input_dim, hidden_dim)  # input gate
        self.W_c = nn.Linear(gate_input_dim, hidden_dim)  # cell candidate
        self.W_o = nn.Linear(gate_input_dim, hidden_dim)  # output gate

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
        adj_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single FS-GCLSTM step.

        Args:
            x:        Current node features    [N, input_dim]
            h_prev:   Previous hidden state    [N, hidden_dim]
            c_prev:   Previous cell state      [N, hidden_dim]
            adj_norm: Normalized adjacency     [N, N]

        Returns:
            h_t: New hidden state [N, hidden_dim]
            c_t: New cell state   [N, hidden_dim]
        """
        assert x.dim() == 2, f"x: expected [N, d], got {x.shape}"
        assert h_prev.dim() == 2, f"h_prev: expected [N, hidden], got {h_prev.shape}"
        assert c_prev.dim() == 2, f"c_prev: expected [N, hidden], got {c_prev.shape}"
        N = x.shape[0]
        assert adj_norm.shape == (N, N), f"adj_norm: expected [{N},{N}], got {adj_norm.shape}"

        # Apply GCN to each input — implements H_tilde(·) from paper
        H_x = self.gcn_x(x, adj_norm)          # [N, hidden_dim]  — graph-convolved X_t
        H_h = self.gcn_h(h_prev, adj_norm)      # [N, hidden_dim]  — graph-convolved h_{t-1}
        H_c = self.gcn_c(c_prev, adj_norm)      # [N, hidden_dim]  — graph-convolved c_{t-1} (key innovation)

        # Concatenate H_tilde(h_{t-1}) and H_tilde(X_t) for gate inputs
        gate_in = torch.cat([H_h, H_x], dim=-1)  # [N, 2*hidden_dim]

        # LSTM gate computations — Eqs. in Section III.b
        f_t = torch.sigmoid(self.W_f(gate_in))          # forget gate
        i_t = torch.sigmoid(self.W_i(gate_in))          # input gate
        g_t = torch.tanh(self.W_c(gate_in))             # cell candidate
        o_t = torch.sigmoid(self.W_o(gate_in))          # output gate

        # Cell state: uses H_tilde(c_{t-1}) — the full-state innovation
        c_t = f_t * H_c + i_t * g_t                     # [N, hidden_dim]
        h_t = o_t * torch.tanh(c_t)                     # [N, hidden_dim]

        return h_t, c_t

    def init_hidden(self, n_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states to zeros."""
        h = torch.zeros(n_nodes, self.hidden_dim, device=device)
        c = torch.zeros(n_nodes, self.hidden_dim, device=device)
        return h, c

    def __repr__(self) -> str:
        return f"FSGCLSTMCell(input_dim={self.input_dim}, hidden_dim={self.hidden_dim})"
