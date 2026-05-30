"""
models/fsgclstm_model.py
========================
Full FS-GCLSTM model: 3 stacked FS-GCLSTM cells + flatten + MLP head.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section III.c

Architecture (Figure 2):
  Input: rolling windows of d-day price data → node features [N, d]
  → 3 stacked FS-GCLSTM cells (each cell processes the full sequence)
  → Concatenate final hidden states from all 3 layers [N, 3*hidden_dim]
  → Select N_pred target nodes
  → Flatten [N_pred * 3 * hidden_dim]
  → MLP → predicted next-day returns [N_pred]

NOTE: MLP layer sizes are ASSUMED (not stated in paper, conf: 0.50).
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from .fsgclstm_cell import FSGCLSTMCell
from .gcn_layer import normalize_adjacency


class FSGCLSTMModel(nn.Module):
    """Full-State Graph Convolutional LSTM for stock return prediction.

    Paper: Section III.c — "Three stacked FS-GCLSTM cells process the temporal
    sequence, with each cell applying GCN transformations to all LSTM inputs."

    Args:
        input_dim: Node feature dimension (= input_seq_len, i.e. d rolling-window days)
        hidden_dim: Hidden state dimension (ASSUMED: 64, not stated in paper)
        n_lstm_layers: Number of stacked cells (stated: 3)
        n_pred: Number of output stocks N_pred
        mlp_hidden: MLP intermediate size (ASSUMED: not stated in paper)
        dropout: Dropout rate (ASSUMED: not mentioned)
        pred_node_indices: Indices of target stocks in the full graph (length N_pred)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_lstm_layers: int,
        n_pred: int,
        mlp_hidden: int = 128,
        dropout: float = 0.0,
        pred_node_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.n_pred = n_pred
        self.pred_node_indices = pred_node_indices

        # Build stacked FS-GCLSTM cells
        # Layer 0: input_dim -> hidden_dim
        # Layers 1+: hidden_dim -> hidden_dim
        cells = []
        for i in range(n_lstm_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            cells.append(FSGCLSTMCell(in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)

        # MLP head — ASSUMED architecture (conf: 0.50)
        # Input: n_pred * n_lstm_layers * hidden_dim (after selecting target nodes + flatten)
        mlp_in = n_pred * n_lstm_layers * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_pred),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_seq:  Sequence of node feature matrices [seq_len, N, input_dim]
            adj:    Raw adjacency matrix [N, N] (will be normalized internally)

        Returns:
            Predicted next-day returns for N_pred target stocks [N_pred]
        """
        assert x_seq.dim() == 3, f"x_seq: expected [T, N, d], got {x_seq.shape}"
        seq_len, N, _ = x_seq.shape
        device = x_seq.device

        # Normalize adjacency once per forward pass
        adj_norm = normalize_adjacency(adj)  # [N, N]

        # Initialize hidden and cell states for each layer
        states: List[Tuple[torch.Tensor, torch.Tensor]] = [
            cell.init_hidden(N, device) for cell in self.cells
        ]

        # Process temporal sequence
        for t in range(seq_len):
            x_t = x_seq[t]          # [N, input_dim] (or [N, hidden_dim] for layer 1+)
            new_states = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = states[layer_idx]
                h_t, c_t = cell(x_t, h_prev, c_prev, adj_norm)
                new_states.append((h_t, c_t))
                x_t = h_t            # Output of layer i is input to layer i+1
            states = new_states

        # Collect final hidden states from all layers: [n_lstm_layers, N, hidden_dim]
        final_hiddens = torch.stack([s[0] for s in states], dim=0)  # [L, N, hidden_dim]

        # Rearrange to [N, L, hidden_dim] then select N_pred target nodes
        final_hiddens = final_hiddens.permute(1, 0, 2)              # [N, L, hidden_dim]
        if self.pred_node_indices is not None:
            idx = torch.tensor(self.pred_node_indices, device=device, dtype=torch.long)
            final_hiddens = final_hiddens[idx]                       # [N_pred, L, hidden_dim]
        else:
            final_hiddens = final_hiddens[:self.n_pred]

        # Flatten and pass through MLP
        flat = final_hiddens.reshape(1, -1)                          # [1, N_pred*L*hidden_dim]
        out = self.mlp(flat).squeeze(0)                              # [N_pred]
        return out

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"FSGCLSTMModel(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_lstm_layers}, n_pred={self.n_pred})"
        )
