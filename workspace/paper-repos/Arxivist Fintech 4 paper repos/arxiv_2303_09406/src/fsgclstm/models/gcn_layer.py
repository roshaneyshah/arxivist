"""
models/gcn_layer.py
===================
Graph Convolutional Layer implementing Eq. (1)-(2) of the paper.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section III.a

Eq. (1): Z = D_tilde^{-1/2} A_tilde D_tilde^{-1/2} X W
Eq. (2): H = f(Z)  where f = ReLU

Two of these layers are stacked and applied to each of X_t, h_{t-1}, c_{t-1}
in the FS-GCLSTM cell.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Single graph convolutional layer (Kipf & Welling 2016, Eq. 1-2).

    Computes: H = ReLU(D_tilde^{-1/2} A_tilde D_tilde^{-1/2} X W)

    Args:
        in_features: Input feature dimension d
        out_features: Output feature dimension d_out
        bias: Whether to include bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj_norm: Pre-normalized adjacency D_tilde^{-1/2} A_tilde D_tilde^{-1/2} [N, N]

        Returns:
            H: [N, out_features]
        """
        assert x.dim() == 2, f"Expected [N, d], got {x.shape}"
        assert adj_norm.dim() == 2, f"Expected [N, N], got {adj_norm.shape}"
        assert adj_norm.shape[0] == x.shape[0], "Node count mismatch between x and adj_norm"

        # Eq. (1): Z = A_norm @ X @ W
        support = torch.mm(x, self.weight)           # [N, out_features]
        out = torch.mm(adj_norm, support)             # [N, out_features]
        if self.bias is not None:
            out = out + self.bias
        return F.relu(out)                            # Eq. (2): H = ReLU(Z)

    def __repr__(self) -> str:
        return f"GraphConvLayer({self.in_features} -> {self.out_features})"


class TwoLayerGCN(nn.Module):
    """Two stacked GraphConvLayers as described in Section III.a.

    Paper: "Stacking layers enables multi-hop neighborhood aggregation;
    in this work, two GCN layers are applied to each processed tensor."

    This module is denoted H_tilde(·) in the paper equations.

    Args:
        in_features: Input dimension
        hidden_features: Intermediate dimension (ASSUMED = out_features if not specified)
        out_features: Output dimension
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.layer1 = GraphConvLayer(in_features, hidden_features)
        self.layer2 = GraphConvLayer(hidden_features, out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj_norm: Normalized adjacency [N, N]

        Returns:
            H_tilde: [N, out_features]
        """
        h = self.layer1(x, adj_norm)      # [N, hidden_features]
        return self.layer2(h, adj_norm)   # [N, out_features]

    def __repr__(self) -> str:
        return f"TwoLayerGCN({self.layer1.in_features} -> {self.layer1.out_features} -> {self.layer2.out_features})"


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Compute symmetric normalized adjacency D_tilde^{-1/2} A_tilde D_tilde^{-1/2}.

    Following Kipf & Welling (2016): add self-loops, then normalize.
    A_tilde = A + I_n

    Args:
        adj: Raw adjacency matrix [N, N] (binary or weighted)

    Returns:
        adj_norm: Normalized adjacency [N, N]
    """
    N = adj.shape[0]
    # Add self-loops: A_tilde = A + I
    adj_tilde = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
    # Degree matrix: D_tilde_ii = sum_j A_tilde_ij
    deg = adj_tilde.sum(dim=1)                          # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0   # Handle isolated nodes
    # D^{-1/2} A D^{-1/2}
    adj_norm = deg_inv_sqrt.unsqueeze(1) * adj_tilde * deg_inv_sqrt.unsqueeze(0)
    return adj_norm
