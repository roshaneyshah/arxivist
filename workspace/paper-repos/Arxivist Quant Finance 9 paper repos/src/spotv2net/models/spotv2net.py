"""SpotV2Net: full model (Sec. 5, arXiv:2401.06249).

Stacks EdgeAwareGATLayer modules per the tuned 2-hidden-layer architecture
(Table 8) and applies a final affine prediction head:
    y_hat_i = O x'_i + u
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from spotv2net.models.gat_layer import EdgeAwareGATLayer


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


class SpotV2Net(nn.Module):
    """GAT-based multivariate intraday spot volatility forecaster.

    Paper reference: Sec. 5 (architecture), Table 8 (hyperparameters).

    Args:
        node_in_dim: Node feature dimension M (Eq. 1: (L+1) * N_covols terms).
        edge_in_dim: Edge feature dimension E (Eq. 2). Ignored if
            ``use_edge_features=False`` (SpotV2Net-NE ablation, Sec. 7.2).
        hidden_dims: Per-layer total output dims, e.g. ``[400, 200]`` (Table 8).
        heads: Number of attention heads K (Table 8).
        output_dim: 1 for single-step forecasts (Sec. 7.2) or 14 for the
            non-recursive multi-step functional forecast (Sec. 7.4).
        use_edge_features: If False, edge features are zeroed out internally,
            reproducing the SpotV2Net-NE ablation used in Table 2/5.
        dropout: Architecture dropout (Table 8 "Dropout (Architecture)").
        attn_dropout: Attention dropout (Table 8 "Dropout (Attention)").
        negative_slope: LeakyReLU slope for attention scores (Table 8).
        activation: Nonlinearity applied after each hidden GAT layer (Table 8).
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dims: List[int],
        heads: int = 4,
        output_dim: int = 1,
        use_edge_features: bool = True,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        negative_slope: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer size")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}', choose from {list(_ACTIVATIONS)}")

        self.use_edge_features = use_edge_features
        self.edge_in_dim = edge_in_dim
        self.activation = _ACTIVATIONS[activation]

        layers = []
        in_dim = node_in_dim
        num_layers = len(hidden_dims)
        for i, h_dim in enumerate(hidden_dims):
            is_last_hidden = i == num_layers - 1
            # Sec. 5: concatenate at all hidden layers EXCEPT the one immediately
            # preceding the prediction layer, where heads are averaged instead.
            concat = not is_last_hidden
            layers.append(
                EdgeAwareGATLayer(
                    in_dim=in_dim,
                    out_dim=h_dim,
                    edge_dim=edge_in_dim,
                    heads=heads,
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            )
            in_dim = h_dim if not concat else h_dim  # out_dim already accounts for concat
        self.gat_layers = nn.ModuleList(layers)

        # Prediction (output) layer: y_hat_i = O x'_i + u
        self.output_layer = nn.Linear(in_dim, output_dim)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"SpotV2Net(layers={len(self.gat_layers)}, "
            f"use_edge_features={self.use_edge_features}, "
            f"output_dim={self.output_layer.out_features})"
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features, shape ``[N, node_in_dim]``.
            edge_index: Edge list, shape ``[2, num_edges]``.
            edge_attr: Edge features, shape ``[num_edges, edge_in_dim]``.

        Returns:
            Predictions, shape ``[N, output_dim]``.
        """
        assert x.dim() == 2, f"Expected node features [N, D], got {x.shape}"

        if not self.use_edge_features:
            # SpotV2Net-NE ablation (Sec. 7.2): zero out edge features so the
            # attention mechanism degenerates to the base (no edge-feature) GAT.
            edge_attr = torch.zeros(
                edge_attr.size(0), self.edge_in_dim, device=x.device, dtype=x.dtype
            )

        h = x
        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_attr)
            h = self.activation(h)

        y_hat = self.output_layer(h)
        return y_hat
