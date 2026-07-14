"""Edge-aware Graph Attention layer.

Implements the modified GAT attention mechanism of Sec. 5, arXiv:2401.06249, which
extends Velickovic et al. (2017) to condition attention coefficients on edge features
(the spot volatility-of-volatility / co-volatility-of-volatility pairs).

Equation reference (SIR mathematical_spec):
    alpha'_ij = softmax_j( LeakyReLU( q'^T [W x_i || W x_j || U x^e_ij] ) )
    x'_i = sigma( sum_j alpha_ij * W x_j )   (single head)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGATLayer(nn.Module):
    """Multi-head Graph Attention layer with edge-feature-conditioned attention.

    Paper reference: Sec. 5, arXiv:2401.06249 ("To include edge features in a GAT...").

    Args:
        in_dim: Input node feature dimension (M or M').
        out_dim: Output dimension **per head** if ``concat`` else the shared output dim.
        edge_dim: Input edge feature dimension (E or E').
        heads: Number of independent attention heads (K in the paper).
        concat: If True, head outputs are concatenated (hidden layers). If False,
            head outputs are averaged (the layer immediately preceding the
            prediction layer, per Sec. 5).
        negative_slope: LeakyReLU slope ``c`` used inside the attention score (Table 8).
        dropout: Dropout applied to the node/edge linear projections.
        attn_dropout: Dropout applied to normalized attention coefficients.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.1,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.concat = concat
        self.out_dim = out_dim
        self.negative_slope = negative_slope

        # NOTE on ambiguity (SIR ambiguities[0]/[1], architecture_plan risk #4):
        # `out_dim` is treated as the layer's *total* output dimension (matching the
        # Table 8 "Dimension Hidden Layers" values [400, 200] directly). When concat=True
        # each head therefore produces out_dim // heads features; when concat=False
        # (averaging) each head produces the full out_dim and outputs are averaged.
        self.head_dim = out_dim // heads if concat else out_dim

        self.W = nn.Linear(in_dim, self.head_dim * heads, bias=False)
        self.U = nn.Linear(edge_dim, self.head_dim * heads, bias=False)

        # q' in the paper: single-layer feedforward attention scorer, per head.
        self.attn = nn.Parameter(torch.empty(heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"EdgeAwareGATLayer(head_dim={self.head_dim}, heads={self.heads}, "
            f"concat={self.concat})"
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass over a single fully-connected graph snapshot.

        Args:
            x: Node features, shape ``[N, in_dim]``.
            edge_index: Edge list, shape ``[2, num_edges]`` (source, target).
            edge_attr: Edge features, shape ``[num_edges, edge_dim]``, aligned with
                ``edge_index`` column order.

        Returns:
            Updated node embeddings, shape ``[N, out_dim]`` (heads concatenated) or
            ``[N, head_dim]`` (heads averaged, when ``concat=False``).
        """
        assert x.dim() == 2, f"Expected node features [N, in_dim], got {x.shape}"
        assert edge_index.dim() == 2 and edge_index.size(0) == 2, (
            f"Expected edge_index [2, num_edges], got {edge_index.shape}"
        )
        assert edge_attr.size(0) == edge_index.size(1), (
            "edge_attr and edge_index must have matching number of edges: "
            f"{edge_attr.size(0)} vs {edge_index.size(1)}"
        )

        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]  # message flows src -> dst

        Wx = self.dropout(self.W(x)).view(num_nodes, self.heads, self.head_dim)  # [N,K,d]
        Ue = self.dropout(self.U(edge_attr)).view(-1, self.heads, self.head_dim)  # [E,K,d]

        Wx_src = Wx[src]  # [E, K, d]
        Wx_dst = Wx[dst]  # [E, K, d]

        # q'^T [W x_i || W x_j || U x^e_ij]  computed per head via elementwise dot.
        concat_feats = torch.cat([Wx_dst, Wx_src, Ue], dim=-1)  # [E, K, 3d]
        e = self.leaky_relu((concat_feats * self.attn.unsqueeze(0)).sum(dim=-1))  # [E, K]

        alpha = _segment_softmax(e, dst, num_nodes)  # normalize over incoming edges per dst
        alpha = self.attn_dropout(alpha)  # [E, K]

        weighted = Wx_src * alpha.unsqueeze(-1)  # [E, K, d]
        out = torch.zeros(num_nodes, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, weighted)  # sum_j alpha_ij * W x_j, aggregated per target node

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.head_dim)  # [N, out_dim]
        else:
            out = out.mean(dim=1)  # [N, head_dim] — averaging rule, Sec. 5

        return out


def _segment_softmax(
    scores: torch.Tensor, index: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Numerically stable softmax of ``scores`` grouped by ``index`` (per target node).

    Equivalent to ``alpha_ij = softmax_j(e_ij)`` in Eq. of Sec. 5, computed per head.

    Args:
        scores: Raw attention scores, shape ``[E, K]``.
        index: Destination node index per edge, shape ``[E]``.
        num_segments: Number of nodes N (graph is fully connected, so this is also
            the softmax normalization group count).

    Returns:
        Normalized attention weights, shape ``[E, K]``.
    """
    scores_max = torch.full(
        (num_segments, scores.size(1)), float("-inf"), device=scores.device, dtype=scores.dtype
    )
    scores_max.index_reduce_(0, index, scores, reduce="amax", include_self=True)
    scores_max = scores_max.index_select(0, index)
    scores = scores - scores_max

    exp_scores = torch.exp(scores)
    denom = torch.zeros(num_segments, scores.size(1), device=scores.device, dtype=scores.dtype)
    denom.index_add_(0, index, exp_scores)
    denom = denom.index_select(0, index).clamp_min(1e-16)

    return exp_scores / denom
