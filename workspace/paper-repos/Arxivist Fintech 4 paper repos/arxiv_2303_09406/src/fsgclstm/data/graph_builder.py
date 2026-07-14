"""
data/graph_builder.py
=====================
Constructs value-chain graph adjacency matrices from LSEG supply-chain data.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section IV.A-B

Graph structure:
  - Nodes: companies in the universe (N total)
  - Edges: supplier-customer relationships from LSEG
  - Edge weights: LSEG confidence scores (only keep > 0.20 threshold)
  - Edges treated as bidirectional (undirected)

Data note: LSEG value-chain data requires a paid subscription.
See data/README_data.md for instructions.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch


class ValueChainGraphBuilder:
    """Builds adjacency matrices from LSEG value-chain relationship data.

    Paper Section IV.A: "For each company, we retrieve its suppliers and
    customers to construct a directed graph... edges are treated as bidirectional."

    Args:
        confidence_threshold: Minimum LSEG confidence score to include edge (paper: 0.20)
        bidirectional: If True, treat all edges as undirected (paper: True)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.20,
        bidirectional: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.bidirectional = bidirectional

    def build_adjacency(
        self,
        node_ids: List[str],
        edges_df: pd.DataFrame,
        use_confidence_weights: bool = True,
    ) -> torch.Tensor:
        """Build adjacency matrix from edge list.

        Args:
            node_ids: Ordered list of company identifiers (defines row/col ordering)
            edges_df: DataFrame with columns ['source', 'target', 'confidence', 'date']
            use_confidence_weights: If True, use confidence scores as weights;
                                    if False, binary adjacency

        Returns:
            adj: Adjacency matrix [N, N] as float32 tensor
        """
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)
        adj = np.zeros((N, N), dtype=np.float32)

        # Filter by confidence threshold
        filtered = edges_df[edges_df["confidence"] >= self.confidence_threshold].copy()

        for _, row in filtered.iterrows():
            src, tgt = row["source"], row["target"]
            if src not in node_index or tgt not in node_index:
                continue
            i, j = node_index[src], node_index[tgt]
            weight = float(row["confidence"]) if use_confidence_weights else 1.0
            adj[i, j] = weight
            if self.bidirectional:
                adj[j, i] = weight  # make undirected

        return torch.from_numpy(adj)

    def build_from_snapshot(
        self,
        node_ids: List[str],
        edges_df: pd.DataFrame,
        snapshot_date: str,
        use_most_recent: bool = True,
    ) -> torch.Tensor:
        """Build adjacency using the most recent relationships as of snapshot_date.

        Paper Section IV.C: "Graphs are constructed using the most recent
        value-chain relationships."

        Args:
            node_ids: Company identifiers
            edges_df: Full edge history with 'date' column
            snapshot_date: Date string for snapshot
            use_most_recent: Use most recent record per edge pair before snapshot_date

        Returns:
            adj: [N, N] adjacency matrix
        """
        snap = pd.Timestamp(snapshot_date)
        valid = edges_df[pd.to_datetime(edges_df["date"]) <= snap].copy()
        if use_most_recent:
            valid = (
                valid.sort_values("date")
                .groupby(["source", "target"])
                .last()
                .reset_index()
            )
        return self.build_adjacency(node_ids, valid)

    def __repr__(self) -> str:
        return f"ValueChainGraphBuilder(threshold={self.confidence_threshold}, bidirectional={self.bidirectional})"


def generate_synthetic_graph(
    n_nodes: int = 100,
    n_pred: int = 40,
    density: float = 8e-4,
    seed: int = 42,
) -> Tuple[torch.Tensor, List[int]]:
    """Generate a synthetic value-chain graph for testing without LSEG data.

    Mimics paper Table I statistics: density ~8e-4 for Eurostoxx 600.

    Args:
        n_nodes: Total number of graph nodes
        n_pred: Number of prediction target nodes
        density: Graph edge density (paper: 8.1e-4 Eurostoxx, 6.5e-4 S&P500)
        seed: Random seed

    Returns:
        adj: [n_nodes, n_nodes] binary adjacency tensor
        pred_indices: List of n_pred target node indices
    """
    rng = np.random.default_rng(seed)
    n_edges = max(1, int(density * n_nodes * (n_nodes - 1)))
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _ in range(n_edges):
        i = rng.integers(0, n_nodes)
        j = rng.integers(0, n_nodes)
        if i != j:
            w = rng.uniform(0.2, 1.0)
            adj[i, j] = w
            adj[j, i] = w  # bidirectional
    pred_indices = sorted(rng.choice(n_nodes, size=n_pred, replace=False).tolist())
    return torch.from_numpy(adj), pred_indices
