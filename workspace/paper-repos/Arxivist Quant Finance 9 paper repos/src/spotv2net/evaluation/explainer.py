"""GNNExplainer wrapper for per-node subgraph interpretation (Sec. 7.3, Ying et al. 2019).

ASSUMED (SIR evaluation_protocol.special_conditions, low confidence): the paper
does not disclose GNNExplainer's internal mask-optimization hyperparameters
(epochs, learning rate). This wrapper uses PyTorch Geometric's ``GNNExplainer``
defaults and documents that assumption here.
"""

from __future__ import annotations

from typing import Dict

import torch


class SpotV2NetExplainer:
    """Thin wrapper around PyG's GNNExplainer for SpotV2Net's fully connected graphs.

    Identifies, for a given target node, the subgraph of ``top_k`` most influential
    source nodes (Sec. 7.3: N*=5 in the paper's empirical study).
    """

    def explain_node(
        self, model: torch.nn.Module, graph: Dict[str, torch.Tensor], node_idx: int, top_k: int = 5
    ) -> Dict[str, object]:
        """Run GNNExplainer for a single node/timestamp and return the top-k subgraph.

        Args:
            model: A trained ``SpotV2Net`` instance.
            graph: Dict with 'x', 'edge_index', 'edge_attr' tensors for one snapshot.
            node_idx: Index of the target node whose prediction is explained.
            top_k: Number of most influential source nodes to retain (paper: N*=5).

        Returns:
            Dict with 'target_node', 'influential_nodes' (list[int], ranked),
            'edge_mask' (torch.Tensor importance scores per edge).
        """
        try:
            from torch_geometric.explain import Explainer, GNNExplainer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "torch_geometric.explain is required for SpotV2NetExplainer. "
                "Install torch-geometric>=2.4 (see requirements.txt)."
            ) from exc

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),  # ASSUMED: PyG default epoch count
            explanation_type="model",
            node_mask_type="object",
            edge_mask_type="object",
            model_config=dict(mode="regression", task_level="node", return_type="raw"),
        )

        explanation = explainer(
            x=graph["x"], edge_index=graph["edge_index"], edge_attr=graph["edge_attr"], index=node_idx
        )

        edge_mask = explanation.edge_mask
        edge_index = graph["edge_index"]

        # Rank source nodes (edges pointing INTO node_idx) by importance, per Sec. 7.3.
        incoming = edge_index[1] == node_idx
        src_nodes = edge_index[0][incoming]
        importances = edge_mask[incoming]
        order = torch.argsort(importances, descending=True)
        ranked_sources = src_nodes[order][:top_k].tolist()

        return {
            "target_node": node_idx,
            "influential_nodes": ranked_sources,
            "edge_mask": edge_mask,
        }
