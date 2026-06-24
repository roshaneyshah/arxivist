"""
chunkers/experimental/densex.py
================================
STUB: DenseX (Propositions) chunker.

Paper: arXiv:2606.00881, Section 2.2.
T-marker: slowest method by far (avg 15.05h); lowest Accuracy@5 (69.10%).

Results:
  Table 1 Accuracy@5 avg: 69.10%  ← lowest of all methods
  Table 2 Recall@10 avg:  27.43%
  Table 3 time avg:        15.05h  ← slowest method
  Table 4 LLM-judge avg:   3.83

Reference: Chen et al. 2024. "Dense X Retrieval: What retrieval granularity
should we use?" EMNLP 2024.
"""
from rag_chunking_bench.chunkers.base import BaseChunker
from typing import List


class DenseXChunker(BaseChunker):
    """
    STUB: DenseX (Propositions) — atomic factoid-level chunker.

    SIR Note: Expected T-marker (>48h) on most datasets. Despite fine-grained
    atomic chunks, achieves the lowest Accuracy@5 of all evaluated methods,
    demonstrating that chunk quantity does not substitute for chunk quality.
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config or {})

    @property
    def name(self) -> str:
        return "densex"

    def chunk(self, document: str) -> List[str]:
        raise NotImplementedError(
            "STUB: DenseX is not implemented. "
            "Expected T-marker (>48h) and lowest Accuracy@5 (69.10%). "
            "See Chen et al. 2024 (EMNLP) for original implementation."
        )

    def __repr__(self) -> str:
        return "DenseXChunker(STUB)"
