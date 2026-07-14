"""
chunkers/experimental/lumberchunker.py
=======================================
STUB: LumberChunker — LLM-based narrative chunker.

Paper: arXiv:2606.00881, Section 2.2.
T-marker: exceeds 48-hour time limit on most datasets.
Paper uses GPT-OSS-20B as the underlying LLM (not the original GPT-4).

Results where it runs:
  Table 1 Accuracy@5 avg (completed runs): 85.44%
  Table 2 Recall@10 avg (completed runs):  78.16%  ← highest Recall@10 overall
  Table 4 LLM-judge avg (completed runs):   4.35   ← highest judge score overall
  Table 3 time avg (completed runs):         8.37h

Reference: Duarte et al. 2024. "LumberChunker: Long-form narrative document
segmentation." EMNLP 2024 Findings.
"""
from rag_chunking_bench.chunkers.base import BaseChunker
from typing import List


class LumberChunker(BaseChunker):
    """
    STUB: LumberChunker — LLM-based semantic shift detector.

    SIR Note: Expected T-marker (>48h timeout) on most datasets.
    Paper replaces original GPT-4 with GPT-OSS-20B as the internal LLM.
    SIR confidence on LLM temperature/sampling: 0.55 (not stated in paper).
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config or {})

    @property
    def name(self) -> str:
        return "lumberchunker"

    def chunk(self, document: str) -> List[str]:
        raise NotImplementedError(
            "STUB: LumberChunker is not implemented. "
            "Expected T-marker (>48h) on most datasets. "
            "See Duarte et al. 2024 (EMNLP) for original implementation."
        )

    def __repr__(self) -> str:
        return "LumberChunker(STUB)"
