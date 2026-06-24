"""
chunkers/fixed_size.py
======================
Fixed-size chunking method.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 2.2: "Fixed-size Chunking divides text into segments of predetermined
length, often with overlapping windows to reduce mid-sentence splits."
Section 3: "we used a chunk size of 512 (i.e. quite common choice) with an
overlap of 50."

Results (Table 1 Accuracy@5 avg): 87.71%
Results (Table 2 Recall@10 avg):  44.75%
Results (Table 3 time avg):        <1s   ← fastest by orders of magnitude
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag_chunking_bench.chunkers.base import BaseChunker


@dataclass
class FixedSizeConfig:
    chunk_size: int = 512  # SIR conf 0.99 — explicitly stated
    overlap: int = 50      # SIR conf 0.99 — explicitly stated


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size text chunker using LangChain's RecursiveCharacterTextSplitter.

    Divides text into segments of predetermined character/token length with
    an overlapping window to reduce mid-sentence splits.

    Paper reference: Section 2.2 and Section 3.
    Config: chunk_size=512, overlap=50 (both explicitly stated in paper).

    Args:
        config: Dict with keys 'chunk_size' (int) and 'overlap' (int).
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = FixedSizeConfig(**(config or {}))
        super().__init__({"chunk_size": cfg.chunk_size, "overlap": cfg.overlap})
        self._chunk_size = cfg.chunk_size
        self._overlap = cfg.overlap

        # LangChain splitter: character-based, hierarchy of separators
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size * 4,   # ~4 chars per token on average
            chunk_overlap=self._overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    @property
    def name(self) -> str:
        return "fixed_size"

    def chunk(self, document: str) -> List[str]:
        """
        Segment document into fixed-size overlapping chunks.

        EQ2: f_theta(D) -> C, theta = {chunk_size=512, overlap=50}

        Args:
            document: Raw text string.

        Returns:
            List of text chunks.
        """
        if not document.strip():
            return []
        return self._splitter.split_text(document)

    def __repr__(self) -> str:
        return f"FixedSizeChunker(chunk_size={self._chunk_size}, overlap={self._overlap})"
