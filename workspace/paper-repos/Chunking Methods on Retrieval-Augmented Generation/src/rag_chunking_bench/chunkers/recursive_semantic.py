"""
chunkers/recursive_semantic.py
===============================
Recursive semantic chunker (best Accuracy@5 among robust methods).

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 2.2: "Recursive Character Splitting uses a hierarchy of separators
(paragraphs, lines, spaces) to iteratively split oversized segments, preserving
semantic coherence when possible."

Results (Table 1 Accuracy@5 avg): 89.36%  ← highest among robust methods
Results (Table 2 Recall@10 avg):  53.81%
Results (Table 3 time avg):        4.90m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rag_chunking_bench.chunkers.base import BaseChunker


@dataclass
class RecursiveSemanticConfig:
    chunk_size: int = 512
    overlap: int = 50
    # TODO: verify exact separator list from the LangChain version used in paper
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )


class RecursiveSemanticChunker(BaseChunker):
    """
    Recursive character text splitter with semantic separator hierarchy.

    Uses LangChain's RecursiveCharacterTextSplitter with a descending
    separator hierarchy: paragraphs → lines → sentences → words.

    Paper reference: Section 2.2 (Recursive Character Splitting) and Section 3
    ("Default hyperparameters taken from the official code repositories").

    Args:
        config: Dict with keys 'chunk_size', 'overlap', 'separators'.
    """

    def __init__(self, config: dict | None = None) -> None:
        raw = config or {}
        cfg = RecursiveSemanticConfig(
            chunk_size=raw.get("chunk_size", 512),
            overlap=raw.get("overlap", 50),
            separators=raw.get("separators", ["\n\n", "\n", ". ", " ", ""]),
        )
        super().__init__({"chunk_size": cfg.chunk_size, "overlap": cfg.overlap})
        self._chunk_size = cfg.chunk_size
        self._overlap = cfg.overlap
        self._separators = cfg.separators

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size * 4,
            chunk_overlap=self._overlap * 4,
            separators=self._separators,
            length_function=len,
        )

    @property
    def name(self) -> str:
        return "recursive_semantic"

    def chunk(self, document: str) -> List[str]:
        """
        Recursively split document using separator hierarchy.

        EQ2: f_theta(D) -> C, theta = {chunk_size, overlap, separators}

        Args:
            document: Raw text string.

        Returns:
            List of text chunks preserving semantic boundaries where possible.
        """
        if not document.strip():
            return []
        return self._splitter.split_text(document)

    def __repr__(self) -> str:
        return (
            f"RecursiveSemanticChunker("
            f"chunk_size={self._chunk_size}, overlap={self._overlap})"
        )
