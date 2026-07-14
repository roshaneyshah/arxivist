"""
chunkers/base.py
================
Abstract base class for all chunking methods.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 2.1: Theoretical Background — EQ2 defines the chunking function.

EQ2: f_theta : D → C = {c_1, c_2, ..., c_m}, where c_i = <t_start_i, ..., t_end_i>

All concrete chunkers inherit from BaseChunker and implement chunk().
This enforces a uniform interface so the pipeline can swap methods via config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    """
    Abstract base class for all text chunkers.

    Implements the chunking function f_theta(D) -> C from EQ2 of the paper
    (Section 2.1). Subclasses implement chunk() with method-specific logic.

    Args:
        config: Method-specific config dict (from BenchConfig.chunkers.<method>).
    """

    def __init__(self, config: dict) -> None:
        self._config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this chunking method."""
        ...

    @abstractmethod
    def chunk(self, document: str) -> List[str]:
        """
        Segment a document into chunks.

        Implements EQ2: f_theta(D) -> C = {c_1, ..., c_m}

        Args:
            document: Raw text string D = <t_1, ..., t_N>

        Returns:
            List of text chunk strings [c_1, ..., c_m].
            May be overlapping (depending on method).
        """
        ...

    def chunk_batch(self, documents: List[str]) -> List[List[str]]:
        """
        Segment a list of documents. Default: calls chunk() per document.

        Args:
            documents: List of raw text documents.

        Returns:
            List of chunk lists, one per document.
        """
        return [self.chunk(doc) for doc in documents]

    def get_config(self) -> dict:
        """Return a copy of this chunker's configuration."""
        return dict(self._config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self._config})"
