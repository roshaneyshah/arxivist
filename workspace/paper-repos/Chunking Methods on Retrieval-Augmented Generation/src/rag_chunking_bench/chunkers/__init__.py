"""
chunkers/__init__.py — ChunkerRegistry for RAG Chunking Benchmark (arXiv:2606.00881)
"""
from __future__ import annotations
from typing import Dict, List, Type
from rag_chunking_bench.chunkers.base import BaseChunker
from rag_chunking_bench.chunkers.fixed_size import FixedSizeChunker
from rag_chunking_bench.chunkers.recursive_semantic import RecursiveSemanticChunker
from rag_chunking_bench.chunkers.sequential_hac import SequentialHACChunker
from rag_chunking_bench.chunkers.max_min import MaxMinChunker
from rag_chunking_bench.chunkers.texttiling import TextTilingChunker
from rag_chunking_bench.chunkers.experimental.graphseg import GraphSegChunker
from rag_chunking_bench.chunkers.experimental.lumberchunker import LumberChunker
from rag_chunking_bench.chunkers.experimental.densex import DenseXChunker

_REGISTRY: Dict[str, Type[BaseChunker]] = {
    "fixed_size": FixedSizeChunker,
    "recursive_semantic": RecursiveSemanticChunker,
    "sequential_hac": SequentialHACChunker,
    "max_min": MaxMinChunker,
    "texttiling": TextTilingChunker,
    "graphseg": GraphSegChunker,
    "lumberchunker": LumberChunker,
    "densex": DenseXChunker,
}

class ChunkerRegistry:
    @staticmethod
    def register(name: str, cls: Type[BaseChunker]) -> None:
        _REGISTRY[name] = cls

    @staticmethod
    def get(name: str, config: dict | None = None) -> BaseChunker:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown chunker '{name}'. Available: {ChunkerRegistry.list_available()}")
        return _REGISTRY[name](config)

    @staticmethod
    def list_available() -> List[str]:
        return sorted(_REGISTRY.keys())

    @staticmethod
    def list_robust() -> List[str]:
        """Four robustly-completing methods per paper Section 5."""
        return ["fixed_size", "recursive_semantic", "sequential_hac", "max_min"]
