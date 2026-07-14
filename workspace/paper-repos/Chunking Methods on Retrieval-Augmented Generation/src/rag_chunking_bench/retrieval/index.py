"""
retrieval/index.py
==================
FAISS-based vector index for cosine similarity chunk retrieval.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 2.1: EQ3 — C_retrieved = argTopK_{c ∈ C} [E(q)·E(c) / (||E(q)|| ||E(c)||)]

Since all embeddings are L2-normalized by ChunkEmbedder, cosine similarity
reduces to the inner product: E(q)·E(c). FAISS IndexFlatIP implements this
exactly.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


class FAISSChunkIndex:
    """
    FAISS inner-product index over L2-normalized chunk embeddings.

    Implements EQ3 (Section 2.1): cosine top-k retrieval.
    Since embeddings are L2-normalized, inner product = cosine similarity.

    Args:
        embedding_dim: Dimensionality of chunk embeddings (default 1024 for bge-m3).
        use_gpu: If True, transfer index to GPU (requires faiss-gpu).
    """

    def __init__(self, embedding_dim: int = 1024, use_gpu: bool = False) -> None:
        self._dim = embedding_dim
        self._use_gpu = use_gpu
        self._index = None
        self._chunks: List[str] = []

    def build(self, chunks: List[str], embeddings: np.ndarray) -> None:
        """
        Build the FAISS index from chunk texts and their embeddings.

        EQ3 setup: stores all E(c) vectors for cosine top-k search.

        Args:
            chunks: List of chunk text strings (parallel to embeddings).
            embeddings: np.ndarray [N, d] of L2-normalized float32 embeddings.
        """
        import faiss

        assert embeddings.ndim == 2, (
            f"Expected embeddings shape [N, d], got {embeddings.shape}"
        )
        assert embeddings.shape[1] == self._dim, (
            f"Embedding dim mismatch: expected {self._dim}, got {embeddings.shape[1]}"
        )
        assert len(chunks) == len(embeddings), (
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
        )

        self._chunks = list(chunks)
        emb = embeddings.astype(np.float32)

        # IndexFlatIP: exact inner product search on L2-normalized vectors
        # = exact cosine similarity search (EQ3)
        index = faiss.IndexFlatIP(self._dim)

        if self._use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(emb)
        self._index = index

    def search(self, query_emb: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Retrieve the top-k chunks by cosine similarity to the query.

        Implements EQ3: C_retrieved = argTopK cosine(E(q), E(c))

        Args:
            query_emb: np.ndarray [d] L2-normalized query embedding.
            k: Number of top chunks to retrieve.

        Returns:
            Tuple of (chunk_texts, scores), each a list of length min(k, N).
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")

        assert query_emb.ndim == 1 and len(query_emb) == self._dim, (
            f"Expected query_emb shape [{self._dim}], got {query_emb.shape}"
        )

        q = query_emb.astype(np.float32).reshape(1, -1)
        k_actual = min(k, len(self._chunks))
        scores, indices = self._index.search(q, k_actual)  # [1, k]

        texts = [self._chunks[i] for i in indices[0] if i >= 0]
        score_list = [float(s) for s in scores[0] if s > -1e9]
        return texts, score_list

    def save(self, path: str) -> None:
        """Save index and chunk texts to disk."""
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        with open(path / "meta.json", "w") as f:
            json.dump({"embedding_dim": self._dim, "n_chunks": len(self._chunks)}, f)

    def load(self, path: str) -> None:
        """Load index and chunk texts from disk."""
        import faiss
        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        self._dim = meta["embedding_dim"]

    @property
    def n_chunks(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"FAISSChunkIndex(dim={self._dim}, n_chunks={self.n_chunks})"
