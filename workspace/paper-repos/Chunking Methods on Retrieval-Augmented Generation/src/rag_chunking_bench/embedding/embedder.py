"""
embedding/embedder.py
=====================
Embedding module wrapping BAAI/bge-m3 for chunk and query encoding.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3.1: "the bge-m3 model was employed as the primary retriever"
Section 2.1: EQ3 — cosine similarity retrieval requires E(q) and E(c) in R^d.

bge-m3 embedding dimension: d = 1024 (from model card, SIR conf 0.90).
Embeddings are L2-normalized to enable cosine similarity via dot product.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


class ChunkEmbedder:
    """
    Wrapper around sentence-transformers for bge-m3 encoding.

    Produces L2-normalized dense vectors E(c) and E(q) in R^1024 for use
    in the cosine similarity retrieval formula (EQ3, Section 2.1).

    Args:
        model_name: HuggingFace model identifier. Default: BAAI/bge-m3.
        device: Compute device — 'cpu', 'cuda', or 'auto'.
        batch_size: Number of texts per encode batch.
        normalize: If True, L2-normalize all output vectors (required for EQ3).
        embedding_dim: Expected output dimension for shape validation.
            SIR confidence 0.90 — from bge-m3 spec, not stated in paper.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = True,
        embedding_dim: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._embedding_dim = embedding_dim

        # Resolve device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device
        self._model = None  # lazy-loaded

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)

    def encode(
        self,
        texts: List[str],
        normalize: bool | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Returns E(c) for all chunks c in C (EQ3 component).

        Args:
            texts: List of text strings to encode.
            normalize: Override instance normalize setting if provided.
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape [N, embedding_dim] (float32).
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        self._load()
        norm = normalize if normalize is not None else self._normalize
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=norm,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        assert embeddings.shape[1] == self._embedding_dim, (
            f"Expected embedding dim {self._embedding_dim}, got {embeddings.shape[1]}. "
            f"If using a different model, update embedding.embedding_dim in config."
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str, normalize: bool | None = None) -> np.ndarray:
        """
        Encode a single query string into a dense vector E(q).

        Args:
            query: Query text string.
            normalize: Override instance normalize setting.

        Returns:
            np.ndarray of shape [embedding_dim] (float32).
        """
        result = self.encode([query], normalize=normalize)
        return result[0]

    def __repr__(self) -> str:
        return (
            f"ChunkEmbedder(model={self._model_name}, "
            f"device={self._device}, dim={self._embedding_dim})"
        )
