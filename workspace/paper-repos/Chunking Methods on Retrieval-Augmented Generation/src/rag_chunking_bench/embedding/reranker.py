"""
embedding/reranker.py
=====================
Reranker module wrapping BAAI/bge-reranker-v2-m3.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3.1: "bge-reranker-v2-m3 served as the reranking model."

The reranker is a cross-encoder: it takes (query, chunk) pairs and produces
relevance scores, which are used to re-sort retrieved chunks before passing
them to the generator or evaluation.
"""

from __future__ import annotations

from typing import List, Tuple


class ChunkReranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Receives the top-k retrieved chunks from the FAISS index and re-scores
    each (query, chunk) pair to produce a better-ranked list.

    Paper reference: Section 3.1 (Evidence Retrieval experiment).

    Args:
        model_name: HuggingFace model ID. Default: BAAI/bge-reranker-v2-m3.
        device: Compute device — 'cpu', 'cuda', or 'auto'.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
    ) -> None:
        self._model_name = model_name

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
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, device=self._device)

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Re-score and re-rank retrieved chunks for a given query.

        Args:
            query: Query string.
            chunks: List of candidate chunk strings from initial retrieval.
            top_k: Number of top chunks to return after reranking.

        Returns:
            List of (chunk, score) tuples sorted by score descending,
            truncated to top_k.
        """
        if not chunks:
            return []

        self._load()
        pairs = [(query, chunk) for chunk in chunks]
        scores = self._model.predict(pairs)  # np.ndarray of floats

        ranked = sorted(
            zip(chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def rerank_texts(self, query: str, chunks: List[str], top_k: int = 10) -> List[str]:
        """
        Convenience wrapper returning only chunk texts (no scores).

        Args:
            query: Query string.
            chunks: Candidate chunks.
            top_k: Number to return.

        Returns:
            Top-k reranked chunk strings.
        """
        ranked = self.rerank(query, chunks, top_k)
        return [text for text, _ in ranked]

    def __repr__(self) -> str:
        return f"ChunkReranker(model={self._model_name}, device={self._device})"
