"""
retrieval/retriever.py
======================
End-to-end retrieval pipeline: embed query → search index → rerank.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3.1: "bge-m3 model was employed as the primary retriever, while
bge-reranker-v2-m3 served as the reranking model."
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from rag_chunking_bench.embedding.embedder import ChunkEmbedder
from rag_chunking_bench.embedding.reranker import ChunkReranker
from rag_chunking_bench.retrieval.index import FAISSChunkIndex


class RAGRetriever:
    """
    Two-stage retriever: dense retrieval (bge-m3 + FAISS) → reranking (bge-reranker-v2-m3).

    Stage 1: embed query with bge-m3, search FAISS index (EQ3).
    Stage 2: rerank top-k candidates with bge-reranker-v2-m3 cross-encoder.

    Args:
        index: Pre-built FAISSChunkIndex.
        embedder: ChunkEmbedder for query encoding.
        reranker: Optional ChunkReranker. If None, returns raw retrieval order.
        top_k_index: Number of candidates to retrieve before reranking.
    """

    def __init__(
        self,
        index: FAISSChunkIndex,
        embedder: ChunkEmbedder,
        reranker: Optional[ChunkReranker] = None,
        top_k_index: int = 10,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self._top_k_index = top_k_index

    def retrieve(self, query: str, k: int = 10) -> List[str]:
        """
        Retrieve top-k relevant chunks for a query.

        Args:
            query: Query string.
            k: Number of chunks to return after reranking.

        Returns:
            List of up to k chunk strings, ordered by relevance.
        """
        return [text for text, _ in self.retrieve_with_scores(query, k)]

    def retrieve_with_scores(
        self, query: str, k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks with their relevance scores.

        Pipeline:
          1. encode_query → E(q): [d]
          2. index.search(E(q), top_k_index) → candidates + cosine scores (EQ3)
          3. reranker.rerank(query, candidates, k) → reranked (chunk, score) pairs

        Args:
            query: Query string.
            k: Number of final chunks to return.

        Returns:
            List of (chunk_text, score) tuples sorted by score descending.
        """
        # Step 1: encode query → E(q)
        query_emb = self._embedder.encode_query(query)

        # Step 2: dense retrieval via cosine similarity (EQ3)
        candidates, scores = self._index.search(query_emb, k=self._top_k_index)

        if not candidates:
            return []

        # Step 3: optional cross-encoder reranking
        if self._reranker is not None:
            return self._reranker.rerank(query, candidates, top_k=k)

        # No reranker: return raw retrieval scores
        return list(zip(candidates, scores))[:k]

    def __repr__(self) -> str:
        return (
            f"RAGRetriever(index={self._index}, "
            f"reranker={'yes' if self._reranker else 'no'})"
        )
