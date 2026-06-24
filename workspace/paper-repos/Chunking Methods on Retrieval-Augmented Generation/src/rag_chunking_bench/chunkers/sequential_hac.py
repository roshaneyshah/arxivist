"""
chunkers/sequential_hac.py
==========================
Sequential Hierarchical Agglomerative Chunking (Sequential HAC).

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 2.2: "Sequential Hierarchical Agglomerative Chunking merges adjacent
sentences based on semantic similarity while enforcing a strict structural
constraint, ensuring narrative continuity without tuning positional weights or
predefining the number of chunks."

Original paper: Qu et al. 2025, "Is semantic chunking worth the computational
cost?" NAACL Findings 2025.

Results (Table 1 Accuracy@5 avg): 80.09%
Results (Table 2 Recall@10 avg):  33.85%
Results (Table 3 time avg):        2.10m
"""

from __future__ import annotations

from typing import List

import numpy as np

from rag_chunking_bench.chunkers.base import BaseChunker
from rag_chunking_bench.utils.text_utils import split_sentences


class SequentialHACChunker(BaseChunker):
    """
    Agglomerative chunker that merges adjacent sentences by cosine similarity.

    Algorithm:
    1. Split document into sentences.
    2. Embed all sentences with bge-m3.
    3. Greedily merge adjacent sentence pairs whose cosine similarity exceeds
       similarity_threshold, subject to max_chunk_tokens.
    4. Repeat until no more merges are possible.

    This enforces narrative continuity (only adjacent sentences merge) and
    avoids positional weight tuning — matching the description in Section 2.2.

    Args:
        config: Dict with keys:
            - similarity_threshold (float): ASSUMED 0.85 from Qu et al. 2025 defaults.
              SIR confidence: 0.88 — cross-reference original paper.
            - max_chunk_tokens (int): Maximum tokens per chunk (default 512).
            - embed_model (str): Sentence embedding model.
    """

    def __init__(self, config: dict | None = None) -> None:
        raw = config or {}
        # ASSUMED: threshold from Qu et al. 2025 defaults — SIR conf 0.88
        similarity_threshold: float = raw.get("similarity_threshold", 0.85)
        max_chunk_tokens: int = raw.get("max_chunk_tokens", 512)
        embed_model: str = raw.get("embed_model", "BAAI/bge-m3")

        super().__init__({
            "similarity_threshold": similarity_threshold,
            "max_chunk_tokens": max_chunk_tokens,
            "embed_model": embed_model,
        })
        self._threshold = similarity_threshold
        self._max_tokens = max_chunk_tokens
        self._embed_model_name = embed_model
        self._embedder = None  # lazy-loaded

    def _load_embedder(self):
        """Lazy-load sentence-transformers model to avoid import overhead."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name)

    @property
    def name(self) -> str:
        return "sequential_hac"

    def chunk(self, document: str) -> List[str]:
        """
        Segment document via sequential agglomerative merging.

        EQ2: f_theta(D) -> C, theta = {similarity_threshold, max_chunk_tokens}

        Args:
            document: Raw text string.

        Returns:
            List of text chunks with narrative continuity preserved.
        """
        if not document.strip():
            return []

        sentences = split_sentences(document)
        if len(sentences) == 0:
            return [document]
        if len(sentences) == 1:
            return sentences

        self._load_embedder()
        embeddings = self._embed_sentences(sentences)
        return self._agglomerate(sentences, embeddings)

    def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Embed a list of sentences using bge-m3.

        Returns:
            np.ndarray of shape [N, d] with L2-normalized embeddings.
        """
        return self._embedder.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors (EQ3 component)."""
        # Both vectors already normalized, so dot product = cosine similarity
        return float(np.dot(a, b))

    def _token_count(self, text: str) -> int:
        """Approximate token count: characters / 4."""
        return len(text) // 4

    def _agglomerate(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[str]:
        """
        Greedily merge adjacent sentences whose cosine similarity >= threshold.

        Enforces max_chunk_tokens to prevent unbounded chunk growth.
        Only adjacent sentences are merged (no cross-segment merging),
        preserving narrative continuity as described in Section 2.2.

        Args:
            sentences: List of sentence strings.
            embeddings: np.ndarray [N, d] of L2-normalized embeddings.

        Returns:
            List of merged chunk strings.
        """
        # Represent each current "chunk" as (text, mean_embedding)
        chunks: List[str] = list(sentences)
        chunk_embs: List[np.ndarray] = list(embeddings)

        merged = True
        while merged:
            merged = False
            new_chunks: List[str] = []
            new_embs: List[np.ndarray] = []
            i = 0
            while i < len(chunks):
                if i < len(chunks) - 1:
                    sim = self._cosine_sim(chunk_embs[i], chunk_embs[i + 1])
                    combined = chunks[i] + " " + chunks[i + 1]
                    would_exceed = self._token_count(combined) > self._max_tokens
                    if sim >= self._threshold and not would_exceed:
                        # Merge: average embeddings as proxy for merged chunk emb
                        merged_emb = (chunk_embs[i] + chunk_embs[i + 1]) / 2.0
                        norm = np.linalg.norm(merged_emb)
                        merged_emb = merged_emb / (norm + 1e-8)
                        new_chunks.append(combined)
                        new_embs.append(merged_emb)
                        i += 2
                        merged = True
                        continue
                new_chunks.append(chunks[i])
                new_embs.append(chunk_embs[i])
                i += 1
            chunks = new_chunks
            chunk_embs = new_embs

        return chunks

    def __repr__(self) -> str:
        return (
            f"SequentialHACChunker("
            f"threshold={self._threshold}, max_tokens={self._max_tokens})"
        )
