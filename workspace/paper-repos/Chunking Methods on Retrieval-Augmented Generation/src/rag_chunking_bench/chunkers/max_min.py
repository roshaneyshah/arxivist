"""
chunkers/max_min.py
===================
Max-Min Semantic Chunking method.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 2.2: "Max-Min Semantic Chunking uses a greedy strategy that sequentially
adds sentences to a chunk only if their maximum similarity exceeds an adaptive
threshold derived from the chunk's minimum coherence, dynamically controlling
chunk growth."

Original: Kiss et al. 2025, "Max-min semantic chunking", Discover Computing 28.

Results (Table 1 Accuracy@5 avg): 85.75%
Results (Table 2 Recall@10 avg):  41.68%
Results (Table 3 time avg):        2.44m
"""

from __future__ import annotations

from typing import List

import numpy as np

from rag_chunking_bench.chunkers.base import BaseChunker
from rag_chunking_bench.utils.text_utils import split_sentences


class MaxMinChunker(BaseChunker):
    """
    Greedy max-min semantic chunker.

    Algorithm:
    1. Split document into sentences and embed with bge-m3.
    2. Start a new chunk with the first sentence.
    3. For each subsequent sentence s:
       a. Compute max similarity between s and all sentences in current chunk.
       b. Compute adaptive threshold = alpha * min(pairwise similarities in chunk).
       c. If max_sim >= adaptive_threshold: extend current chunk with s.
       d. Else: start a new chunk with s.

    This dynamically controls chunk growth based on the chunk's own coherence,
    as described in Section 2.2 of the paper.

    Args:
        config: Dict with keys:
            - alpha (float): Threshold scaling factor.
              ASSUMED 0.5 from Kiss et al. 2025 defaults. SIR conf 0.83.
            - embed_model (str): Sentence embedding model.
    """

    def __init__(self, config: dict | None = None) -> None:
        raw = config or {}
        # ASSUMED: alpha from Kiss et al. 2025 — SIR conf 0.83
        alpha: float = raw.get("alpha", 0.5)
        embed_model: str = raw.get("embed_model", "BAAI/bge-m3")

        super().__init__({"alpha": alpha, "embed_model": embed_model})
        self._alpha = alpha
        self._embed_model_name = embed_model
        self._embedder = None  # lazy-loaded

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name)

    @property
    def name(self) -> str:
        return "max_min"

    def chunk(self, document: str) -> List[str]:
        """
        Segment document using greedy max-min adaptive threshold strategy.

        EQ2: f_theta(D) -> C, theta = {alpha}

        Args:
            document: Raw text string.

        Returns:
            List of semantically coherent text chunks.
        """
        if not document.strip():
            return []

        sentences = split_sentences(document)
        if len(sentences) <= 1:
            return [document] if document.strip() else []

        self._load_embedder()
        embeddings: np.ndarray = self._embedder.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )  # shape: [N, d]

        return self._greedy_chunk(sentences, embeddings)

    def _compute_adaptive_threshold(self, chunk_embs: np.ndarray) -> float:
        """
        Compute adaptive threshold = alpha * min pairwise cosine similarity
        among sentences already in the current chunk.

        When chunk has only 1 sentence, no pairwise similarities exist;
        return alpha as a base threshold.

        Args:
            chunk_embs: np.ndarray [k, d] of L2-normalized sentence embeddings.

        Returns:
            Adaptive threshold float.
        """
        if len(chunk_embs) < 2:
            return self._alpha  # base threshold for singleton chunk

        # Pairwise cosine similarities (vectors already L2-normalized)
        sim_matrix = chunk_embs @ chunk_embs.T  # [k, k]
        # Exclude diagonal (self-similarity = 1.0)
        k = len(chunk_embs)
        mask = ~np.eye(k, dtype=bool)
        pairwise = sim_matrix[mask]
        min_coherence = float(pairwise.min())
        # Adaptive threshold = alpha * minimum coherence among chunk sentences
        return self._alpha * min_coherence

    def _should_extend(
        self,
        sentence_emb: np.ndarray,
        chunk_embs: np.ndarray,
        threshold: float,
    ) -> bool:
        """
        Decide whether to add sentence to current chunk.

        Rule: extend if max(cosine_sim(sentence, c_j) for c_j in chunk) >= threshold.

        Args:
            sentence_emb: [d] L2-normalized embedding of candidate sentence.
            chunk_embs: [k, d] embeddings of sentences in current chunk.
            threshold: Adaptive threshold from _compute_adaptive_threshold.

        Returns:
            True if sentence should be added to current chunk.
        """
        # Max cosine similarity between sentence and any sentence in current chunk
        sims = chunk_embs @ sentence_emb  # [k]
        max_sim = float(sims.max())
        return max_sim >= threshold

    def _greedy_chunk(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[str]:
        """
        Main greedy chunking loop.

        Args:
            sentences: List of sentence strings.
            embeddings: [N, d] L2-normalized embeddings.

        Returns:
            List of chunk strings.
        """
        chunks: List[str] = []
        current_sentences: List[str] = [sentences[0]]
        current_embs: List[np.ndarray] = [embeddings[0]]

        for i in range(1, len(sentences)):
            sent = sentences[i]
            emb = embeddings[i]
            chunk_embs_arr = np.stack(current_embs)  # [k, d]

            threshold = self._compute_adaptive_threshold(chunk_embs_arr)

            if self._should_extend(emb, chunk_embs_arr, threshold):
                current_sentences.append(sent)
                current_embs.append(emb)
            else:
                # Finalize current chunk and start a new one
                chunks.append(" ".join(current_sentences))
                current_sentences = [sent]
                current_embs = [emb]

        # Flush last chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def __repr__(self) -> str:
        return f"MaxMinChunker(alpha={self._alpha})"
