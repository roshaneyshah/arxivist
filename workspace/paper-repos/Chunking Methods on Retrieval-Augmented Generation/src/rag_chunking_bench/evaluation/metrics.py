"""
evaluation/metrics.py
=====================
Retrieval evaluation metrics: Accuracy@k (EQ5) and Recall@k (EQ6).

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3.1: Evidence Retrieval evaluation methodology.

Equations implemented:
  EQ4: relevant(c, q) = 1[c ∩ answer_span(q) ≠ ∅]
  EQ5: Accuracy@k = (1/|Q|) Σ_q 1[∃c ∈ C_retrieved^(k): relevant(c,q)=1]
  EQ6: Recall@k   = (1/|Q|) Σ_q |C_retrieved^(k) ∩ C_relevant(q)| / |C_relevant(q)|
"""

from __future__ import annotations

from typing import Dict, List

from rag_chunking_bench.utils.text_utils import spans_overlap


class RetrievalMetrics:
    """
    Computes Accuracy@k and Recall@k for evidence retrieval evaluation.

    Paper: Section 3.1; Tables 1 and 2.
    """

    @staticmethod
    def is_relevant(chunk: str, answer_span: str) -> bool:
        """
        EQ4: Check whether chunk overlaps with the extractive answer span.

            relevant(c, q) = 1[c ∩ answer_span(q) ≠ ∅]

        Args:
            chunk: Text chunk string c_i.
            answer_span: Ground-truth extractive answer text.

        Returns:
            True if chunk overlaps with answer_span.
        """
        return spans_overlap(chunk, answer_span)

    @staticmethod
    def accuracy_at_k(
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 5,
    ) -> float:
        """
        EQ5: Accuracy@k — 1 if any retrieved chunk is relevant, else 0.

            Accuracy@k = 1[∃c ∈ C_retrieved^(k): relevant(c,q)=1]

        For a single query. Averaged over all queries in compute_all().

        Args:
            retrieved_chunks: Ordered list of retrieved chunks (all candidates).
            relevant_chunks: List of ground-truth relevant chunks.
            k: Cutoff position (default 5 for Accuracy@5).

        Returns:
            1.0 if any chunk in top-k is relevant, 0.0 otherwise.
        """
        top_k = retrieved_chunks[:k]
        relevant_set = set(relevant_chunks)

        for chunk in top_k:
            # Check exact match first, then substring overlap
            if chunk in relevant_set:
                return 1.0
            for rel in relevant_chunks:
                if spans_overlap(chunk, rel):
                    return 1.0
        return 0.0

    @staticmethod
    def recall_at_k(
        retrieved_chunks: List[str],
        relevant_chunks: List[str],
        k: int = 10,
    ) -> float:
        """
        EQ6: Recall@k — fraction of relevant chunks found in top-k.

            Recall@k = |C_retrieved^(k) ∩ C_relevant(q)| / |C_relevant(q)|

        For a single query. Averaged over all queries in compute_all().

        Args:
            retrieved_chunks: Ordered list of retrieved chunks.
            relevant_chunks: List of ground-truth relevant chunks.
            k: Cutoff position (default 10 for Recall@10).

        Returns:
            Float in [0, 1].
        """
        if not relevant_chunks:
            return 0.0

        top_k = retrieved_chunks[:k]
        hits = 0
        for rel in relevant_chunks:
            for chunk in top_k:
                if spans_overlap(chunk, rel):
                    hits += 1
                    break  # count each relevant chunk at most once

        return hits / len(relevant_chunks)

    @classmethod
    def compute_all(
        cls,
        results: List[Dict],
        k_acc: int = 5,
        k_rec: int = 10,
    ) -> Dict[str, float]:
        """
        Compute average Accuracy@k and Recall@k over all query results.

        Args:
            results: List of dicts, each with keys:
                - 'retrieved_chunks': list[str]
                - 'relevant_chunks': list[str]
            k_acc: k for Accuracy (default 5).
            k_rec: k for Recall (default 10).

        Returns:
            Dict with keys 'accuracy_at_{k}' and 'recall_at_{k}'.
        """
        if not results:
            return {f"accuracy_at_{k_acc}": 0.0, f"recall_at_{k_rec}": 0.0}

        acc_scores = []
        rec_scores = []
        for r in results:
            retrieved = r["retrieved_chunks"]
            relevant = r["relevant_chunks"]
            acc_scores.append(cls.accuracy_at_k(retrieved, relevant, k_acc))
            rec_scores.append(cls.recall_at_k(retrieved, relevant, k_rec))

        return {
            f"accuracy_at_{k_acc}": float(sum(acc_scores) / len(acc_scores)),
            f"recall_at_{k_rec}": float(sum(rec_scores) / len(rec_scores)),
        }
