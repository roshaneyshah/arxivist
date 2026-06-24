"""
pipeline/eval_pipeline.py
==========================
Orchestrates Experiment 1 (evidence retrieval) and Experiment 2
(end-to-end RAG answer generation) for a given (chunker, dataset) pair.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3.1: Evidence Retrieval — Accuracy@5, Recall@10
Section 3.2: End-to-End RAG Answer Generation — LLM-as-a-Judge (1-5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rag_chunking_bench.data.dataset_loader import DatasetLoader
from rag_chunking_bench.evaluation.metrics import RetrievalMetrics
from rag_chunking_bench.evaluation.reporter import ResultReporter, FAILURE_T, FAILURE_S
from rag_chunking_bench.generation.generator import RAGGenerator
from rag_chunking_bench.generation.judge import LLMJudge
from rag_chunking_bench.retrieval.retriever import RAGRetriever
from rag_chunking_bench.utils.text_utils import find_relevant_chunks

logger = logging.getLogger(__name__)


class EvalPipeline:
    """
    Runs EXP1 and EXP2 evaluations given a pre-built retriever.

    Args:
        retriever: RAGRetriever (embedding model + FAISS index + optional reranker).
        metrics: RetrievalMetrics instance.
        reporter: ResultReporter for logging and export.
        all_chunks: All chunks in the index (for relevance fallback in EQ4).
        generator: Optional RAGGenerator for EXP2. None skips generation.
        judge: Optional LLMJudge for EXP2 scoring. None skips scoring.
        top_k_acc: k for Accuracy@k (default 5).
        top_k_rec: k for Recall@k (default 10).
        top_k_gen: k chunks to pass to generator (default 5).
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        metrics: RetrievalMetrics,
        reporter: ResultReporter,
        all_chunks: List[str],
        generator: Optional[RAGGenerator] = None,
        judge: Optional[LLMJudge] = None,
        top_k_acc: int = 5,    # Accuracy@5 — EQ5; SIR conf 0.99
        top_k_rec: int = 10,   # Recall@10 — EQ6; SIR conf 0.99
        top_k_gen: int = 5,    # chunks to generator; SIR conf 0.99
    ) -> None:
        self._retriever = retriever
        self._metrics = metrics
        self._reporter = reporter
        self._all_chunks = all_chunks
        self._generator = generator
        self._judge = judge
        self._top_k_acc = top_k_acc
        self._top_k_rec = top_k_rec
        self._top_k_gen = top_k_gen

    # ------------------------------------------------------------------
    # Experiment 1: Evidence Retrieval (Section 3.1)
    # ------------------------------------------------------------------

    def run_retrieval_eval(
        self,
        queries: List[Dict],
        method_name: str,
        dataset_name: str,
    ) -> Dict:
        """
        Run Experiment 1: evidence retrieval evaluation.

        For each query:
          1. Retrieve top-k_rec chunks via RAGRetriever (EQ3 + reranking)
          2. Compute Accuracy@k_acc (EQ5) and Recall@k_rec (EQ6)
          3. Record results in reporter

        Paper Section 3.1: "A chunk was considered relevant if it overlapped
        with the span of the extractive answer. In cases where no extractive
        answer was available, all chunks originating from the relevant document
        were treated as relevant."

        Args:
            queries: List of query dicts from DatasetLoader.load_queries().
            method_name: Chunker identifier for reporting.
            dataset_name: Dataset identifier for reporting.

        Returns:
            Dict with 'accuracy_at_k' and 'recall_at_k' float values.
        """
        logger.info(
            f"[EXP1] {method_name} / {dataset_name}: "
            f"evaluating {len(queries)} queries"
        )
        results = []
        top_k = max(self._top_k_acc, self._top_k_rec)

        for qr in queries:
            query_text = qr["query"]
            answer_span = qr.get("answer_span", "")

            # Retrieve candidates (EQ3 + reranking)
            retrieved = self._retriever.retrieve(query_text, k=top_k)

            # Determine relevant chunks (EQ4)
            relevant = find_relevant_chunks(
                self._all_chunks,
                answer_span=answer_span,
                all_doc_chunks=self._all_chunks if not answer_span else None,
            )

            results.append({
                "retrieved_chunks": retrieved,
                "relevant_chunks": relevant,
            })

        # Aggregate metrics (EQ5, EQ6)
        agg = self._metrics.compute_all(
            results, k_acc=self._top_k_acc, k_rec=self._top_k_rec
        )

        acc = agg[f"accuracy_at_{self._top_k_acc}"]
        rec = agg[f"recall_at_{self._top_k_rec}"]

        self._reporter.record(method_name, dataset_name, "accuracy_at_5", acc)
        self._reporter.record(method_name, dataset_name, "recall_at_10", rec)

        logger.info(
            f"[EXP1] {method_name}/{dataset_name} → "
            f"Accuracy@{self._top_k_acc}={acc*100:.2f}%, "
            f"Recall@{self._top_k_rec}={rec*100:.2f}%"
        )
        return agg

    # ------------------------------------------------------------------
    # Experiment 2: End-to-End RAG Generation (Section 3.2)
    # ------------------------------------------------------------------

    def run_generation_eval(
        self,
        queries: List[Dict],
        method_name: str,
        dataset_name: str,
    ) -> Dict:
        """
        Run Experiment 2: end-to-end RAG answer generation + LLM-judge scoring.

        For each query:
          1. Retrieve top-5 chunks
          2. Generate answer with RAGGenerator (GPT-OSS-20B, ≤4000 tokens)
          3. Score with LLMJudge (5-point Likert scale)

        Paper Section 3.2: "The generated responses were then evaluated against
        ground-truth answers using an LLM-as-a-judge approach."

        NOTE: TextTiling is excluded from this experiment per paper Section 3.2.

        Args:
            queries: Query dicts (must include 'ground_truth' key).
            method_name: Chunker identifier.
            dataset_name: Dataset identifier.

        Returns:
            Dict with 'avg_llm_judge' score.
        """
        if self._generator is None or self._judge is None:
            logger.warning("[EXP2] Generator or judge not configured — skipping.")
            return {}

        if method_name == "texttiling":
            logger.info(
                "[EXP2] TextTiling excluded from Experiment 2 per paper Section 3.2."
            )
            return {}

        logger.info(
            f"[EXP2] {method_name} / {dataset_name}: "
            f"generating for {len(queries)} queries"
        )

        scores = []
        for qr in queries:
            query_text = qr["query"]
            ground_truth = qr.get("ground_truth", qr.get("answer_span", ""))

            if not ground_truth:
                # Skip queries with no ground truth (paper: "missing answers")
                continue

            # Step 1: retrieve top-5 chunks
            top5 = self._retriever.retrieve(query_text, k=self._top_k_gen)

            # Step 2: generate answer (≤4000 token context, paper Section 3.2)
            answer = self._generator.generate(query_text, top5)

            # Step 3: judge answer quality (1-5 Likert scale)
            score = self._judge.score(query_text, answer, ground_truth)
            scores.append(score)

        if not scores:
            logger.warning(f"[EXP2] No scoreable queries for {method_name}/{dataset_name}")
            return {}

        avg_score = sum(scores) / len(scores)
        self._reporter.record(method_name, dataset_name, "llm_judge", avg_score)

        logger.info(
            f"[EXP2] {method_name}/{dataset_name} → "
            f"avg LLM-judge = {avg_score:.2f} (n={len(scores)})"
        )
        return {"avg_llm_judge": avg_score, "n_scored": len(scores)}
