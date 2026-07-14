"""
tests/test_metrics.py
=====================
Unit tests for EQ4, EQ5, EQ6 metric implementations.

Paper: arXiv:2606.00881, Section 3.1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from rag_chunking_bench.evaluation.metrics import RetrievalMetrics
from rag_chunking_bench.utils.text_utils import spans_overlap


class TestSpansOverlap:
    """EQ4: relevant(c, q) = 1[c ∩ answer_span(q) ≠ ∅]"""

    def test_exact_match(self):
        assert spans_overlap("The cat sat on the mat", "The cat sat on the mat")

    def test_substring_match(self):
        assert spans_overlap("The cat sat on the mat near the window", "cat sat on the mat")

    def test_no_overlap(self):
        assert not spans_overlap("The cat is sleeping", "dog barked loudly")

    def test_empty_answer_span(self):
        assert not spans_overlap("some chunk", "")

    def test_empty_chunk(self):
        assert not spans_overlap("", "answer span")

    def test_case_insensitive(self):
        assert spans_overlap("The Quick Brown Fox", "quick brown fox")


class TestAccuracyAtK:
    """EQ5: Accuracy@k = 1 if ∃ relevant chunk in top-k, else 0"""

    def test_relevant_in_top_5(self):
        retrieved = ["chunk about the answer", "irrelevant chunk", "another irrelevant"]
        relevant = ["answer"]
        score = RetrievalMetrics.accuracy_at_k(retrieved, relevant, k=5)
        assert score == 1.0

    def test_relevant_outside_top_k(self):
        retrieved = ["irrelevant1", "irrelevant2", "irrelevant3", "irrelevant4",
                     "irrelevant5", "contains answer text"]
        relevant = ["answer text"]
        score = RetrievalMetrics.accuracy_at_k(retrieved, relevant, k=5)
        assert score == 0.0

    def test_empty_relevant(self):
        retrieved = ["chunk one", "chunk two"]
        score = RetrievalMetrics.accuracy_at_k(retrieved, [], k=5)
        assert score == 0.0

    def test_empty_retrieved(self):
        score = RetrievalMetrics.accuracy_at_k([], ["answer"], k=5)
        assert score == 0.0

    def test_exact_chunk_match(self):
        chunk = "The mitochondria is the powerhouse of the cell"
        retrieved = [chunk, "unrelated text"]
        relevant = [chunk]
        score = RetrievalMetrics.accuracy_at_k(retrieved, relevant, k=5)
        assert score == 1.0


class TestRecallAtK:
    """EQ6: Recall@k = |retrieved ∩ relevant| / |relevant|"""

    def test_all_relevant_retrieved(self):
        rel1 = "first relevant chunk"
        rel2 = "second relevant chunk"
        retrieved = [rel1, rel2, "noise"]
        relevant = [rel1, rel2]
        score = RetrievalMetrics.recall_at_k(retrieved, relevant, k=10)
        assert score == 1.0

    def test_partial_recall(self):
        rel1 = "first relevant chunk"
        rel2 = "second relevant chunk"
        retrieved = [rel1, "noise", "more noise"]
        relevant = [rel1, rel2]
        score = RetrievalMetrics.recall_at_k(retrieved, relevant, k=10)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_zero_recall(self):
        retrieved = ["noise1", "noise2"]
        relevant = ["relevant content"]
        score = RetrievalMetrics.recall_at_k(retrieved, relevant, k=10)
        assert score == 0.0

    def test_empty_relevant(self):
        score = RetrievalMetrics.recall_at_k(["chunk"], [], k=10)
        assert score == 0.0

    def test_cutoff_at_k(self):
        relevant_chunk = "the answer is here"
        retrieved = ["noise"] * 9 + [relevant_chunk]  # relevant at position 10
        relevant = [relevant_chunk]
        # At k=9, shouldn't find it
        score_9 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=9)
        score_10 = RetrievalMetrics.recall_at_k(retrieved, relevant, k=10)
        assert score_9 == 0.0
        assert score_10 == 1.0


class TestComputeAll:
    """Test aggregate metric computation over multiple query results."""

    def test_basic_aggregation(self):
        results = [
            {"retrieved_chunks": ["answer here", "noise"], "relevant_chunks": ["answer here"]},
            {"retrieved_chunks": ["noise1", "noise2"], "relevant_chunks": ["missing"]},
        ]
        agg = RetrievalMetrics.compute_all(results, k_acc=5, k_rec=10)
        assert "accuracy_at_5" in agg
        assert "recall_at_10" in agg
        # First query: hit; second: miss → avg accuracy = 0.5
        assert agg["accuracy_at_5"] == pytest.approx(0.5, abs=0.01)

    def test_empty_results(self):
        agg = RetrievalMetrics.compute_all([], k_acc=5, k_rec=10)
        assert agg["accuracy_at_5"] == 0.0
        assert agg["recall_at_10"] == 0.0
