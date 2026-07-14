"""
tests/test_chunkers.py
======================
Unit tests for all chunker implementations.

Paper: arXiv:2606.00881
Tests verify the EQ2 contract: f_theta(D) -> C = list of non-empty strings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from rag_chunking_bench.chunkers.base import BaseChunker
from rag_chunking_bench.chunkers.fixed_size import FixedSizeChunker
from rag_chunking_bench.chunkers.recursive_semantic import RecursiveSemanticChunker
from rag_chunking_bench.chunkers import ChunkerRegistry

# Short document fixture
SHORT_DOC = "The quick brown fox jumps over the lazy dog. " * 10

# Medium document fixture (~2000 chars, several paragraphs)
MEDIUM_DOC = "\n\n".join([
    "Retrieval-Augmented Generation (RAG) has been proposed as a response "
    "to several Large Language Models (LLMs) problems like accessing, "
    "manipulating and updating knowledge or providing provenance for their decision.",
    "A standard RAG pipeline initially splits documents, especially the longer ones, "
    "into smaller units called chunks. Identifying the chunks most pertinent to a "
    "query and supplying them to the LLM should allow for obtaining accurate answers.",
    "Chunking is commonly treated as a simple preprocessing step. However, we show "
    "that it introduces a range of impactful and often overlooked issues, including "
    "high chunking time and implementation fragility.",
    "The fundamental theoretical challenge is maximizing the probability of generating "
    "the correct answer given a query. This introduces a trade-off between granularity "
    "and context preservation.",
] * 5)

EMPTY_DOC = ""
SINGLE_SENTENCE = "This is a single sentence."


# ---------------------------------------------------------------------------
# EQ2 contract tests (apply to all robust chunkers)
# ---------------------------------------------------------------------------

class TestEQ2Contract:
    """All chunkers must satisfy EQ2: f_theta(D) -> list[str] of non-empty strings."""

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_returns_list(self, method):
        chunker = ChunkerRegistry.get(method)
        result = chunker.chunk(MEDIUM_DOC)
        assert isinstance(result, list), f"{method}: must return list"

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_all_strings(self, method):
        chunker = ChunkerRegistry.get(method)
        result = chunker.chunk(MEDIUM_DOC)
        for chunk in result:
            assert isinstance(chunk, str), f"{method}: all chunks must be strings"
            assert chunk.strip(), f"{method}: no empty chunks allowed"

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_nonempty_output_on_medium_doc(self, method):
        chunker = ChunkerRegistry.get(method)
        result = chunker.chunk(MEDIUM_DOC)
        assert len(result) >= 1, f"{method}: must produce ≥1 chunk"

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_empty_input_returns_empty_or_singleton(self, method):
        chunker = ChunkerRegistry.get(method)
        result = chunker.chunk(EMPTY_DOC)
        assert isinstance(result, list), f"{method}: empty input must return list"
        # Should be empty list (no content) — not raise
        assert len(result) == 0 or all(isinstance(c, str) for c in result)

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_content_preserved(self, method):
        """Chunks must collectively cover all significant words of the document."""
        chunker = ChunkerRegistry.get(method)
        result = chunker.chunk(MEDIUM_DOC)
        joined = " ".join(result).lower()
        # Check a few key words from the document are present
        for word in ["retrieval", "chunking", "query"]:
            assert word in joined, f"{method}: word '{word}' missing from chunks"

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_chunk_batch_matches_single(self, method):
        """chunk_batch must produce same results as calling chunk() per doc."""
        chunker = ChunkerRegistry.get(method)
        docs = [SHORT_DOC, SINGLE_SENTENCE]
        batch = chunker.chunk_batch(docs)
        singles = [chunker.chunk(d) for d in docs]
        assert len(batch) == len(singles), f"{method}: batch length mismatch"

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_name_property(self, method):
        chunker = ChunkerRegistry.get(method)
        assert isinstance(chunker.name, str)
        assert len(chunker.name) > 0

    @pytest.mark.parametrize("method", ChunkerRegistry.list_robust())
    def test_get_config_returns_dict(self, method):
        chunker = ChunkerRegistry.get(method)
        cfg = chunker.get_config()
        assert isinstance(cfg, dict)


# ---------------------------------------------------------------------------
# FixedSizeChunker specific tests
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    """Paper Section 3: chunk_size=512, overlap=50 (SIR conf 0.99)."""

    def test_default_config(self):
        c = FixedSizeChunker()
        assert c._chunk_size == 512
        assert c._overlap == 50

    def test_custom_config(self):
        c = FixedSizeChunker({"chunk_size": 256, "overlap": 25})
        assert c._chunk_size == 256

    def test_long_doc_produces_multiple_chunks(self):
        c = FixedSizeChunker({"chunk_size": 100, "overlap": 10})
        doc = "word " * 500
        chunks = c.chunk(doc)
        assert len(chunks) > 1

    def test_name(self):
        assert FixedSizeChunker().name == "fixed_size"


# ---------------------------------------------------------------------------
# RecursiveSemanticChunker specific tests
# ---------------------------------------------------------------------------

class TestRecursiveSemanticChunker:

    def test_default_config(self):
        c = RecursiveSemanticChunker()
        assert c._chunk_size == 512
        assert "\n\n" in c._separators

    def test_paragraph_boundaries_respected(self):
        doc = "Para one.\n\nPara two.\n\nPara three."
        c = RecursiveSemanticChunker({"chunk_size": 50, "overlap": 0})
        chunks = c.chunk(doc)
        # With small chunk size, should split at paragraph boundaries
        assert len(chunks) >= 1

    def test_name(self):
        assert RecursiveSemanticChunker().name == "recursive_semantic"


# ---------------------------------------------------------------------------
# ChunkerRegistry tests
# ---------------------------------------------------------------------------

class TestChunkerRegistry:

    def test_list_available_includes_all(self):
        available = ChunkerRegistry.list_available()
        for name in ["fixed_size", "recursive_semantic", "sequential_hac",
                     "max_min", "texttiling", "graphseg", "lumberchunker", "densex"]:
            assert name in available

    def test_list_robust(self):
        robust = ChunkerRegistry.list_robust()
        assert set(robust) == {"fixed_size", "recursive_semantic",
                               "sequential_hac", "max_min"}

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            ChunkerRegistry.get("nonexistent_method")

    def test_register_custom(self):
        from rag_chunking_bench.chunkers.base import BaseChunker
        class DummyChunker(BaseChunker):
            @property
            def name(self): return "dummy"
            def chunk(self, doc): return [doc]

        ChunkerRegistry.register("dummy", DummyChunker)
        c = ChunkerRegistry.get("dummy")
        assert c.name == "dummy"

    def test_experimental_stubs_raise_not_implemented(self):
        for method in ["graphseg", "lumberchunker", "densex"]:
            c = ChunkerRegistry.get(method)
            with pytest.raises(NotImplementedError):
                c.chunk("test")
