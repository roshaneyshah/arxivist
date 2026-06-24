"""
tests/test_utils.py
===================
Unit tests for utility modules: Timer, config loading, text utils.

Paper: arXiv:2606.00881
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from rag_chunking_bench.utils.timing import Timer
from rag_chunking_bench.utils.text_utils import split_sentences, count_tokens, truncate_to_tokens
from rag_chunking_bench.utils.config import BenchConfig, set_seed


class TestTimer:

    def test_elapsed_increases(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_seconds() >= 0.04

    def test_human_readable_seconds(self):
        with Timer() as t:
            time.sleep(0.01)
        h = t.elapsed_human()
        assert isinstance(h, str)

    def test_not_started_raises(self):
        t = Timer()
        with pytest.raises(RuntimeError):
            t.elapsed_seconds()

    def test_less_than_one_second(self):
        with Timer() as t:
            pass
        assert t.elapsed_human() == "<1s"


class TestTextUtils:

    def test_split_sentences_basic(self):
        text = "Hello world. This is a test. Another sentence here."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_split_sentences_empty(self):
        assert split_sentences("") == []

    def test_count_tokens_basic(self):
        text = "Hello world"
        n = count_tokens(text)
        assert n > 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_truncate_preserves_short_text(self):
        text = "Short text"
        result = truncate_to_tokens(text, max_tokens=1000)
        assert result == text

    def test_truncate_cuts_long_text(self):
        text = "word " * 2000
        result = truncate_to_tokens(text, max_tokens=100)
        assert count_tokens(result) <= 100


class TestBenchConfig:

    def test_default_config(self):
        cfg = BenchConfig()
        assert cfg.embedding.model == "BAAI/bge-m3"
        assert cfg.retrieval.top_k_accuracy == 5
        assert cfg.retrieval.top_k_recall == 10
        assert cfg.generation.max_context_tokens == 4000
        assert cfg.pipeline.timeout_hours == 48.0
        assert cfg.chunkers.fixed_size.chunk_size == 512
        assert cfg.chunkers.fixed_size.overlap == 50

    def test_from_yaml(self, tmp_path):
        yaml_content = """
chunkers:
  enabled: [fixed_size]
embedding:
  model: BAAI/bge-m3
  batch_size: 32
pipeline:
  timeout_hours: 1.0
"""
        cfg_path = tmp_path / "test_config.yaml"
        cfg_path.write_text(yaml_content)
        cfg = BenchConfig.from_yaml(str(cfg_path))
        assert cfg.embedding.batch_size == 32
        assert cfg.pipeline.timeout_hours == 1.0
        assert "fixed_size" in cfg.chunkers.enabled

    def test_set_seed_runs_without_error(self):
        set_seed(42)  # should not raise
