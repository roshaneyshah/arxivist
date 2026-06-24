"""
utils/config.py
===============
Configuration dataclasses and YAML loader for the RAG Chunking Benchmark.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3: Methodology — hyperparameters and experimental setup.

All parameters with low SIR confidence are marked ASSUMED.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FixedSizeConfig:
    chunk_size: int = 512    # SIR conf 0.99 — explicitly stated Section 3
    overlap: int = 50        # SIR conf 0.99 — explicitly stated Section 3


@dataclass
class RecursiveSemanticConfig:
    chunk_size: int = 512
    overlap: int = 50
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )  # TODO: verify exact list from paper's LangChain version


@dataclass
class SequentialHACConfig:
    # ASSUMED: default from Qu et al. 2025 (NAACL) — SIR conf 0.88
    similarity_threshold: float = 0.85
    max_chunk_tokens: int = 512


@dataclass
class MaxMinConfig:
    # ASSUMED: alpha from Kiss et al. 2025 (Discover Computing) — SIR conf 0.83
    alpha: float = 0.5


@dataclass
class TextTilingConfig:
    w: int = 20                        # pseudosentence size in words
    k: int = 10                        # block comparison size
    smoothing_width: int = 2
    boundary_alignment: str = "sentence"  # explicitly stated in paper Section 3


@dataclass
class ChunkersConfig:
    enabled: List[str] = field(
        default_factory=lambda: [
            "fixed_size", "recursive_semantic", "sequential_hac", "max_min"
        ]
    )
    optional: List[str] = field(
        default_factory=lambda: ["texttiling", "graphseg", "lumberchunker", "densex"]
    )
    fixed_size: FixedSizeConfig = field(default_factory=FixedSizeConfig)
    recursive_semantic: RecursiveSemanticConfig = field(default_factory=RecursiveSemanticConfig)
    sequential_hac: SequentialHACConfig = field(default_factory=SequentialHACConfig)
    max_min: MaxMinConfig = field(default_factory=MaxMinConfig)
    texttiling: TextTilingConfig = field(default_factory=TextTilingConfig)


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-m3"   # SIR conf 0.99 — explicitly stated
    batch_size: int = 64
    device: str = "auto"
    normalize: bool = True        # required for cosine similarity (EQ3)
    embedding_dim: int = 1024     # SIR conf 0.90 — from bge-m3 spec, not stated in paper


@dataclass
class RerankerConfig:
    model: str = "BAAI/bge-reranker-v2-m3"  # SIR conf 0.99 — explicitly stated
    device: str = "auto"


@dataclass
class RetrievalConfig:
    top_k_index: int = 10      # candidates retrieved before reranking
    top_k_accuracy: int = 5    # Accuracy@5 — EQ5; SIR conf 0.99
    top_k_recall: int = 10     # Recall@10 — EQ6; SIR conf 0.99
    top_k_generation: int = 5  # chunks to generator in EXP2; SIR conf 0.99
    index_type: str = "cosine"


@dataclass
class GenerationConfig:
    model: str = "gpt-oss-20b"        # SIR conf 0.99 — explicitly stated
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_context_tokens: int = 4000     # SIR conf 0.99 — explicitly stated
    temperature: float = 1.0           # ASSUMED: SIR conf 0.55 — not stated in paper


@dataclass
class JudgeConfig:
    model: str = "gpt-oss-20b"        # SIR conf 0.99
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    scale_min: int = 1                 # SIR conf 0.99 — explicitly stated
    scale_max: int = 5                 # SIR conf 0.99 — explicitly stated


@dataclass
class DatasetsConfig:
    data_dir: str = "data/"
    enabled: List[str] = field(default_factory=lambda: [
        "squad", "triviaqa", "triviaqa_merged",
        "poquad", "poquad_merged", "nq", "qasper",
        "gutenqa", "gutenqa_merged", "literaryqa", "novelqa",
    ])


@dataclass
class PipelineConfig:
    timeout_hours: float = 48.0  # SIR conf 0.99 — explicitly stated Section 3
    output_dir: str = "results/"
    resume: bool = True
    log_failures: bool = True
    seed: int = 42


@dataclass
class BenchConfig:
    """
    Top-level configuration for the RAG Chunking Benchmark.

    Paper: arXiv:2606.00881, Section 3 — Methodology.
    Load from YAML via BenchConfig.from_yaml(path).
    """
    chunkers: ChunkersConfig = field(default_factory=ChunkersConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "BenchConfig":
        """Load config from a YAML file, merging with defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        _deep_merge(cfg, raw or {})
        cfg = _resolve_env_vars(cfg)
        _validate(cfg)
        return cfg

    def to_yaml(self, path: str) -> None:
        """Serialize config back to YAML."""
        import dataclasses
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)

    def __repr__(self) -> str:
        return (
            f"BenchConfig(chunkers={self.chunkers.enabled}, "
            f"embed={self.embedding.model}, "
            f"datasets={self.datasets.enabled[:3]}...)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(obj: object, d: dict) -> None:
    """Recursively merge dict d into dataclass obj."""
    import dataclasses
    for k, v in d.items():
        if not hasattr(obj, k):
            continue
        attr = getattr(obj, k)
        if dataclasses.is_dataclass(attr) and isinstance(v, dict):
            _deep_merge(attr, v)
        else:
            setattr(obj, k, v)


def _resolve_env_vars(cfg: BenchConfig) -> BenchConfig:
    """Override API keys/base from environment variables if not set in config."""
    if cfg.generation.api_base is None:
        cfg.generation.api_base = os.environ.get("OPENAI_API_BASE", "")
    if cfg.generation.api_key is None:
        cfg.generation.api_key = os.environ.get("OPENAI_API_KEY", "")
    if cfg.judge.api_base is None:
        cfg.judge.api_base = os.environ.get("OPENAI_API_BASE", "")
    if cfg.judge.api_key is None:
        cfg.judge.api_key = os.environ.get("OPENAI_API_KEY", "")
    return cfg


def _validate(cfg: BenchConfig) -> None:
    """Raise ValueError for obviously invalid config values."""
    if cfg.embedding.embedding_dim <= 0:
        raise ValueError(f"embedding.embedding_dim must be > 0, got {cfg.embedding.embedding_dim}")
    if cfg.retrieval.top_k_index < cfg.retrieval.top_k_generation:
        raise ValueError(
            f"retrieval.top_k_index ({cfg.retrieval.top_k_index}) must be >= "
            f"top_k_generation ({cfg.retrieval.top_k_generation})"
        )
    if cfg.judge.scale_min >= cfg.judge.scale_max:
        raise ValueError("judge.scale_min must be < scale_max")
    if cfg.pipeline.timeout_hours <= 0:
        raise ValueError("pipeline.timeout_hours must be > 0")
    if not cfg.chunkers.enabled:
        raise ValueError("chunkers.enabled must have at least one method")


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and (if available) PyTorch.
    Called at the top of every entrypoint for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not required for chunking-only runs
