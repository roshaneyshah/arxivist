"""
utils/config.py
===============
Configuration loading, validation, and reproducibility utilities.

ArXivist paper: arxiv_1706_03762 — "Attention Is All You Need"
"""

from __future__ import annotations

import os
import random
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    N: int = 6
    d_model: int = 512
    d_ff: int = 2048
    h: int = 8
    d_k: int = 64
    d_v: int = 64
    dropout: float = 0.1
    max_seq_len: int = 512
    weight_tying: bool = True  # ASSUMED: 3-way tie; Section 3.4, confidence 0.82

    def __post_init__(self):
        if self.d_model % self.h != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by h ({self.h}). "
                f"Paper uses d_k = d_v = d_model / h = {self.d_model // self.h}."
            )
        if self.d_k != self.d_model // self.h:
            raise ValueError(
                f"Expected d_k = d_model / h = {self.d_model // self.h}, got {self.d_k}."
            )


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    beta1: float = 0.9          # Section 5.3
    beta2: float = 0.98         # Section 5.3
    epsilon: float = 1e-9       # Section 5.3
    weight_decay: float = 0.0   # ASSUMED: not stated, Adam default
    warmup_steps: int = 4000    # Eq. 3, Section 5.3
    max_steps: int = 100000
    label_smoothing: float = 0.1  # Section 5.4
    max_tokens_per_batch: int = 25000  # Section 5.1
    gradient_clipping: Optional[float] = None  # ASSUMED: none
    checkpoint_every_steps: int = 1000
    log_every_steps: int = 100
    avg_last_n_checkpoints: int = 5  # Section 6.1 (base)
    seed: int = 42


@dataclass
class DataConfig:
    src_lang: str = "en"
    tgt_lang: str = "de"
    tokenizer: str = "sentencepiece_bpe"
    vocab_size: int = 37000  # Section 5.1
    shared_vocab: bool = True
    max_seq_len: int = 512
    data_dir: str = "data/wmt14_en_de"
    sp_model_path: str = "data/wmt14_en_de/spm.model"
    train_prefix: str = "train"
    val_prefix: str = "newstest2013"
    test_prefix: str = "newstest2014"
    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_idx: int = 0


@dataclass
class EvalConfig:
    beam_size: int = 4           # Section 6.1
    length_penalty_alpha: float = 0.6  # Section 6.1
    max_decode_len_offset: int = 50    # Section 6.1
    eval_every_steps: int = 5000
    bleu_tool: str = "sacrebleu"


@dataclass
class HardwareConfig:
    device: str = "cuda"
    num_gpus: int = 1
    mixed_precision: Optional[str] = None  # ASSUMED: none
    dataloader_num_workers: int = 4
    deterministic: bool = False


@dataclass
class TransformerConfig:
    """
    Master config for the Transformer model and training pipeline.

    Paper: "Attention Is All You Need", Vaswani et al. (2017)
    Load from YAML via TransformerConfig.from_yaml(path).
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "TransformerConfig":
        """Load config from a YAML file. Missing keys fall back to dataclass defaults."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        raw = raw or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            data=DataConfig(**raw.get("data", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
            hardware=HardwareConfig(**raw.get("hardware", {})),
        )

    def to_yaml(self, path: str) -> None:
        """Serialize config back to YAML."""
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)

    def __repr__(self) -> str:
        return (
            f"TransformerConfig(N={self.model.N}, d_model={self.model.d_model}, "
            f"h={self.model.h}, d_ff={self.model.d_ff}, "
            f"steps={self.training.max_steps})"
        )


# ---------------------------------------------------------------------------
# Reproducibility utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed:          Integer seed value.
        deterministic: If True, enable torch deterministic mode (may slow training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.11
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass


def get_device(config: HardwareConfig) -> torch.device:
    """Resolve the compute device from config."""
    if config.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(config.device)
