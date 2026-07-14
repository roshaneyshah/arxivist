"""
utils/config.py
===============
Configuration loading and reproducibility utilities for FS-GCLSTM.

Paper: Liu (2023/2025) — arXiv:2303.09406
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 64          # ASSUMED: not stated in paper
    n_gcn_layers: int = 2         # Stated: Section III.a
    n_lstm_layers: int = 3        # Stated: Section III.c
    input_seq_len: int = 60       # Stated: Section IV.A
    mlp_hidden: int = 128         # ASSUMED: not stated in paper
    dropout: float = 0.0          # ASSUMED


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-5
    lr_schedule: str = "onecycle"
    max_epochs: int = 30
    early_stop_patience: int = 10
    loss: str = "mse"             # ASSUMED
    initial_window_days: int = 3000
    train_frac: float = 0.70
    val_frac: float = 0.20
    test_frac: float = 0.10
    advance_days: int = 300
    log_every_n_steps: int = 10
    checkpoint_every_n_steps: int = 50


@dataclass
class DataConfig:
    price_start: str = "2000-01-01"
    price_end: str = "2024-12-31"
    min_trading_days: int = 5000
    lseg_confidence_threshold: float = 0.20
    bidirectional_edges: bool = True
    edge_weight: str = "confidence_score"
    transaction_cost_bps: float = 1.0
    price_data_path: str = "data/prices.parquet"
    graph_data_path: str = "data/value_chain.parquet"


@dataclass
class EvaluationConfig:
    risk_free_eurostoxx: str = "EONIA"
    risk_free_sp500: str = "USD_LIBOR_ON"
    portfolio_type: str = "equal_weight_long_only"
    rebalance: str = "daily"
    sequence_lengths_robustness: List[int] = field(default_factory=lambda: [30, 60, 90, 120])


@dataclass
class HardwareConfig:
    device: str = "cuda"
    precision: str = "float32"
    seed: int = 42
    num_workers: int = 4
    deterministic: bool = False


@dataclass
class FSGCLSTMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def load_config(path: str | Path) -> FSGCLSTMConfig:
    """Load YAML config and return typed FSGCLSTMConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = FSGCLSTMConfig()
    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
    if "training" in raw:
        for k, v in raw["training"].items():
            if hasattr(cfg.training, k):
                setattr(cfg.training, k, v)
    if "data" in raw:
        for k, v in raw["data"].items():
            if hasattr(cfg.data, k):
                setattr(cfg.data, k, v)
    if "evaluation" in raw:
        for k, v in raw["evaluation"].items():
            if hasattr(cfg.evaluation, k):
                setattr(cfg.evaluation, k, v)
    if "hardware" in raw:
        for k, v in raw["hardware"].items():
            if hasattr(cfg.hardware, k):
                setattr(cfg.hardware, k, v)
    return cfg


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
