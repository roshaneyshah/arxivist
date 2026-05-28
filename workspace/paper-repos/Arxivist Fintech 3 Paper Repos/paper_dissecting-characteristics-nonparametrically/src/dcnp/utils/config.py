"""
utils/config.py
===============
Configuration loading utilities for the DCNP replication.

Loads config.yaml and exposes a typed Config dataclass. Also provides
reproducibility seed utilities.

Paper: Freyberger, Neuhierl & Weber (2017) — NBER WP 23227
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    n_knots: int = 14
    n_characteristics: int = 36


@dataclass
class LassoConfig:
    lambda1_grid: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    lambda2_grid: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    bic_criterion: str = "yuan_lin_2006"


@dataclass
class EstimationConfig:
    rolling_window_months: int = 120
    hedge_decile: float = 0.10
    weighting: str = "equal"
    n_confidence_sims: int = 10000  # ASSUMED: not specified in paper
    alpha_bands: float = 0.05
    annualization_factor: int = 12


@dataclass
class DataConfig:
    crsp_path: str = "data/crsp_monthly.parquet"
    compustat_path: str = "data/compustat_annual.parquet"
    ff3_factors_path: str = "data/ff3_factors.csv"
    sample_start: str = "1963-07"
    sample_end: str = "2015-06"
    oos_selection_end: str = "1990-12"
    oos_start: str = "1991-01"
    min_price: float = 5.0
    exchanges: List[str] = field(default_factory=lambda: ["NYSE", "AMEX", "NASDAQ"])
    min_compustat_years: int = 2
    share_codes: List[int] = field(default_factory=lambda: [10, 11])


@dataclass
class EvaluationConfig:
    compute_ff3_alpha: bool = True
    compute_firm_level_r2: bool = True
    nyse_size_percentiles: List[int] = field(default_factory=lambda: [10, 20, 50])


@dataclass
class ReproducibilityConfig:
    seed: int = 42  # ASSUMED: not specified in paper
    n_jobs: int = -1


@dataclass
class DCNPConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lasso: LassoConfig = field(default_factory=LassoConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    def __repr__(self) -> str:
        return (
            f"DCNPConfig(n_knots={self.model.n_knots}, "
            f"n_chars={self.model.n_characteristics}, "
            f"window={self.estimation.rolling_window_months}mo, "
            f"sample={self.data.sample_start}:{self.data.sample_end})"
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> DCNPConfig:
    """Load YAML config from disk and validate critical values.

    Args:
        config_path: Path to config.yaml

    Returns:
        Populated DCNPConfig dataclass

    Raises:
        ValueError: If critical config values are out of valid range
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    cfg = DCNPConfig()

    if "model" in raw:
        m = raw["model"]
        cfg.model.n_knots = m.get("n_knots", cfg.model.n_knots)
        cfg.model.n_characteristics = m.get("n_characteristics", cfg.model.n_characteristics)

    if "lasso" in raw:
        l = raw["lasso"]
        cfg.lasso.lambda1_grid = l.get("lambda1_grid", cfg.lasso.lambda1_grid)
        cfg.lasso.lambda2_grid = l.get("lambda2_grid", cfg.lasso.lambda2_grid)
        cfg.lasso.bic_criterion = l.get("bic_criterion", cfg.lasso.bic_criterion)

    if "estimation" in raw:
        e = raw["estimation"]
        cfg.estimation.rolling_window_months = e.get("rolling_window_months", cfg.estimation.rolling_window_months)
        cfg.estimation.hedge_decile = e.get("hedge_decile", cfg.estimation.hedge_decile)
        cfg.estimation.weighting = e.get("weighting", cfg.estimation.weighting)
        cfg.estimation.n_confidence_sims = e.get("n_confidence_sims", cfg.estimation.n_confidence_sims)
        cfg.estimation.alpha_bands = e.get("alpha_bands", cfg.estimation.alpha_bands)
        cfg.estimation.annualization_factor = e.get("annualization_factor", cfg.estimation.annualization_factor)

    if "data" in raw:
        d = raw["data"]
        for key in vars(cfg.data):
            if key in d:
                setattr(cfg.data, key, d[key])

    if "evaluation" in raw:
        ev = raw["evaluation"]
        for key in vars(cfg.evaluation):
            if key in ev:
                setattr(cfg.evaluation, key, ev[key])

    if "reproducibility" in raw:
        r = raw["reproducibility"]
        cfg.reproducibility.seed = r.get("seed", cfg.reproducibility.seed)
        cfg.reproducibility.n_jobs = r.get("n_jobs", cfg.reproducibility.n_jobs)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: DCNPConfig) -> None:
    """Validate config values at load time."""
    if cfg.model.n_knots not in [4, 9, 14, 19]:
        raise ValueError(
            f"n_knots={cfg.model.n_knots} is non-standard. "
            f"Paper uses 4, 9, 14, or 19. Override only if intentional."
        )
    if not (0.01 <= cfg.estimation.hedge_decile <= 0.5):
        raise ValueError(f"hedge_decile must be in [0.01, 0.5], got {cfg.estimation.hedge_decile}")
    if cfg.estimation.weighting not in ("equal", "value"):
        raise ValueError(f"weighting must be 'equal' or 'value', got {cfg.estimation.weighting}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Seeds Python random, NumPy. Note: this paper uses no neural networks
    so torch seeding is not required.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
