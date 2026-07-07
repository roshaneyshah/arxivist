"""Configuration loading and global reproducibility utilities.

Implements the config system defined in the ArXivist architecture plan for
arXiv:2401.06249 (SpotV2Net). No paper equations are implemented here.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import yaml


REQUIRED_TOP_LEVEL_KEYS = ("model", "training", "data", "evaluation", "hardware")


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML config file.

    Args:
        path: Path to a config YAML (e.g. ``configs/config.yaml``).

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required top-level section is missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in config]
    if missing:
        raise ValueError(
            f"Config at {path} is missing required section(s): {missing}. "
            f"Required sections: {REQUIRED_TOP_LEVEL_KEYS}"
        )

    _validate_model_section(config["model"])
    _validate_training_section(config["training"])
    return config


def _validate_model_section(model_cfg: Dict[str, Any]) -> None:
    if model_cfg.get("num_lags", 0) < 0:
        raise ValueError("model.num_lags must be >= 0")
    if model_cfg.get("output_dim", 1) not in (1, 14):
        # 1 = single-step (Sec. 7.2), 14 = multi-step functional forecast (Sec. 7.4)
        raise ValueError(
            "model.output_dim should be 1 (single-step) or 14 (multi-step, per Sec. 7.4 "
            "H=14 intraday 30-minute steps)."
        )


def _validate_training_section(training_cfg: Dict[str, Any]) -> None:
    if training_cfg.get("batch_size", 0) <= 0:
        raise ValueError("training.batch_size must be > 0")
    if training_cfg.get("epochs", 0) <= 0:
        raise ValueError("training.epochs must be > 0")


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    ASSUMED: the paper does not disclose the random seed(s) used; this utility
    implements a standard deterministic-seeding pattern (SIR implementation_assumptions).

    Args:
        seed: Random seed value.
        deterministic: If True, forces PyTorch deterministic algorithms
            (may reduce throughput; noted in the paper's absence of guidance).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not installed in this environment (e.g. pure-data-prep step)
        pass


def get_device(hardware_cfg: Dict[str, Any]):
    """Resolve the torch.device from the hardware config block."""
    import torch

    requested = hardware_cfg.get("device", "cuda_if_available")
    if requested == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
