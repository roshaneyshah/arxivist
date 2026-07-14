"""
utils/config.py
===============
Configuration dataclasses and loading utilities for the Dropout reproduction.

Loads YAML configs into typed Python dataclasses. All hyperparameter values
with SIR confidence < 0.7 are annotated with # ASSUMED comments in the YAML
and flagged at load time.

Paper: Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks
       from Overfitting", JMLR 15:1929-1958.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for full reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables torch.use_deterministic_algorithms(True).
                       This may significantly slow down training on GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        warnings.warn(
            "Deterministic mode enabled — training may be significantly slower.",
            UserWarning,
        )
    else:
        # cuDNN benchmark speeds up conv workloads; disable only for full determinism
        torch.backends.cudnn.benchmark = True


def get_device(device_str: str) -> torch.device:
    """
    Resolve device string to a torch.device.

    Args:
        device_str: "auto", "cuda", "cpu", or "cuda:N".

    Returns:
        Resolved torch.device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Model architecture hyperparameters.

    All values sourced from the paper's Appendix B.1 unless marked ASSUMED.
    SIR confidence scores are noted in comments.

    Key convention note:
        p_hidden and p_input are RETENTION probabilities (paper convention).
        They are converted to DROP probabilities when passed to nn.Dropout:
            nn.Dropout(p=1 - p_paper)
        For p_hidden=0.5: nn.Dropout(0.5)  (same number, different meaning)
        For p_input=0.8:  nn.Dropout(0.2)  (DIFFERENT — 0.2 drop, not 0.2 retain)
    """
    input_dim: int = 784                         # MNIST: 28×28 pixels (confidence: 0.97)
    hidden_dims: List[int] = field(
        default_factory=lambda: [1024, 1024, 1024]  # Appendix B.1 (confidence: 0.95)
    )
    num_classes: int = 10                        # MNIST digit classes (confidence: 0.97)
    activation: str = "relu"                     # Section 6.1.1, Table 2 (confidence: 0.95)
    use_dropout: bool = True
    p_hidden: float = 0.5                        # Appendix B.1 (confidence: 0.95)
    p_input: float = 0.8                         # Appendix B.1 (confidence: 0.95)
    max_norm_c: float = 2.0                      # Appendix B.1: c=2 (confidence: 0.95)

    def __post_init__(self) -> None:
        assert self.activation in ("relu", "logistic"), \
            f"activation must be 'relu' or 'logistic', got '{self.activation}'"
        assert 0.0 < self.p_hidden <= 1.0, \
            f"p_hidden must be in (0, 1], got {self.p_hidden}"
        assert 0.0 < self.p_input <= 1.0, \
            f"p_input must be in (0, 1], got {self.p_input}"
        assert self.max_norm_c > 0, \
            f"max_norm_c must be positive, got {self.max_norm_c}"
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer")


@dataclass
class TrainingConfig:
    """
    Training loop hyperparameters.

    Values marked ASSUMED were not explicitly stated in the paper for MNIST.
    See SIR implementation_assumptions for full reasoning.
    """
    optimizer: str = "sgd"
    learning_rate: float = 0.01          # ASSUMED: not stated for MNIST; see Appendix A.2 for heuristic
    momentum: float = 0.95               # Appendix B.1 (confidence: 0.95)
    weight_decay: float = 0.0            # Dropout + max-norm is the primary regularizer
    n_weight_updates: int = 1_000_000    # Appendix B.1 (confidence: 0.95)
    batch_size: int = 128                # ASSUMED: not stated in paper (confidence: 0.65)
    use_max_norm: bool = True            # Appendix B.1 (confidence: 0.95)
    checkpoint_interval: int = 10_000
    log_interval: int = 1_000
    seed: int = 42

    def __post_init__(self) -> None:
        assert self.optimizer in ("sgd",), \
            f"Only 'sgd' optimizer is implemented (paper uses SGD). Got: {self.optimizer}"
        assert self.learning_rate > 0, f"learning_rate must be positive"
        assert 0.0 <= self.momentum < 1.0, f"momentum must be in [0, 1)"
        assert self.n_weight_updates > 0
        assert self.batch_size > 0


@dataclass
class DataConfig:
    """Dataset and preprocessing settings."""
    dataset: str = "mnist"
    data_dir: str = "./data"
    val_size: int = 10_000        # Appendix B.1: "held out 10,000 random training images"
    num_workers: int = -1         # -1 = auto: 0 on Windows (spawn), 4 on Linux/Mac (fork)
    normalize_mean: float = 0.1307
    normalize_std: float = 0.3081

    def __post_init__(self) -> None:
        assert self.dataset in ("mnist",), \
            f"Only 'mnist' is implemented in this repro. Got: {self.dataset}"
        assert self.val_size > 0


@dataclass
class HardwareConfig:
    """Hardware and precision settings."""
    device: str = "auto"          # "auto", "cuda", "cpu", or "cuda:N"
    deterministic: bool = False   # Enables torch.use_deterministic_algorithms


@dataclass
class ExperimentConfig:
    """Experiment tracking and output settings."""
    run_name: str = "dropout_repro"
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    target_description: str = ""
    expected_test_error_pct: Optional[float] = None


@dataclass
class DropoutConfig:
    """
    Top-level configuration container for the Dropout reproduction.

    Load from YAML with:
        config = DropoutConfig.from_yaml("configs/mnist_3layer_1024.yaml")
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DropoutConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            Populated DropoutConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If required fields are missing or invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        model_cfg = ModelConfig(**raw.get("model", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        data_cfg = DataConfig(**raw.get("data", {}))
        hardware_cfg = HardwareConfig(**raw.get("hardware", {}))
        experiment_cfg = ExperimentConfig(**raw.get("experiment", {}))

        return cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            hardware=hardware_cfg,
            experiment=experiment_cfg,
        )

    def to_dict(self) -> dict:
        """Serialize config to a plain dict (for logging/checkpointing)."""
        import dataclasses
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        lines = ["DropoutConfig("]
        lines.append(f"  model={self.model},")
        lines.append(f"  training={self.training},")
        lines.append(f"  data={self.data},")
        lines.append(f"  hardware={self.hardware},")
        lines.append(f"  experiment={self.experiment}")
        lines.append(")")
        return "\n".join(lines)
