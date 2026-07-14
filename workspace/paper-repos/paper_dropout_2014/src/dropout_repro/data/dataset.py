"""
data/dataset.py
===============
MNIST data loading and splitting.

Implements the data protocol from Appendix B.1:
    - 60,000 training images total
    - 10,000 held out for validation (hyperparameter tuning)
    - 10,000 test images
    - Two-phase training:
        Phase 1: Train on 50K, tune hyperparams on 10K val
        Phase 2: Combine 60K (train+val), train for final evaluation

Paper: Srivastava et al. (2014) JMLR 15:1929-1958, Appendix B.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import MNIST

from dropout_repro.data.transforms import get_mnist_transforms


class MNISTDataModule:
    """
    MNIST data module handling download, splitting, and DataLoader creation.

    Implements the two-phase training protocol from Appendix B.1:
        Phase 1 (hyperparameter tuning):
            train_dataloader()  → 50,000 samples
            val_dataloader()    → 10,000 samples
        Phase 2 (final training):
            combined_dataloader() → 60,000 samples (train + val merged)
        Always:
            test_dataloader() → 10,000 test samples

    Args:
        data_dir:   Root directory for dataset storage.
        batch_size: Mini-batch size. (ASSUMED: 128 — not stated in paper.)
        val_size:   Number of training samples held out for validation.
                    Appendix B.1: "We held out 10,000 random training images."
        num_workers: DataLoader worker processes.
        seed:       Random seed for reproducible train/val split.
        mean:       Normalization mean (MNIST standard: 0.1307).
        std:        Normalization std  (MNIST standard: 0.3081).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,      # ASSUMED: not stated in paper
        val_size: int = 10_000,     # Appendix B.1: 10,000 validation images
        num_workers: int = -1,      # -1 = auto: 0 on Windows (spawn), 4 on Linux/Mac (fork)
        seed: int = 42,
        mean: float = 0.1307,
        std: float = 0.3081,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_size = val_size
        self.seed = seed

        # Auto-detect num_workers: Windows (spawn) requires 0 to avoid pickle errors.
        # On Linux/Mac (fork), 4 workers speeds up data loading significantly.
        import platform
        if num_workers == -1:
            self.num_workers = 0 if platform.system() == "Windows" else 4
        else:
            self.num_workers = num_workers

        self.train_transform, self.test_transform = get_mnist_transforms(mean, std)

        self._train_dataset: Optional[Subset] = None
        self._val_dataset: Optional[Subset] = None
        self._test_dataset: Optional[MNIST] = None
        self._full_train_dataset: Optional[MNIST] = None

    def setup(self) -> None:
        """
        Download MNIST and create train/val/test splits.

        Appendix B.1: "We held out 10,000 random training images for validation.
        Hyperparameters were tuned on the validation set..."
        """
        # Full training set (60K) for transforms
        full_train = MNIST(
            root=self.data_dir,
            train=True,
            transform=self.train_transform,
            download=True,
        )
        self._full_train_dataset = full_train

        # Split into train (50K) and val (10K)
        train_size = len(full_train) - self.val_size
        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset = random_split(
            full_train, [train_size, self.val_size], generator=generator
        )
        self._train_dataset = train_subset
        self._val_dataset = val_subset

        # Test set (10K, fixed by MNIST)
        self._test_dataset = MNIST(
            root=self.data_dir,
            train=False,
            transform=self.test_transform,
            download=True,
        )

        print(
            f"MNISTDataModule ready: "
            f"train={len(self._train_dataset):,}, "
            f"val={len(self._val_dataset):,}, "
            f"test={len(self._test_dataset):,}"
        )

    def _check_setup(self) -> None:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing data loaders.")

    def train_dataloader(self) -> DataLoader:
        """DataLoader for training split (50K samples)."""
        self._check_setup()
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """DataLoader for validation split (10K samples)."""
        self._check_setup()
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader for test split (10K samples)."""
        self._check_setup()
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def combined_dataloader(self) -> DataLoader:
        """
        DataLoader for Phase 2 training: full 60K (train + val merged).

        Appendix B.1: "The validation set was then combined with the training set
        and training was done for 1 million weight updates."
        """
        self._check_setup()
        return DataLoader(
            self._full_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def get_subset(self, size: int, split: str = "train") -> DataLoader:
        """
        Return a DataLoader for a random subset of the training data.

        Used for the dataset-size ablation (Section 7.4, Figure 10).

        Args:
            size:  Number of samples to use.
            split: "train" uses the 50K split; "combined" uses the full 60K.

        Returns:
            DataLoader with the requested subset size.
        """
        self._check_setup()
        if split == "combined":
            dataset = self._full_train_dataset
        else:
            dataset = self._train_dataset

        if size >= len(dataset):
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:size]
        subset = Subset(dataset, indices.tolist())
        return DataLoader(
            subset,
            batch_size=min(self.batch_size, size),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def __repr__(self) -> str:
        return (
            f"MNISTDataModule("
            f"batch_size={self.batch_size}, "
            f"val_size={self.val_size}, "
            f"data_dir={self.data_dir})"
        )
