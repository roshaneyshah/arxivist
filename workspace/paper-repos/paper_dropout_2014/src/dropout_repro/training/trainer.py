"""
training/trainer.py
===================
Training loop for the Dropout reproduction.

Implements the training procedure from Section 5.1 and Appendix A/B:
    - SGD with momentum (Appendix A.2: momentum 0.95-0.99 for dropout nets)
    - Max-norm constraint applied after every gradient step (Section 5.1)
    - Training tracked in weight updates (not epochs), matching Figure 4 x-axis
    - Two-phase training protocol (Appendix B.1):
        Phase 1: train on 50K, tune hyperparams on 10K val
        Phase 2: retrain on full 60K for 1M weight updates

Key paper quote (Section 5.1):
    "Dropout neural networks can be trained using stochastic gradient descent
    in a manner similar to standard neural nets. The only difference is that
    for each training case in a mini-batch, we sample a thinned network by
    dropping out units."

Paper: Srivastava et al. (2014) JMLR 15:1929-1958, Section 5, Appendix A-B.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dropout_repro.training.losses import cross_entropy_loss, l1_regularization, kl_sparsity_loss
from dropout_repro.utils.config import DropoutConfig
from dropout_repro.utils.max_norm import apply_max_norm_constraint, check_max_norm_satisfied


class Trainer:
    """
    Training loop for DropoutNet.

    Supports:
        - SGD with momentum (paper's optimizer, Appendix A.2)
        - Max-norm constraint projected after every step (Section 5.1)
        - L2 / L1 / KL-sparsity regularization (for Table 9 ablation)
        - Configurable weight-update-based logging and checkpointing
        - Resume from checkpoint

    Args:
        model:   DropoutNet instance.
        config:  DropoutConfig with all hyperparameters.
        device:  torch.device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DropoutConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device

        tc = config.training
        mc = config.model

        # SGD optimizer with momentum (Section 5.1, Appendix A.2)
        # Appendix A.2: "values around 0.95 to 0.99 work quite a lot better" for dropout
        self.optimizer = SGD(
            model.parameters(),
            lr=tc.learning_rate,       # ASSUMED 0.01 for MNIST
            momentum=tc.momentum,      # 0.95 (Appendix B.1)
            weight_decay=tc.weight_decay,
            nesterov=False,            # paper does not mention Nesterov
        )

        self.n_weight_updates = tc.n_weight_updates
        self.use_max_norm = tc.use_max_norm
        self.max_norm_c = mc.max_norm_c
        self.log_interval = tc.log_interval
        self.checkpoint_interval = tc.checkpoint_interval

        # Output paths
        self.run_name = config.experiment.run_name
        self.output_dir = Path(config.experiment.output_dir) / self.run_name
        self.log_dir = Path(config.experiment.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.best_val_error = float("inf")
        self.history: Dict[str, List] = {
            "step": [], "train_loss": [], "train_error": [], "val_error": []
        }

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        n_updates: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Run the training loop for n_updates weight updates.

        Paper tracks training in weight updates (not epochs), matching Figure 4.
        Each forward+backward+step on one mini-batch = 1 weight update.

        Appendix B.1: "Hyperparameters were tuned on the validation set such
        that the best validation error was produced after 1 million weight updates."

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data (optional).
            n_updates:    Number of weight updates. Defaults to config value.

        Returns:
            History dict with keys: step, train_loss, train_error, val_error.
        """
        if n_updates is None:
            n_updates = self.n_weight_updates

        self._print_training_summary(train_loader, n_updates)

        self.model.train()
        start_time = time.time()

        # Infinite DataLoader cycling (weight-update based, not epoch based)
        data_iter = self._infinite_loader(train_loader)

        pbar = tqdm(
            total=n_updates,
            initial=self.global_step,
            desc=f"Training [{self.run_name}]",
            unit="updates",
        )

        while self.global_step < n_updates:
            x, y = next(data_iter)
            x, y = x.to(self.device), y.to(self.device)

            # --- Forward pass (thinned network sampled by dropout masks) ---
            self.optimizer.zero_grad()
            logits = self.model(x)  # [B, C] — Eq. 2-5, Section 4

            # --- Loss computation ---
            loss = cross_entropy_loss(logits, y)

            # Optional auxiliary regularizers for Table 9 ablation
            loss = self._add_auxiliary_losses(loss)

            # --- Backward pass through the thinned sub-network ---
            # Section 5.1: "the derivatives of the loss function are backpropagated
            # through the sub-network"
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            # --- Max-norm projection (Section 5.1) ---
            # Applied AFTER every gradient update
            # "the neural network was optimized under the constraint ||w||_2 <= c"
            if self.use_max_norm:
                apply_max_norm_constraint(self.model, self.max_norm_c)

            # --- Logging ---
            if self.global_step % self.log_interval == 0:
                train_error = self._compute_batch_error(logits, y)
                val_error = None
                if val_loader is not None:
                    val_result = self.evaluate(val_loader)
                    val_error = val_result["error_rate"]

                    # Save best checkpoint by validation error
                    if val_error < self.best_val_error:
                        self.best_val_error = val_error
                        self.save_checkpoint("best.pt", self.global_step)

                self._log_step(loss.item(), train_error, val_error)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    val_err=f"{val_error:.2f}%" if val_error is not None else "N/A",
                )

            # --- Periodic checkpoint ---
            if self.global_step % self.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.global_step:07d}.pt", self.global_step)

            pbar.update(1)

        pbar.close()
        elapsed = time.time() - start_time
        print(f"\nTraining complete: {n_updates:,} updates in {elapsed/60:.1f} min")
        print(f"Best val error: {self.best_val_error:.4f}%")

        # Save final checkpoint
        self.save_checkpoint("final.pt", self.global_step)
        self._save_history()

        return self.history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a DataLoader.

        Sets model to eval() mode (disables dropout masks per PyTorch convention),
        runs inference, then restores train() mode.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            Dict with keys: 'error_rate' (%), 'loss'.
        """
        was_training = self.model.training
        self.model.eval()

        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = cross_entropy_loss(logits, y)

                preds = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.shape[0]
                total_loss += loss.item()
                n_batches += 1

        error_rate = (1.0 - total_correct / total_samples) * 100.0
        avg_loss = total_loss / n_batches

        if was_training:
            self.model.train()

        return {"error_rate": error_rate, "loss": avg_loss}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str, step: int) -> None:
        """Save model weights, optimizer state, and training config."""
        path = self.output_dir / filename
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "best_val_error": self.best_val_error,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint and restore model/optimizer state.

        Args:
            path: Path to checkpoint file (.pt).

        Returns:
            Global step at which training was interrupted.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["step"]
        self.best_val_error = ckpt.get("best_val_error", float("inf"))
        self.history = ckpt.get("history", self.history)
        print(f"Resumed from step {self.global_step:,} (best val error: {self.best_val_error:.4f}%)")
        return self.global_step

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infinite_loader(loader: DataLoader):
        """Cycle through a DataLoader indefinitely."""
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def _compute_batch_error(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Classification error rate on a single batch (%)."""
        preds = logits.argmax(dim=1)
        return (1.0 - (preds == targets).float().mean().item()) * 100.0

    def _add_auxiliary_losses(self, base_loss: torch.Tensor) -> torch.Tensor:
        """
        Add optional auxiliary regularization terms (for Table 9 ablation).

        These are added to the base cross-entropy loss when configured.
        Only active when non-zero coefficients are set in training config.
        """
        tc = self.config.training
        loss = base_loss

        # L1 regularization (lasso, Tibshirani 1996) — Table 9 ablation
        if hasattr(tc, "l1_weight") and tc.l1_weight > 0:
            loss = loss + l1_regularization(self.model, tc.l1_weight)

        return loss

    def _log_step(
        self,
        train_loss: float,
        train_error: float,
        val_error: Optional[float],
    ) -> None:
        """Record metrics and append to JSONL log file."""
        self.history["step"].append(self.global_step)
        self.history["train_loss"].append(train_loss)
        self.history["train_error"].append(train_error)
        self.history["val_error"].append(val_error)

        log_entry = {
            "step": self.global_step,
            "train_loss": round(train_loss, 6),
            "train_error_pct": round(train_error, 4),
            "val_error_pct": round(val_error, 4) if val_error is not None else None,
        }

        log_path = self.log_dir / f"{self.run_name}.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _save_history(self) -> None:
        """Persist full training history to JSON."""
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _print_training_summary(self, loader: DataLoader, n_updates: int) -> None:
        """Print training summary before starting."""
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        dataset_size = len(loader.dataset)
        batch_size = loader.batch_size
        updates_per_epoch = dataset_size // batch_size
        n_epochs = n_updates / updates_per_epoch

        print("\n" + "=" * 60)
        print("DROPOUT REPRODUCTION — TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Model:            {type(self.model).__name__}")
        print(f"  Parameters:       {n_params:,}")
        print(f"  Dataset size:     {dataset_size:,}")
        print(f"  Batch size:       {batch_size}")
        print(f"  Weight updates:   {n_updates:,}")
        print(f"  Updates/epoch:    {updates_per_epoch:,}")
        print(f"  Effective epochs: {n_epochs:.1f}")
        print(f"  Optimizer:        SGD(lr={self.config.training.learning_rate}, "
              f"momentum={self.config.training.momentum})")
        print(f"  Max-norm c:       {self.max_norm_c if self.use_max_norm else 'disabled'}")
        print(f"  Device:           {self.device}")
        print(f"  Run name:         {self.run_name}")
        print(f"  Target error:     {self.config.experiment.expected_test_error_pct}%")
        print("=" * 60 + "\n")
