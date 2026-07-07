"""Training loop for SpotV2Net and baselines.

Not paper-specified beyond the hyperparameter tables (Appendix B); implements
standard practice for epoch iteration, checkpointing, and logging (SIR
implementation_assumptions).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from spotv2net.training.losses import mse_loss, QLIKELoss


class Trainer:
    """Epoch-based trainer with best-checkpoint saving on validation MSE.

    Args:
        model: A ``SpotV2Net`` (or compatible) ``nn.Module``.
        train_loader: DataLoader yielding dicts with keys x/edge_index/edge_attr/y.
        val_loader: DataLoader with the same schema, used for checkpoint selection.
        config: Full parsed config dict (uses the ``training`` and ``hardware`` blocks).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        train_cfg = config["training"]
        hw_cfg = config["hardware"]

        device_str = hw_cfg.get("device", "cuda_if_available")
        self.device = torch.device(
            "cuda" if device_str == "cuda_if_available" and torch.cuda.is_available() else
            (device_str if device_str != "cuda_if_available" else "cpu")
        )
        self.model.to(self.device)

        self.optimizer = self._build_optimizer(train_cfg)
        self.qlike = QLIKELoss()
        self.log_every = train_cfg.get("log_every_n_steps", 10)
        self.checkpoint_every = train_cfg.get("checkpoint_every_n_epochs", 5)
        self.grad_clip_norm = train_cfg.get("gradient_clip_norm")
        self.best_val_mse = float("inf")

    def _build_optimizer(self, train_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        name = train_cfg.get("optimizer", "adamw").lower()
        lr = train_cfg.get("learning_rate", 1e-4)
        # ASSUMED defaults (SIR implementation_assumptions): betas/weight_decay not
        # disclosed in the paper; read from config where the user can override.
        betas = (train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.999))
        weight_decay = train_cfg.get("weight_decay", 0.01)

        if name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        if name == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer '{name}' (Table 8 options: RMSProp, Adam, AdamW)")

    def fit(self, epochs: int, checkpoint_dir: str = "checkpoints") -> Dict[str, list]:
        """Run the training loop for ``epochs`` epochs.

        Args:
            epochs: Number of epochs (Table 8: 120 for the tuned single/multi-step model).
            checkpoint_dir: Directory to write periodic and best checkpoints to.

        Returns:
            Training history dict with 'train_loss', 'val_mse', 'val_qlike' lists.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        history: Dict[str, list] = {"train_loss": [], "val_mse": [], "val_qlike": []}

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Trainer] model params: {n_params:,} | train batches: {len(self.train_loader)} "
              f"| val batches: {len(self.val_loader)} | device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_mse, val_qlike = self._validate()

            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_mse)
            history["val_qlike"].append(val_qlike)

            print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.6e} "
                  f"val_mse={val_mse:.6e} val_qlike={val_qlike:.4f}")

            is_best = val_mse < self.best_val_mse
            if is_best:
                self.best_val_mse = val_mse
                self.save_checkpoint(os.path.join(checkpoint_dir, "best.pt"), is_best=True)
            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))

        return history

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            x = batch["x"].to(self.device).squeeze(0)
            edge_index = batch["edge_index"].to(self.device).squeeze(0)
            edge_attr = batch["edge_attr"].to(self.device).squeeze(0)
            y = batch["y"].to(self.device).squeeze(0)

            self.optimizer.zero_grad()
            pred = self.model(x, edge_index, edge_attr)
            loss = mse_loss(pred, y)  # Table 8: Loss Function = MSE
            loss.backward()
            if self.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            if step % self.log_every == 0:
                print(f"  [epoch {epoch}] step {step}/{len(self.train_loader)} loss={loss.item():.6e}")

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_mse, total_qlike = 0.0, 0.0
        for batch in self.val_loader:
            x = batch["x"].to(self.device).squeeze(0)
            edge_index = batch["edge_index"].to(self.device).squeeze(0)
            edge_attr = batch["edge_attr"].to(self.device).squeeze(0)
            y = batch["y"].to(self.device).squeeze(0)

            pred = self.model(x, edge_index, edge_attr)
            total_mse += mse_loss(pred, y).item()
            total_qlike += self.qlike(pred.clamp_min(1e-12), y.clamp_min(1e-12)).item()

        n = max(len(self.val_loader), 1)
        return total_mse / n, total_qlike / n

    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Save model + optimizer state.

        Args:
            path: Destination file path.
            is_best: Informational flag stored alongside the checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_mse": self.best_val_mse,
                "is_best": is_best,
                "config": self.config,
            },
            path,
        )
