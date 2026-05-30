"""
training/trainer.py
===================
Rolling-window trainer for FS-GCLSTM.

Paper: Liu (2023/2025) — arXiv:2303.09406, Section IV.C

Training setup:
  - Adam optimizer, lr=0.001, weight_decay=1e-5
  - OneCycleLR schedule
  - Up to 30 epochs, early stopping after 10 non-improving validation epochs
  - Rolling window: 3000 days initial, advance 300 days per iteration
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .losses import MSELoss


class Trainer:
    """Trains FS-GCLSTM on a single rolling window split.

    Args:
        model: FSGCLSTMModel instance
        device: torch.device
        lr: Learning rate (paper: 0.001)
        weight_decay: L2 regularization (paper: 1e-5)
        max_epochs: Maximum training epochs (paper: 30)
        early_stop_patience: Epochs without val improvement before stopping (paper: 10)
        checkpoint_dir: Directory to save best checkpoint
        log_every: Log training metrics every N steps
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        max_epochs: int = 30,
        early_stop_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        log_every: int = 10,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.loss_fn = MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch, return mean loss."""
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for x_seq, adj, y in loader:
            x_seq = x_seq.squeeze(0).to(self.device)  # [seq_len, N, d]
            adj = adj.squeeze(0).to(self.device)
            y = y.squeeze(0).to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x_seq, adj)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on a data loader, return mean MSE loss."""
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for x_seq, adj, y in loader:
            x_seq = x_seq.squeeze(0).to(self.device)
            adj = adj.squeeze(0).to(self.device)
            y = y.squeeze(0).to(self.device)
            pred = self.model(x_seq, adj)
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Run full training with early stopping.

        Returns:
            dict with 'best_val_loss', 'epochs_trained', 'best_epoch'
        """
        # OneCycleLR scheduler (Section IV.C)
        total_steps = self.max_epochs * max(len(train_loader), 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]["lr"],
            total_steps=total_steps,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")
        print(f"  Training for up to {self.max_epochs} epochs (early stop: {self.early_stop_patience})")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            scheduler.step()
            val_loss = self.evaluate(val_loader)
            elapsed = time.time() - t0

            if epoch % self.log_every == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{self.max_epochs} | train={train_loss:.6f} val={val_loss:.6f} | {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_dir / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    print(f"  Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f} at epoch {best_epoch})")
                    break

        # Restore best weights
        self.model.load_state_dict(torch.load(self.checkpoint_dir / "best.pt", map_location=self.device))
        return {"best_val_loss": best_val_loss, "epochs_trained": epoch, "best_epoch": best_epoch}
