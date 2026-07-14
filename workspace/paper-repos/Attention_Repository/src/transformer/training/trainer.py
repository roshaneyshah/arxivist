"""
training/trainer.py
===================
Training loop, optimizer, LR schedule, and checkpointing.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 5.2 — Hardware and Schedule
Section 5.3 — Optimizer (Adam + warmup_rsqrt LR schedule, Eq. 3)
Section 5.4 — Regularization
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from transformer.models.transformer import Transformer
from transformer.training.losses import LabelSmoothedCrossEntropy
from transformer.utils.config import TransformerConfig
from transformer.utils.masking import MaskFactory


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------

class WarmupRsqrtScheduler(_LRScheduler):
    """
    Warmup + inverse square-root decay LR schedule.

    Paper: Section 5.3, Equation 3:
        lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

    Linearly increases LR for the first warmup_steps steps, then decays
    proportionally to the inverse square root of the step number.

    Args:
        optimizer:     PyTorch optimizer.
        d_model:       Model dimensionality (scaling factor).
        warmup_steps:  Number of linear warmup steps (paper: 4000).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Compute current learning rate per Eq. 3."""
        step = max(1, self._step)
        # Eq. 3: lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
        scale = self.d_model ** -0.5 * min(
            step ** -0.5,
            step * self.warmup_steps ** -1.5,
        )
        return [scale for _ in self.base_lrs]

    def step(self, epoch=None) -> None:
        self._step += 1
        super().step(epoch)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer:
    """
    Training harness for the Transformer model.

    Handles:
    - Training loop with token-based batching
    - Adam optimizer with warmup_rsqrt LR schedule (Section 5.3)
    - Label-smoothed cross-entropy loss (Section 5.4)
    - Checkpoint saving/loading and best-model tracking
    - Periodic logging of loss, LR, tokens/sec

    Args:
        model:       Transformer model instance.
        config:      Full TransformerConfig.
        device:      Torch device.
        output_dir:  Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: Transformer,
        config: TransformerConfig,
        device: torch.device,
        output_dir: str = "checkpoints/",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tc = config.training
        mc = config.model
        dc = config.data

        # Loss
        self.criterion = LabelSmoothedCrossEntropy(
            vocab_size=dc.vocab_size,
            smoothing=tc.label_smoothing,
            ignore_index=dc.pad_idx,
        )

        # Optimizer — Section 5.3: Adam(β1=0.9, β2=0.98, ε=1e-9)
        self.optimizer = Adam(
            model.parameters(),
            lr=1.0,               # LR is fully controlled by scheduler
            betas=(tc.beta1, tc.beta2),
            eps=tc.epsilon,
            weight_decay=tc.weight_decay,
        )

        # LR Schedule — Eq. 3
        self.scheduler = WarmupRsqrtScheduler(
            optimizer=self.optimizer,
            d_model=mc.d_model,
            warmup_steps=tc.warmup_steps,
        )

        self.global_step = 0
        self.best_val_bleu = 0.0
        self._checkpoint_paths: List[Path] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Main training loop.

        Args:
            train_loader: DataLoader yielding batches from TokenBatchSampler.
            val_loader:   Optional validation DataLoader for periodic eval.
        """
        tc = self.config.training
        self._print_training_summary(train_loader)

        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        while self.global_step < tc.max_steps:
            for batch in train_loader:
                if self.global_step >= tc.max_steps:
                    break

                loss, ntokens = self.train_step(batch)
                total_loss += loss * ntokens
                total_tokens += ntokens
                self.global_step += 1

                # Logging
                if self.global_step % tc.log_every_steps == 0:
                    elapsed = time.time() - t0
                    avg_loss = total_loss / max(total_tokens, 1)
                    lr = self.scheduler.get_lr()[0]
                    tok_per_sec = total_tokens / max(elapsed, 1e-6)
                    print(
                        f"Step {self.global_step:6d} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"tok/s {tok_per_sec:,.0f}"
                    )
                    total_loss = 0.0
                    total_tokens = 0
                    t0 = time.time()

                # Checkpoint
                if self.global_step % tc.checkpoint_every_steps == 0:
                    self.save_checkpoint(self.global_step)

        print(f"Training complete at step {self.global_step}.")

    def train_step(self, batch: Dict[str, Tensor]) -> tuple[float, int]:
        """
        Single training step.

        Args:
            batch: Dict with keys 'src', 'tgt_in', 'tgt_out' (all [B, T]).

        Returns:
            (loss_value, num_non_padding_tokens)
        """
        src = batch["src"].to(self.device)
        tgt_in = batch["tgt_in"].to(self.device)
        tgt_out = batch["tgt_out"].to(self.device)

        # Build masks
        pad_idx = self.config.data.pad_idx
        src_mask = MaskFactory.make_padding_mask(src, pad_idx)
        tgt_mask = MaskFactory.make_tgt_mask(tgt_in, pad_idx)

        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)

        # Loss
        loss = self.criterion(logits, tgt_out)

        loss.backward()

        # Gradient clipping — ASSUMED: none (not stated in paper)
        if self.config.training.gradient_clipping is not None:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clipping
            )

        self.optimizer.step()
        self.scheduler.step()

        # Count non-padding tokens for reporting
        ntokens = (tgt_out != pad_idx).sum().item()

        return loss.item(), ntokens

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int, tag: str = "") -> Path:
        """Save a checkpoint. Manages rolling window of last N checkpoints."""
        filename = f"checkpoint_step{step}{('_' + tag) if tag else ''}.pt"
        path = self.output_dir / filename
        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scheduler_step": self.scheduler._step,
                "config": self.config,
            },
            path,
        )
        self._checkpoint_paths.append(path)

        # Keep only last N checkpoints on disk
        n = self.config.training.avg_last_n_checkpoints
        while len(self._checkpoint_paths) > n * 2:  # keep buffer
            old = self._checkpoint_paths.pop(0)
            if old.exists():
                old.unlink()

        print(f"  Saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint and restore all state.

        Returns:
            Step number the checkpoint was saved at.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scheduler._step = ckpt.get("scheduler_step", ckpt["step"])
        self.global_step = ckpt["step"]
        print(f"Resumed from checkpoint: {path} (step {self.global_step})")
        return self.global_step

    def average_checkpoints(self, checkpoint_paths: List[str]) -> Dict:
        """
        Average the last N checkpoints — Section 6.1 (base: last 5, big: last 20).

        Args:
            checkpoint_paths: List of .pt checkpoint file paths.

        Returns:
            Averaged state_dict.
        """
        assert len(checkpoint_paths) >= 1, "Need at least 1 checkpoint to average."
        avg_state = None
        for path in checkpoint_paths:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt["model_state_dict"]
            if avg_state is None:
                avg_state = {k: v.float() for k, v in state.items()}
            else:
                for k in avg_state:
                    avg_state[k] += state[k].float()
        n = len(checkpoint_paths)
        for k in avg_state:
            avg_state[k] /= n
        return avg_state

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _print_training_summary(self, train_loader: DataLoader) -> None:
        """Print a summary before training begins."""
        tc = self.config.training
        mc = self.config.model
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("=" * 60)
        print("Transformer Training Summary")
        print(f"  Model params:    {total:,}  (trainable: {trainable:,})")
        print(f"  N={mc.N}, d_model={mc.d_model}, h={mc.h}, d_ff={mc.d_ff}")
        print(f"  Max steps:       {tc.max_steps:,}")
        print(f"  Warmup steps:    {tc.warmup_steps:,}")
        print(f"  Max tokens/batch:{tc.max_tokens_per_batch:,}")
        print(f"  Label smoothing: {tc.label_smoothing}")
        print(f"  Device:          {self.device}")
        print(f"  Output dir:      {self.output_dir}")
        print("=" * 60)
