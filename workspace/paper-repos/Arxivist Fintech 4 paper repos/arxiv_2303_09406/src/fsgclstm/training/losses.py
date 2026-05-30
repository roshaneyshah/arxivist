"""
training/losses.py
==================
Loss functions for FS-GCLSTM training.

Paper: Liu (2023/2025) — arXiv:2303.09406
NOTE: Training loss not explicitly stated. MSE assumed (conf: 0.70).
"""
import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Mean Squared Error loss.

    ASSUMED: This is the training objective. Paper evaluates MSE on test
    sets but does not explicitly state the training loss (conf: 0.70).
    # TODO: verify training loss from paper or supplementary materials
    """
    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)
