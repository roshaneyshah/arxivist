"""Loss functions: MSE (Eq. 4/7) and QLIKE (Eq. 5/8), arXiv:2401.06249, Sec. 7.2/7.4."""

from __future__ import annotations

import torch
import torch.nn as nn


class QLIKELoss(nn.Module):
    """QLIKE loss: mean(pred/target - log(pred/target) - 1) (Eq. 5, single-step; Eq. 8, multi-step).

    Used for evaluation in the paper; also usable as an alternative training
    objective (training config uses MSE per Table 8's "Loss Function" field).
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
        pred: Model forecast, any shape.
        target: Ground truth, same shape as ``pred``.

        Returns:
            Scalar QLIKE loss.
        """
        ratio = pred.clamp_min(self.eps) / target.clamp_min(self.eps)
        return (ratio - torch.log(ratio) - 1.0).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss (Eq. 4, single-step; Eq. 7, multi-step)."""
    return nn.functional.mse_loss(pred, target)
