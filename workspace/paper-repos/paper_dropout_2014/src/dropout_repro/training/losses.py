"""
training/losses.py
==================
Loss functions used in the paper.

Primary: cross-entropy loss for classification (MNIST, SVHN, CIFAR, TIMIT, Reuters).
Secondary: Code Quality metric (negative KL divergence) for Alternative Splicing.

Paper: Srivastava et al. (2014) JMLR 15:1929-1958.
"""

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Standard cross-entropy loss for classification.

    Primary training objective for MNIST and all other classification tasks.
    The paper uses cross-entropy implicitly (standard for neural net classifiers).

    Args:
        logits:  Unnormalized class scores [B, C].
        targets: Integer class labels [B].

    Returns:
        Scalar mean cross-entropy loss.
    """
    assert logits.dim() == 2, f"Expected logits [B, C], got {tuple(logits.shape)}"
    assert targets.dim() == 1, f"Expected targets [B], got {tuple(targets.shape)}"
    assert logits.shape[0] == targets.shape[0], "Batch size mismatch"

    return F.cross_entropy(logits, targets)


def l1_regularization(model: torch.nn.Module, weight: float) -> torch.Tensor:
    """
    L1 regularization term (lasso, Tibshirani 1996).

    Used in the regularizer comparison ablation (Table 9, Section 6.5).

    Args:
        model:  The neural network.
        weight: L1 penalty coefficient (λ).

    Returns:
        Scalar L1 penalty term.
    """
    l1_sum = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
    return weight * l1_sum


def kl_sparsity_loss(
    activations: list[torch.Tensor],
    target_sparsity: float = 0.05,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    KL-sparsity regularizer for inducing sparse hidden representations.

    Minimizes KL(ρ || ρ̂) where ρ is the target sparsity and ρ̂ is the
    mean activation of each hidden unit over the batch.

    Used in the regularizer comparison ablation (Table 9, Section 6.5):
    "L2 + KL-sparsity: 1.55% error"

    KL(ρ || ρ̂) = ρ log(ρ/ρ̂) + (1-ρ) log((1-ρ)/(1-ρ̂))

    Args:
        activations:     List of hidden layer activation tensors [B, D_l].
        target_sparsity: Desired mean activation ρ (default 0.05).
        weight:          Penalty coefficient.

    Returns:
        Scalar KL sparsity penalty.
    """
    eps = 1e-8
    total_kl = torch.tensor(0.0)

    for act in activations:
        # Mean activation per hidden unit across the batch
        rho_hat = act.mean(dim=0).clamp(eps, 1.0 - eps)  # [D_l]
        rho = torch.tensor(target_sparsity, device=act.device)

        # KL divergence: ρ log(ρ/ρ̂) + (1-ρ) log((1-ρ)/(1-ρ̂))
        kl = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        total_kl = total_kl + kl.mean()

    return weight * total_kl
