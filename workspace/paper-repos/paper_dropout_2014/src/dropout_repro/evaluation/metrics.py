"""
evaluation/metrics.py
=====================
Evaluation metrics for the Dropout reproduction.

Implements:
    - Classification error rate (primary metric for all classification tasks)
    - Sparsity statistics (Section 7.2, Figure 8: Effect on Sparsity)
    - Regularizer comparison table formatter (Table 9, Section 6.5)

Paper: Srivastava et al. (2014) JMLR 15:1929-1958, Sections 6-7.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_error_rate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute classification error rate and loss on a dataset.

    Primary evaluation metric used throughout the paper (Tables 2-9).
    "Classification error %" = (1 - accuracy) * 100.

    Args:
        model:  Trained neural network (must support model.eval()).
        loader: DataLoader for the evaluation split.
        device: torch.device for inference.

    Returns:
        Dict with keys:
            'error_rate': classification error percentage (lower is better)
            'accuracy':   classification accuracy percentage
            'loss':       mean cross-entropy loss
            'n_samples':  total number of evaluated samples
    """
    import torch.nn.functional as F

    was_training = model.training
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.shape[0]
            total_loss += loss.item()
            n_batches += 1

    accuracy = (total_correct / total_samples) * 100.0
    error_rate = 100.0 - accuracy

    if was_training:
        model.train()

    return {
        "error_rate": error_rate,
        "accuracy": accuracy,
        "loss": total_loss / n_batches,
        "n_samples": total_samples,
    }


def compute_sparsity_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_batches: int = 10,
) -> Dict[str, object]:
    """
    Compute hidden unit activation sparsity statistics.

    Reproduces the analysis from Section 7.2 (Figure 8):
        "We found that as a side-effect of doing dropout, the activations of
        the hidden units become sparse, even when no sparsity inducing
        regularizers are present."

    A "good sparse model" (Section 7.2) satisfies:
        - Only a few highly activated units per data case
        - Low average activation per unit across data cases

    Args:
        model:    DropoutNet with get_hidden_activations() method.
        loader:   DataLoader for evaluation data.
        device:   torch.device.
        n_batches: Number of mini-batches to sample (default: 10 to match paper's
                   "random mini-batch taken from the test set").

    Returns:
        Dict with per-layer sparsity statistics:
            'layer_{i}_mean_activation': mean activation value (paper: ~0.7 with dropout, ~2.0 without)
            'layer_{i}_pct_near_zero':   % of activations < 0.1 (sparsity indicator)
            'layer_{i}_activations':     flat array of all activation values (for histogram)
            'layer_{i}_mean_activations_per_unit': per-unit mean activation (for mean histogram)
    """
    was_training = model.training
    model.eval()

    # Accumulate activations across batches
    all_activations: Optional[List[List]] = None

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            x = x.to(device)

            # get_hidden_activations returns pre-dropout activations per layer
            layer_acts = model.get_hidden_activations(x)  # List[Tensor[B, D_l]]

            if all_activations is None:
                all_activations = [[] for _ in layer_acts]

            for i, act in enumerate(layer_acts):
                all_activations[i].append(act.cpu().numpy())

    if was_training:
        model.train()

    if all_activations is None:
        return {}

    results = {}
    for i, layer_batches in enumerate(all_activations):
        # Concatenate batches: [total_samples, D_l]
        acts = np.concatenate(layer_batches, axis=0)

        # Mean activation per unit (across samples) — for "mean activation" histogram (Fig 8 left)
        mean_per_unit = acts.mean(axis=0)  # [D_l]

        # Overall mean activation — paper reports ~2.0 without dropout, ~0.7 with
        overall_mean = float(mean_per_unit.mean())

        # Fraction of activations near zero (sparsity indicator) — for "activation" histogram (Fig 8 right)
        pct_near_zero = float((acts < 0.1).mean() * 100.0)

        results[f"layer_{i}_mean_activation"] = overall_mean
        results[f"layer_{i}_pct_near_zero"] = pct_near_zero
        results[f"layer_{i}_mean_activations_per_unit"] = mean_per_unit.tolist()
        results[f"layer_{i}_activations_sample"] = acts.flatten()[:5000].tolist()  # subsample for storage

    return results


def compute_regularizer_comparison(
    results: Dict[str, float],
) -> str:
    """
    Format a Table 9-style comparison of regularization methods.

    Args:
        results: Dict mapping method name to test error percentage.
                 Example: {"L2": 1.62, "Dropout+Max-norm": 1.05}

    Returns:
        Formatted ASCII table string.
    """
    # Paper's expected results (Table 9, Section 6.5)
    paper_results = {
        "L2": 1.62,
        "L2 + L1": 1.60,
        "L2 + KL-sparsity": 1.55,
        "Max-norm": 1.35,
        "Dropout + L2": 1.25,
        "Dropout + Max-norm": 1.05,
    }

    lines = []
    lines.append("\n" + "=" * 55)
    lines.append("Regularizer Comparison on MNIST (Table 9)")
    lines.append("=" * 55)
    lines.append(f"{'Method':<25} {'Repro Error':>12} {'Paper Error':>12}")
    lines.append("-" * 55)

    for method, repro_err in sorted(results.items(), key=lambda x: x[1]):
        paper_err = paper_results.get(method, None)
        paper_str = f"{paper_err:.2f}%" if paper_err is not None else "N/A"
        lines.append(f"{method:<25} {repro_err:>11.2f}% {paper_str:>12}")

    lines.append("=" * 55)
    return "\n".join(lines)


def print_result_vs_paper(
    method_name: str,
    repro_error: float,
    paper_error: float,
    tolerance: float = 0.3,
) -> None:
    """
    Print a simple pass/fail comparison against the paper's reported result.

    Args:
        method_name:  Human-readable method description.
        repro_error:  Reproduced test error (%).
        paper_error:  Paper's reported test error (%).
        tolerance:    Acceptable absolute deviation (default: ±0.3%).
    """
    diff = abs(repro_error - paper_error)
    status = "✓ PASS" if diff <= tolerance else "✗ FAIL"
    print(f"\n{status} — {method_name}")
    print(f"  Reproduced: {repro_error:.2f}%")
    print(f"  Paper:      {paper_error:.2f}%")
    print(f"  Δ = {repro_error - paper_error:+.2f}% (tolerance: ±{tolerance}%)")
