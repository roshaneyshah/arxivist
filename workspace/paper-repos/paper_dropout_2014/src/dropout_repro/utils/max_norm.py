"""
utils/max_norm.py
=================
Max-norm regularization constraint (Section 5.1, Appendix A.3).

Implements the constraint ||w||_2 <= c applied to the incoming weight vector
of each hidden unit after every gradient update step.

Paper equation (Section 5.1):
    "the neural network was optimized under the constraint ||w||_2 <= c.
     This constraint was imposed during optimization by projecting w onto
     the surface of a ball of radius c, whenever w went out of it."

From Appendix A.3:
    "Typical values of c range from 3 to 4."
    (Appendix B.1 states c=2 specifically for MNIST.)

Paper: Srivastava et al. (2014) JMLR 15:1929-1958, Section 5.1, Appendix A.3.
"""

import torch
import torch.nn as nn


def apply_max_norm_constraint(model: nn.Module, max_norm_c: float) -> None:
    """
    Project all hidden-layer weight vectors onto the L2 ball of radius c.

    Applied IN-PLACE after every optimizer.step() call. Only applied to
    hidden Linear layers — NOT to the output layer, per paper convention
    (the output layer is a softmax classifier, not subject to this constraint).

    Implementation of constraint: ||w_i||_2 <= c for each row w_i of W.
    Projection: if ||w_i||_2 > c, rescale w_i = w_i * (c / ||w_i||_2).
                if ||w_i||_2 <= c, leave w_i unchanged (scale = 1.0).

    This is also called "max-norm regularization" (Srebro & Shraibman, 2005),
    referenced in paper Section 5.1.

    Args:
        model:      The DropoutNet (or any nn.Module with hidden Linear layers).
        max_norm_c: Upper bound c on the L2 norm of any incoming weight vector.
                    Typical: 2–4. Paper uses c=2 for MNIST (Appendix B.1).

    Example:
        >>> optimizer.step()
        >>> apply_max_norm_constraint(model, max_norm_c=2.0)
    """
    assert max_norm_c > 0, f"max_norm_c must be positive, got {max_norm_c}"

    # Collect hidden Linear layers (skip the final output layer)
    linear_layers = [
        module for module in model.modules()
        if isinstance(module, nn.Linear)
    ]

    # The output layer is the last Linear in a feed-forward net;
    # apply max-norm only to hidden layers (all but last)
    hidden_linear_layers = linear_layers[:-1]

    with torch.no_grad():
        for layer in hidden_linear_layers:
            # W shape: [D_out, D_in]
            # Each row w_i is the incoming weight vector for hidden unit i
            W = layer.weight  # [D_out, D_in]

            # Compute L2 norm of each row (incoming weight vector per unit)
            # norms shape: [D_out]
            norms = W.norm(p=2, dim=1, keepdim=True)  # [D_out, 1]

            # Compute scale factor: clamp to 1.0 max so we never scale UP
            # scale = min(1.0, c / ||w_i||_2)
            scale = (max_norm_c / norms).clamp(max=1.0)  # [D_out, 1]

            # Apply scaling in-place
            # W = W * scale broadcasts over D_in dimension
            W.mul_(scale)  # [D_out, D_in]


def check_max_norm_satisfied(model: nn.Module, max_norm_c: float) -> dict:
    """
    Diagnostic: verify max-norm constraint is satisfied for all hidden layers.

    Args:
        model:      The DropoutNet module.
        max_norm_c: Constraint upper bound.

    Returns:
        Dict with per-layer statistics: max_norm, mean_norm, satisfied (bool).
    """
    results = {}
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    hidden_layers = linear_layers[:-1]

    with torch.no_grad():
        for i, layer in enumerate(hidden_layers):
            norms = layer.weight.norm(p=2, dim=1)
            results[f"hidden_layer_{i}"] = {
                "max_norm":  norms.max().item(),
                "mean_norm": norms.mean().item(),
                "min_norm":  norms.min().item(),
                "satisfied": bool((norms <= max_norm_c + 1e-6).all().item()),
            }

    return results
