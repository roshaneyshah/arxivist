"""
models/dropout_net.py
=====================
Primary model: configurable feed-forward neural network with Bernoulli dropout.

Implements the dropout feed-forward network described in:
    Section 4 (Model Description) — formal dropout equations
    Section 5.1 (Backpropagation) — training procedure
    Section 6.1.1 (MNIST experiments) — architecture details
    Appendix B.1 — exact hyperparameters

Architecture (primary repro target, Table 2):
    Input (784) → Dropout(p_input=0.8) →
    Linear(784→1024) → ReLU → Dropout(p_hidden=0.5) →
    Linear(1024→1024) → ReLU → Dropout(p_hidden=0.5) →
    Linear(1024→1024) → ReLU → Dropout(p_hidden=0.5) →
    Linear(1024→10) → logits

CRITICAL CONVENTION NOTE:
    The paper defines p as the RETENTION probability.
    PyTorch nn.Dropout(p) takes the DROP probability.
    Therefore:
        nn.Dropout(p=1 - p_paper)
    Examples:
        p_hidden=0.5 (retain 50%) → nn.Dropout(0.5)  [same value, retain=drop=0.5]
        p_input=0.8  (retain 80%) → nn.Dropout(0.2)  [DIFFERENT: drop 20%, not 80%]

    PyTorch uses INVERTED DROPOUT: scales activations by 1/(1-p_drop) at TRAIN time,
    so no weight scaling is needed at test time. Mathematically equivalent to the
    paper's test-time weight scaling W_test = p_retain * W.

Paper: Srivastava et al. (2014) JMLR 15:1929-1958.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropoutNet(nn.Module):
    """
    Configurable L-layer feed-forward neural network with Bernoulli dropout.

    Faithfully implements the dropout model from Section 4 (Eq. 2–5):
        r_j^(l) ~ Bernoulli(p)
        ỹ^(l) = r^(l) * y^(l)           [Eq. thinned output]
        z_i^(l+1) = w_i^(l+1) ỹ^(l) + b_i^(l+1)
        y_i^(l+1) = f(z_i^(l+1))

    At test time, model.eval() disables all nn.Dropout masks (PyTorch built-in).
    No manual weight scaling is needed due to PyTorch's inverted dropout convention.

    Args:
        input_dim:   Number of input features (784 for MNIST).
        hidden_dims: List of hidden layer widths, e.g. [1024, 1024, 1024].
        num_classes: Number of output classes (10 for MNIST).
        p_hidden:    RETENTION probability for hidden layers (paper convention).
                     Default: 0.5 (Appendix B.1). Passed to nn.Dropout as 1-p_hidden.
        p_input:     RETENTION probability for input layer (paper convention).
                     Default: 0.8 (Appendix B.1). Passed to nn.Dropout as 1-p_input.
        activation:  Activation function: 'relu' (default, Appendix B.1) or 'logistic'.
        use_dropout: If False, disables all dropout (for baseline comparison).
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 10,
        p_hidden: float = 0.5,
        p_input: float = 0.8,
        activation: str = "relu",
        use_dropout: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 1024, 1024]

        # --- Validate inputs ---
        assert len(hidden_dims) >= 1, "Must have at least one hidden layer"
        assert 0.0 < p_hidden <= 1.0, f"p_hidden (retention prob) must be in (0,1], got {p_hidden}"
        assert 0.0 < p_input <= 1.0,  f"p_input  (retention prob) must be in (0,1], got {p_input}"
        assert activation in ("relu", "logistic"), \
            f"activation must be 'relu' or 'logistic', got '{activation}'"

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.p_hidden = p_hidden
        self.p_input = p_input
        self.activation_name = activation
        self.use_dropout = use_dropout

        # --- Build layers ---
        # Input dropout: paper p_input=0.8 means retain 80%, so DROP 20%
        # nn.Dropout(p) drops with probability p → pass (1 - p_paper)
        self.input_dropout = nn.Dropout(p=1.0 - p_input) if use_dropout else nn.Identity()

        # Hidden layers: Linear → Activation → Dropout
        dims = [input_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.hidden_layers.append(nn.Linear(in_d, out_d))
            # paper p_hidden=0.5 → retain 50% → drop 50% → nn.Dropout(0.5)
            # paper p_hidden=0.8 → retain 80% → drop 20% → nn.Dropout(0.2)
            drop = nn.Dropout(p=1.0 - p_hidden) if use_dropout else nn.Identity()
            self.hidden_dropouts.append(drop)

        # Output layer: NO dropout applied (standard practice; not stated but implied)
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)

        # Activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU(inplace=True)
        else:
            self.activation_fn = nn.Sigmoid()

        # Weight initialization: Kaiming uniform for ReLU (standard practice;
        # paper does not specify — ASSUMED from common DL practice)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights. Paper doesn't specify strategy — using Kaiming uniform
        for ReLU layers as it preserves activation variance (ASSUMED).
        """
        for layer in self.hidden_layers:
            if self.activation_name == "relu":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Eq. 2–5 from Section 4.

        At training time (model.train()):
            - Bernoulli masks are sampled and applied at input and each hidden layer
            - Activations are scaled by 1/(1-p_drop) = 1/p_retain (inverted dropout)
        At test time (model.eval()):
            - All dropout masks are disabled (nn.Dropout passes through unchanged)
            - No weight scaling needed (PyTorch inverted dropout convention)

        Args:
            x: Input tensor of shape [B, input_dim].

        Returns:
            logits: Unnormalized class scores of shape [B, num_classes].
        """
        assert x.dim() == 2, \
            f"Expected 2D input [B, D], got shape {tuple(x.shape)}"
        assert x.shape[1] == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {x.shape[1]}"

        # --- Input dropout ---
        # Eq. 2: r_j^(0) ~ Bernoulli(p_input), ỹ^(0) = r^(0) * x
        x = self.input_dropout(x)  # [B, input_dim]

        # --- Hidden layers ---
        # Eq. 3-5 applied at each layer l:
        #   z_i^(l+1) = w_i^(l+1) ỹ^(l) + b_i^(l+1)
        #   y^(l+1) = f(z^(l+1))
        #   r^(l+1) ~ Bernoulli(p_hidden), ỹ^(l+1) = r^(l+1) * y^(l+1)
        for linear, dropout in zip(self.hidden_layers, self.hidden_dropouts):
            x = linear(x)             # affine transform
            x = self.activation_fn(x) # non-linearity (ReLU or sigmoid)
            x = dropout(x)            # Bernoulli mask (Section 4, Eq. thinned output)

        # --- Output layer (no dropout) ---
        logits = self.output_layer(x)  # [B, num_classes]
        return logits

    def get_hidden_activations(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Return intermediate hidden activations BEFORE dropout for sparsity analysis.
        Used to reproduce Figure 8 (Section 7.2: Effect on Sparsity).

        Args:
            x: Input tensor [B, input_dim].

        Returns:
            List of activation tensors, one per hidden layer, each [B, D_l].
        """
        activations = []
        x = self.input_dropout(x)

        for linear, dropout in zip(self.hidden_layers, self.hidden_dropouts):
            x = linear(x)
            x = self.activation_fn(x)
            activations.append(x.detach().clone())  # pre-dropout activations
            x = dropout(x)

        return activations

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        act = self.activation_name
        dp = f"p_hidden={self.p_hidden}, p_input={self.p_input}" if self.use_dropout else "no dropout"
        dims = " → ".join(
            [str(self.input_dim)] + [str(d) for d in self.hidden_dims] + [str(self.num_classes)]
        )
        return (
            f"DropoutNet(\n"
            f"  dims={dims}\n"
            f"  activation={act}, {dp}\n"
            f"  max_norm_c=applied externally\n"
            f"  params={self.count_parameters():,}\n"
            f")"
        )
