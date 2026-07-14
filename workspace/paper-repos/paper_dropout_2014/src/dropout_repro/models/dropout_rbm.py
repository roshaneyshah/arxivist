"""
models/dropout_rbm.py
=====================
Dropout Restricted Boltzmann Machine (Section 8 of paper).

Implements the DropoutRBM model described in Section 8.1:

    P(r, h, v; p, θ) = P(r; p) · P(h, v | r; θ)

    P(r; p) = ∏_j  p^{r_j} (1-p)^{1-r_j}

    P(h_j=1 | r_j, v) = 1(r_j=1) · σ(b_j + ∑_i W_{ij} v_i)

where r ∈ {0,1}^F is the dropout mask, h ∈ {0,1}^F are hidden units,
v ∈ {0,1}^D are visible units.

Training uses CD-1 (Contrastive Divergence k=1) per Section 8.2:
    "Learning algorithms developed for RBMs such as Contrastive Divergence
    (Hinton et al., 2006) can be directly applied for learning Dropout RBMs.
    The only difference is that r is first sampled and only the hidden units
    that are retained are used for training."

Paper: Srivastava et al. (2014) JMLR 15:1929-1958, Section 8.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DropoutRBM(nn.Module):
    """
    Dropout Restricted Boltzmann Machine (Section 8).

    A binary RBM where hidden units are randomly masked during training
    using a Bernoulli dropout mask r ~ Bernoulli(p_hidden).

    At each training step, a new mask r is sampled for each mini-batch.
    Hidden units with r_j=0 are forced to h_j=0 (dropped from the model).

    This can be seen as a mixture of 2^F RBMs with shared weights,
    each using a different subset of hidden units (Section 8.1).

    Args:
        n_visible: Number of visible units (784 for MNIST).
        n_hidden:  Number of hidden units (256 used in paper's Figure 12).
        p_hidden:  RETENTION probability for hidden units. Default: 0.5 (Section 8).
        learning_rate: CD-1 learning rate (ASSUMED: not stated in paper for RBM).
    """

    def __init__(
        self,
        n_visible: int = 784,
        n_hidden: int = 256,
        p_hidden: float = 0.5,
        learning_rate: float = 0.01,  # ASSUMED
    ) -> None:
        super().__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.p_hidden = p_hidden
        self.learning_rate = learning_rate

        # RBM parameters: W ∈ R^{D×F}, a ∈ R^F (hidden bias), b ∈ R^D (visible bias)
        # Named to match paper notation (Section 8.1)
        self.W = nn.Parameter(
            torch.randn(n_visible, n_hidden) * 0.01
        )  # weight matrix [D, F]
        self.a = nn.Parameter(torch.zeros(n_hidden))   # hidden bias [F]
        self.b = nn.Parameter(torch.zeros(n_visible))  # visible bias [D]

    def _sample_dropout_mask(self, batch_size: int) -> torch.Tensor:
        """
        Sample Bernoulli dropout mask r ~ Bernoulli(p_hidden).

        Per Section 8.2: "a different r is sampled for each training case in
        every minibatch."

        Args:
            batch_size: Number of samples in the current batch.

        Returns:
            r: Binary mask tensor of shape [batch_size, n_hidden].
        """
        # r_j ~ Bernoulli(p_hidden): 1 with probability p_hidden (retain)
        return torch.bernoulli(
            torch.full((batch_size, self.n_hidden), self.p_hidden, device=self.W.device)
        )

    def sample_hidden(
        self, v: torch.Tensor, r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units conditioned on visible and dropout mask.

        Implements Section 8.1 Eq.:
            P(h_j=1 | r_j, v) = 1(r_j=1) · σ(b_j + ∑_i W_{ij} v_i)

        Args:
            v: Visible unit states [B, n_visible].
            r: Dropout mask [B, n_hidden] (1=retain, 0=drop).

        Returns:
            h_probs:  Hidden activation probabilities [B, n_hidden].
            h_sample: Binary hidden unit samples [B, n_hidden].
        """
        # Pre-activation: b_j + ∑_i W_{ij} v_i
        pre_act = v @ self.W + self.a  # [B, n_hidden]

        # Sigmoid activation; zero out dropped units
        h_probs = torch.sigmoid(pre_act) * r  # [B, n_hidden]; Eq. P(h_j=1|r_j,v)

        # Sample binary hidden states
        h_sample = torch.bernoulli(h_probs)  # [B, n_hidden]
        return h_probs, h_sample

    def sample_visible(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units conditioned on hidden units.

        Implements Section 8.1 Eq.:
            P(v_i=1 | h) = σ(a_i + ∑_j W_{ij} h_j)

        Args:
            h: Hidden unit states [B, n_hidden].

        Returns:
            v_probs:  Visible activation probabilities [B, n_visible].
            v_sample: Binary visible unit samples [B, n_visible].
        """
        pre_act = h @ self.W.T + self.b  # [B, n_visible]
        v_probs = torch.sigmoid(pre_act)
        v_sample = torch.bernoulli(v_probs)
        return v_probs, v_sample

    def contrastive_divergence(
        self, v0: torch.Tensor, k: int = 1
    ) -> dict[str, torch.Tensor]:
        """
        Compute CD-k parameter gradients.

        Section 8.2: "In our experiments, we use CD-1 for training dropout RBMs."

        A single dropout mask r is sampled at the start and reused for both
        positive and negative phases (this is the standard interpretation of
        dropout in the RBM context).

        Args:
            v0: Initial visible unit states (training data) [B, n_visible].
            k:  Number of Gibbs steps (paper uses k=1).

        Returns:
            Dict of gradient tensors: {'dW', 'da', 'db'}.
        """
        B = v0.shape[0]

        # Sample dropout mask — one per training case (Section 8.2)
        r = self._sample_dropout_mask(B)

        # === Positive phase: data statistics ===
        h0_probs, h0_sample = self.sample_hidden(v0, r)

        # === Negative phase: k-step Gibbs chain ===
        hk = h0_sample
        for _ in range(k):
            vk_probs, vk_sample = self.sample_visible(hk)
            hk_probs, hk_sample = self.sample_hidden(vk_sample, r)
            hk = hk_sample

        vk = vk_sample  # negative phase visible state

        # === CD-k gradient estimates ===
        # ΔW = (v0^T h0_probs - vk^T hk_probs) / B
        dW = (v0.T @ h0_probs - vk.T @ hk_probs) / B  # [n_visible, n_hidden]
        # Δa = mean(h0_probs - hk_probs)
        da = (h0_probs - hk_probs).mean(dim=0)          # [n_hidden]
        # Δb = mean(v0 - vk)
        db = (v0 - vk).mean(dim=0)                      # [n_visible]

        return {"dW": dW, "da": da, "db": db}

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute hidden activation probabilities (for feature extraction).

        At eval time, uses full hidden layer (no masking) scaled by p_hidden,
        consistent with the weight-scaling approximation.

        Args:
            v: Visible unit states [B, n_visible].

        Returns:
            h_probs: Hidden activation probabilities [B, n_hidden].
        """
        if self.training:
            B = v.shape[0]
            r = self._sample_dropout_mask(B)
            h_probs, _ = self.sample_hidden(v, r)
        else:
            # Test time: use full network, scale by p_hidden
            # (equivalent to averaging over all dropout masks)
            pre_act = v @ self.W + self.a
            h_probs = torch.sigmoid(pre_act) * self.p_hidden  # scale by retention prob
        return h_probs

    def __repr__(self) -> str:
        return (
            f"DropoutRBM("
            f"n_visible={self.n_visible}, "
            f"n_hidden={self.n_hidden}, "
            f"p_hidden={self.p_hidden})"
        )
