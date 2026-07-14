"""
models/transformer.py
=====================
Top-level Transformer model assembling encoder, decoder, embeddings,
positional encoding, and the final linear projection.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 3 — full model architecture (Figure 1)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from transformer.models.attention import MultiHeadAttention
from transformer.models.decoder import Decoder
from transformer.models.embeddings import PositionalEncoding, TokenEmbedding
from transformer.models.encoder import Encoder
from transformer.utils.config import TransformerConfig


class Transformer(nn.Module):
    """
    Full Transformer model for sequence-to-sequence tasks.

    Paper: "Attention Is All You Need", Vaswani et al. (2017), Figure 1.
    Architecture:
        src → TokenEmbedding → PositionalEncoding → Encoder → memory
        tgt → TokenEmbedding → PositionalEncoding → Decoder(memory) → Linear → logits

    Weight tying (Section 3.4, ASSUMED 3-way, confidence 0.82):
        encoder_embedding.weight == decoder_embedding.weight == output_projection.weight^T

    Args:
        config: TransformerConfig dataclass (loaded from YAML).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        mc = config.model
        dc = config.data

        self.d_model = mc.d_model
        self.pad_idx = dc.pad_idx

        # Embeddings + positional encoding
        self.src_embedding = TokenEmbedding(vocab_size=dc.vocab_size, d_model=mc.d_model)
        self.tgt_embedding = TokenEmbedding(vocab_size=dc.vocab_size, d_model=mc.d_model)
        self.src_pos_enc = PositionalEncoding(d_model=mc.d_model, max_len=mc.max_seq_len, dropout=mc.dropout)
        self.tgt_pos_enc = PositionalEncoding(d_model=mc.d_model, max_len=mc.max_seq_len, dropout=mc.dropout)

        # Encoder and decoder stacks
        self.encoder = Encoder(d_model=mc.d_model, N=mc.N, h=mc.h, d_ff=mc.d_ff, dropout=mc.dropout)
        self.decoder = Decoder(d_model=mc.d_model, N=mc.N, h=mc.h, d_ff=mc.d_ff, dropout=mc.dropout)

        # Output projection: [d_model] → [vocab_size]  (Section 3.4)
        self.output_projection = nn.Linear(mc.d_model, dc.vocab_size, bias=False)

        # 3-way weight tying — Section 3.4, ASSUMED confidence 0.82
        if mc.weight_tying:
            self.tie_weights()

        self._init_weights()
        self._log_param_count()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all linear layers not covered by weight tying."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight is not self.src_embedding.embedding.weight:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def tie_weights(self) -> None:
        """
        3-way weight tying: encoder embed = decoder embed = output projection^T.
        Section 3.4: "share the same weight matrix between the two embedding layers
        and the pre-softmax linear transformation".
        ASSUMED: encoder and decoder share the same embedding matrix.
        SIR confidence: 0.82
        """
        self.tgt_embedding.embedding.weight = self.src_embedding.embedding.weight
        self.output_projection.weight = self.src_embedding.embedding.weight

    def _log_param_count(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._num_params = total
        self._num_trainable = trainable

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Run the encoder stack.

        Args:
            src:      [B, T_src]  source token ids
            src_mask: [B, 1, 1, T_src]  padding mask (True = attend)

        Returns:
            memory [B, T_src, d_model]
        """
        x = self.src_pos_enc(self.src_embedding(src))  # [B, T_src, d_model]
        return self.encoder(x, mask=src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Run the decoder stack.

        Args:
            tgt:      [B, T_tgt]  target token ids (shifted right)
            memory:   [B, T_src, d_model]  encoder output
            src_mask: [B, 1, 1, T_src]
            tgt_mask: [B, 1, T_tgt, T_tgt]

        Returns:
            [B, T_tgt, d_model]
        """
        x = self.tgt_pos_enc(self.tgt_embedding(tgt))  # [B, T_tgt, d_model]
        return self.decoder(x, memory=memory, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Full forward pass: src + tgt → logits.

        Args:
            src:      [B, T_src]  source token ids
            tgt:      [B, T_tgt]  target token ids (shifted right, teacher-forced)
            src_mask: [B, 1, 1, T_src]  source padding mask
            tgt_mask: [B, 1, T_tgt, T_tgt]  target causal + padding mask

        Returns:
            logits [B, T_tgt, vocab_size]  (no softmax — loss fn handles it)
        """
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = self.output_projection(dec_out)  # [B, T_tgt, vocab_size]
        return logits

    def __repr__(self) -> str:
        return (
            f"Transformer(N={len(self.encoder.layers)}, "
            f"d_model={self.d_model}, "
            f"params={self._num_params:,})"
        )
