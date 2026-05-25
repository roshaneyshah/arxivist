"""Per-asset sequence encoders (LSTM, Transformer) + cross-sectional pooling."""
from __future__ import annotations

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """Apply an LSTM along the time axis of each asset's feature sequence.

    Input  : (B, k, T, F)
    Output : (B, k, H)
    """

    def __init__(self, in_features: int, hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, num_layers, batch_first=True)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, k, T, F = x.shape
        flat = x.reshape(B * k, T, F)
        out, _ = self.lstm(flat)
        last = out[:, -1, :]            # (B*k, H)
        return last.reshape(B, k, self.hidden)


class TransformerEncoder(nn.Module):
    """Two-layer self-attention along the time axis of each asset (paper Table 5)."""

    def __init__(self, in_features: int, hidden: int = 128, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 2,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, k, T, F = x.shape
        flat = self.input_proj(x.reshape(B * k, T, F))
        out = self.encoder(flat)        # (B*k, T, H)
        return out[:, -1, :].reshape(B, k, self.hidden)


class CrossSectionalAttention(nn.Module):
    """ASSUMED: scaled dot-product self-attention across assets, single layer.

    Input  : asset_emb (B, k, H), global_feat (B, F_global)
    Output : state_repr (B, H_state)
    """

    def __init__(self, hidden: int, global_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.global_proj = nn.Linear(global_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.hidden = hidden
        self.out_dim = hidden * 2  # concat of pooled-assets and global

    def forward(self, asset_emb: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(asset_emb, asset_emb, asset_emb)
        attended = self.norm(attended + asset_emb)
        pooled = attended.mean(dim=1)             # (B, H)
        g = self.global_proj(global_feat)         # (B, H)
        return torch.cat([pooled, g], dim=-1)     # (B, 2H)


def build_encoder(cfg: dict, in_features: int) -> tuple[nn.Module, int]:
    et = cfg["model"]["encoder_type"]
    if et == "lstm":
        enc = LSTMEncoder(in_features, cfg["model"]["lstm_hidden"], cfg["model"]["lstm_layers"])
        return enc, enc.hidden
    if et == "transformer":
        enc = TransformerEncoder(
            in_features,
            cfg["model"]["transformer_hidden"],
            cfg["model"]["transformer_layers"],
            cfg["model"].get("transformer_heads", 4),
        )
        return enc, enc.hidden
    raise ValueError(f"Unknown encoder_type: {et}")
