"""Transformer baseline for Table 3 and Table 5 comparisons.

Standard Transformer encoder with positional encoding for gait sequence
classification. Demonstrates O(L^2) complexity vs Mamba's O(L).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer baseline: linear projection + Transformer encoder + CLS pooling."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def extract_features(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """x: [B, T, D] -> features: [B, d_model]."""
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = self.output_norm(h)
        # Mean pooling
        if lengths is not None:
            mask = torch.arange(h.size(1), device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
            h = (h * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            h = h.mean(dim=1)
        return h

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        features = self.extract_features(x, lengths)
        return self.classifier(features)

    def compute_loss(
        self,
        x: Tensor,
        targets: Tensor,
        lengths: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        logits = self.forward(x, lengths=lengths)
        loss = F.cross_entropy(logits, targets)
        return {"loss": loss, "logits": logits}


__all__ = ["TransformerModel"]
