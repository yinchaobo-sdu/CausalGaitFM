"""Shared backbone for baseline models.

Uses the same Mamba-based TemporalEncoder as CausalGaitFM
to ensure fair comparison (only the domain generalization strategy differs).
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor, nn

from project.models.backbone import TemporalEncoder, TemporalEncoderConfig


class SharedBackbone(nn.Module):
    """Mamba backbone + classifier head (shared across baselines)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        scales: Sequence[int] = (1, 2, 4),
        bidirectional: bool = True,
        backend: str = "auto",
        dropout: float = 0.1,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        cfg = TemporalEncoderConfig(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            scales=tuple(scales),
            bidirectional=bidirectional,
            backend=backend,
            dropout=dropout,
        )
        self.backbone = TemporalEncoder(config=cfg)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.d_model = d_model

    def extract_features(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Return pooled features [B, d_model]."""
        out = self.backbone(x, lengths=lengths, return_dict=True)
        return out["pooled"]

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        features = self.extract_features(x, lengths)
        return self.classifier(features)


__all__ = ["SharedBackbone"]
