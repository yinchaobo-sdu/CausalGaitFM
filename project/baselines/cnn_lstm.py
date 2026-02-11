"""CNN-LSTM baseline for Table 3 and Table 5 comparisons.

A standard CNN + bidirectional LSTM architecture commonly used in
activity recognition / gait analysis literature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CNNLSTMModel(nn.Module):
    """CNN-LSTM baseline: 1D CNN feature extractor + bidirectional LSTM + classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        cnn_channels: tuple[int, ...] = (64, 128, 128),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()

        # 1D CNN blocks: Conv1d -> BatchNorm -> ReLU -> MaxPool
        cnn_layers = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Classifier on the final hidden state
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),  # bidirectional
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def extract_features(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """x: [B, T, D] -> features: [B, lstm_hidden*2]."""
        # CNN expects [B, C, T]
        h = x.transpose(1, 2)
        h = self.cnn(h)
        # Back to [B, T', C']
        h = h.transpose(1, 2)
        # LSTM
        output, (hn, _) = self.lstm(h)
        # Use mean pooling over time
        features = output.mean(dim=1)
        return features

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


__all__ = ["CNNLSTMModel"]
