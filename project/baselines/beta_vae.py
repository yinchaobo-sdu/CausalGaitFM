"""beta-VAE baseline for Table 3 (representation learning comparison).

Reference: Higgins et al., "beta-VAE: Learning Basic Visual Concepts with
a Constrained Variational Framework", ICLR 2017.

Applied to 1D gait sequences: encodes sequence into latent Gaussian,
reconstructs with beta-weighted KL penalty, then classifies from latent.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BetaVAE(nn.Module):
    """beta-VAE for gait sequences: 1D-CNN encoder/decoder + latent classifier."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 256,
        num_classes: int = 4,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        beta: float = 4.0,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_dim = input_dim

        # Encoder: 1D CNN -> flatten -> mu/logvar
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> upsample -> 1D CNN transpose
        # Compute reduced length after encoder pooling
        self._reduced_len = max(1, seq_len // 8)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * self._reduced_len)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim, kernel_size=4, stride=2, padding=1),
        )

        # Classifier from latent
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: [B, T, D] -> mu, logvar: [B, latent_dim]."""
        h = x.transpose(1, 2)  # [B, D, T]
        h = self.encoder(h).squeeze(-1)  # [B, hidden_dim]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (logvar * 0.5).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """z: [B, latent_dim] -> recon: [B, T, D]."""
        h = self.decoder_fc(z)
        h = h.view(z.size(0), -1, self._reduced_len)  # [B, hidden, T']
        h = self.decoder(h)  # [B, D, T'']
        # Adjust to original seq_len
        if h.size(-1) >= self.seq_len:
            h = h[:, :, :self.seq_len]
        else:
            h = F.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        return h.transpose(1, 2)  # [B, T, D]

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        mu, _ = self.encode(x)
        return self.classifier(mu)

    def compute_loss(
        self,
        x: Tensor,
        targets: Tensor,
        lengths: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        recon = self.decode(z)
        logits = self.classifier(z)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x)
        # KL divergence
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        # Classification
        cls_loss = F.cross_entropy(logits, targets)

        total = recon_loss + self.beta * kl_loss + cls_loss
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "cls_loss": cls_loss,
            "logits": logits,
        }


__all__ = ["BetaVAE"]
