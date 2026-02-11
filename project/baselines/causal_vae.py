"""CausalVAE baseline for Table 3 (causal representation learning comparison).

Reference: Yang et al., "CausalVAE: Disentangled Representation Learning via
Neural Structural Causal Models", CVPR 2021.

Simplified implementation adapted for 1D gait sequences: adds a causal layer
(linear SCM on latent variables) on top of the VAE framework to encourage
causal structure in the latent space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CausalLayer(nn.Module):
    """Simplified causal layer: learns a weighted adjacency matrix over latent dims."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        # Raw adjacency weights (learned)
        self.adj_weight = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.adj_weight)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Apply causal transformation: z_causal = z @ (I + A)^T.

        Returns transformed z and DAG regularization loss.
        """
        # Soft-threshold adjacency to encourage sparsity
        A = torch.sigmoid(self.adj_weight) * (1 - torch.eye(z.size(-1), device=z.device))
        # Causal transformation
        z_causal = z @ (torch.eye(z.size(-1), device=z.device) + A).T
        # DAG constraint: tr(e^(A * A)) - d  (NOTEARS-style)
        A_sq = A * A
        dag_loss = torch.trace(torch.matrix_exp(A_sq)) - A_sq.size(0)
        return z_causal, dag_loss


class CausalVAE(nn.Module):
    """CausalVAE for gait sequences: VAE + causal layer in latent space."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 256,
        num_classes: int = 4,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        beta: float = 1.0,
        gamma: float = 0.01,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma  # weight for DAG loss
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_dim = input_dim

        # Encoder
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

        # Causal layer
        self.causal_layer = CausalLayer(latent_dim)

        # Decoder
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

        # Classifier from causal latent
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = x.transpose(1, 2)
        h = self.encoder(h).squeeze(-1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = (logvar * 0.5).exp()
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), -1, self._reduced_len)
        h = self.decoder(h)
        if h.size(-1) >= self.seq_len:
            h = h[:, :, :self.seq_len]
        else:
            h = F.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        return h.transpose(1, 2)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        mu, _ = self.encode(x)
        z_causal, _ = self.causal_layer(mu)
        return self.classifier(z_causal)

    def compute_loss(
        self,
        x: Tensor,
        targets: Tensor,
        lengths: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_causal, dag_loss = self.causal_layer(z)

        recon = self.decode(z_causal)
        logits = self.classifier(z_causal)

        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        cls_loss = F.cross_entropy(logits, targets)

        total = recon_loss + self.beta * kl_loss + cls_loss + self.gamma * dag_loss
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "cls_loss": cls_loss,
            "dag_loss": dag_loss,
            "logits": logits,
        }


__all__ = ["CausalVAE"]
