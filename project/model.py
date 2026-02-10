from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from project.models.backbone import MambaBlockWrapper, TemporalEncoder, TemporalEncoderConfig
    from project.models.heads import MultiTaskHeads
    from project.models.scm import SCM_Layer
except ModuleNotFoundError:
    from models.backbone import MambaBlockWrapper, TemporalEncoder, TemporalEncoderConfig
    from models.heads import MultiTaskHeads
    from models.scm import SCM_Layer


class DecoderSSMLayer(nn.Module):
    """PreNorm -> SSM -> Dropout -> Residual block for the decoder."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        backend: Literal["auto", "mamba_ssm", "simple"] = "auto",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = MambaBlockWrapper(d_model=d_model, d_state=d_state, backend=backend)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return residual + x


class GaitDecoder(nn.Module):
    """
    Decoder from latent z=[z_c, z_d] to reconstructed gait signal x_hat.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        default_seq_len: int,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 2,
        backend: Literal["auto", "mamba_ssm", "simple"] = "auto",
        dropout: float = 0.1,
        temporal_layer_type: Literal["mamba", "lstm"] = "mamba",
    ) -> None:
        super().__init__()
        self.default_seq_len = default_seq_len
        self.temporal_layer_type = temporal_layer_type

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
        )
        self.time_embedding = nn.Parameter(torch.randn(default_seq_len, d_model) * 0.02)

        if temporal_layer_type == "mamba":
            self.ssm_layers = nn.ModuleList(
                [
                    DecoderSSMLayer(
                        d_model=d_model,
                        d_state=d_state,
                        backend=backend,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.lstm = None
        elif temporal_layer_type == "lstm":
            self.ssm_layers = None
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported temporal_layer_type: {temporal_layer_type}")

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def _time_embed(self, target_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if target_len == self.default_seq_len:
            emb = self.time_embedding
        else:
            emb = self.time_embedding.transpose(0, 1).unsqueeze(0)
            emb = F.interpolate(emb, size=target_len, mode="linear", align_corners=False)
            emb = emb.squeeze(0).transpose(0, 1)
        return emb.to(device=device, dtype=dtype)

    def forward(self, z: Tensor, target_len: int | None = None) -> Tensor:
        if z.dim() != 2:
            raise ValueError(f"GaitDecoder expects [B,latent_dim], got {tuple(z.shape)}")

        seq_len = self.default_seq_len if target_len is None else int(target_len)
        base = self.latent_proj(z).unsqueeze(1).expand(-1, seq_len, -1)
        time_embed = self._time_embed(seq_len, device=z.device, dtype=z.dtype).unsqueeze(0)
        hidden = base + time_embed

        if self.temporal_layer_type == "mamba":
            for layer in self.ssm_layers:
                hidden = layer(hidden)
        else:
            hidden, _ = self.lstm(hidden)

        hidden = self.output_norm(hidden)
        return self.output_proj(hidden)


class CausalGaitModel(nn.Module):
    """
    End-to-end model:
      TemporalEncoder -> SCM_Layer -> MultiTaskHeads
      + GaitDecoder for signal reconstruction / counterfactual generation.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 256,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        scales: Sequence[int] = (1, 2, 4),
        bidirectional: bool = True,
        backend: Literal["auto", "mamba_ssm", "simple"] = "auto",
        dropout: float = 0.1,
        causal_dim: int = 32,
        domain_dim: int = 16,
        num_disease_classes: int = 4,
        num_fall_classes: int = 3,
        num_frailty_classes: int = 5,
        decoder_n_layers: int = 2,
        decoder_layer_type: Literal["mamba", "lstm"] = "mamba",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.causal_dim = causal_dim
        self.domain_dim = domain_dim

        backbone_cfg = TemporalEncoderConfig(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            scales=tuple(scales),
            bidirectional=bidirectional,
            backend=backend,
            dropout=dropout,
        )
        self.backbone = TemporalEncoder(config=backbone_cfg)
        self.scm = SCM_Layer(
            input_dim=d_model,
            causal_dim=causal_dim,
            domain_dim=domain_dim,
        )
        self.heads = MultiTaskHeads(
            input_dim=causal_dim,
            num_disease_classes=num_disease_classes,
            num_fall_classes=num_fall_classes,
            num_frailty_classes=num_frailty_classes,
        )
        self.decoder = GaitDecoder(
            latent_dim=causal_dim + domain_dim,
            output_dim=input_dim,
            default_seq_len=seq_len,
            d_model=d_model,
            d_state=d_state,
            n_layers=decoder_n_layers,
            backend=backend,
            dropout=dropout,
            temporal_layer_type=decoder_layer_type,
        )

    @staticmethod
    def _sample_domain_shuffle_indices(batch_domain_ids: Tensor) -> Tensor:
        if batch_domain_ids.dim() != 1:
            raise ValueError(f"`batch_domain_ids` must be [B], got {tuple(batch_domain_ids.shape)}")

        batch_size = batch_domain_ids.size(0)
        indices = torch.arange(batch_size, device=batch_domain_ids.device)
        shuffled = torch.empty(batch_size, dtype=torch.long, device=batch_domain_ids.device)

        for i in range(batch_size):
            different_domain = indices[batch_domain_ids != batch_domain_ids[i]]
            if different_domain.numel() == 0:
                candidates = indices[indices != i]
            else:
                candidates = different_domain

            if candidates.numel() == 0:
                shuffled[i] = i
            else:
                rand_pos = torch.randint(0, candidates.numel(), (1,), device=batch_domain_ids.device)
                shuffled[i] = candidates[rand_pos]
        return shuffled

    def decode_from_latent(self, z_c: Tensor, z_d: Tensor, target_len: int | None = None) -> Tensor:
        z = torch.cat((z_c, z_d), dim=-1)
        return self.decoder(z, target_len=target_len)

    def generate_counterfactuals(
        self,
        z_c: Tensor,
        z_d: Tensor,
        batch_domain_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Domain intervention:
          keep z_c fixed, shuffle z_d across domains, decode x_cf.
        """
        if z_c.dim() != 2 or z_d.dim() != 2:
            raise ValueError("z_c and z_d must be [B,D].")
        if z_c.size(0) != z_d.size(0):
            raise ValueError("z_c and z_d must share batch size.")
        if batch_domain_ids.size(0) != z_c.size(0):
            raise ValueError("batch_domain_ids must match z_c batch size.")

        shuffle_idx = self._sample_domain_shuffle_indices(batch_domain_ids=batch_domain_ids)
        z_d_shuffled = z_d[shuffle_idx]
        x_cf = self.decode_from_latent(z_c=z_c, z_d=z_d_shuffled, target_len=self.seq_len)
        return x_cf, shuffle_idx

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
        sample_domain: bool = True,
    ) -> dict[str, Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"CausalGaitModel expects [B,T,D] or [T,D], got {tuple(x.shape)}")

        backbone_out = self.backbone(x, lengths=lengths, return_dict=True)
        scm_out = self.scm(backbone_out["sequence"], sample_domain=sample_domain)
        head_out = self.heads(scm_out["z_c"])

        recon = self.decode_from_latent(
            z_c=scm_out["z_c"],
            z_d=scm_out["z_d"],
            target_len=x.size(1),
        )
        return {
            **backbone_out,
            **scm_out,
            **head_out,
            "recon": recon,
            "recon_target": x,
        }


__all__ = ["GaitDecoder", "CausalGaitModel"]

