from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _load_mamba_class() -> Optional[type]:
    candidates = [
        ("mamba_ssm", "Mamba"),
        ("mamba_ssm.modules.mamba_simple", "Mamba"),
    ]
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            mamba_cls = getattr(module, class_name, None)
            if mamba_cls is not None:
                return mamba_cls
        except Exception:
            continue
    return None


def _build_mamba_block(mamba_cls: type, d_model: int, d_state: int) -> nn.Module:
    for kwargs in (
        {"d_model": d_model, "d_state": d_state, "d_conv": 4, "expand": 2},
        {"d_model": d_model, "d_state": d_state},
    ):
        try:
            return mamba_cls(**kwargs)
        except TypeError:
            continue

    # Last fallback for uncommon constructor signatures.
    return mamba_cls(d_model, d_state)


@dataclass
class TemporalEncoderConfig:
    input_dim: int
    d_model: int = 128
    d_state: int = 16
    n_layers: int = 4
    scales: tuple[int, ...] = (1, 2, 4)
    bidirectional: bool = True
    backend: Literal["auto", "mamba_ssm", "simple"] = "auto"
    dropout: float = 0.1


class SimpleSelectiveSSMBlock(nn.Module):
    """Pure PyTorch fallback block approximating selective SSM in Eqs. (2)-(4)."""

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.delta_proj = nn.Linear(d_model, d_model)
        self.a_log = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.b = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"SimpleSelectiveSSMBlock expects [B,T,D], got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        delta = F.softplus(self.delta_proj(x))  # Eq. (4)

        a_pos = F.softplus(self.a_log) + 1e-4
        b_param = self.b

        state = x.new_zeros(batch_size, self.d_model, self.d_state)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [B, D]
            delta_t = delta[:, t, :].unsqueeze(-1)  # [B, D, 1]

            a_bar_t = torch.exp(-delta_t * a_pos.unsqueeze(0))
            b_bar_t = delta_t * b_param.unsqueeze(0)
            state = a_bar_t * state + b_bar_t * x_t.unsqueeze(-1)  # Eq. (2)

            y_t = state.sum(dim=-1)  # Eq. (3) with simplified C_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return self.out_proj(y)


class MambaBlockWrapper(nn.Module):
    """Unifies official mamba-ssm and simplified fallback implementation."""

    _auto_fallback_warned = False

    def __init__(
        self,
        d_model: int,
        d_state: int,
        backend: Literal["auto", "mamba_ssm", "simple"] = "auto",
    ) -> None:
        super().__init__()

        resolved_backend = backend
        mamba_cls = _load_mamba_class()

        if backend == "auto":
            if mamba_cls is None:
                resolved_backend = "simple"
                if not MambaBlockWrapper._auto_fallback_warned:
                    warnings.warn(
                        "mamba_ssm not found. Falling back to SimpleSelectiveSSMBlock.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    MambaBlockWrapper._auto_fallback_warned = True
            else:
                resolved_backend = "mamba_ssm"

        if resolved_backend == "mamba_ssm":
            if mamba_cls is None:
                raise ImportError(
                    "backend='mamba_ssm' requested but mamba_ssm is not available. "
                    "Install `mamba-ssm` or set backend='simple' / backend='auto'."
                )
            self.block = _build_mamba_block(mamba_cls, d_model=d_model, d_state=d_state)
        elif resolved_backend == "simple":
            self.block = SimpleSelectiveSSMBlock(d_model=d_model, d_state=d_state)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend = resolved_backend

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualSSMLayer(nn.Module):
    """PreNorm -> SSM -> Dropout -> Residual."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        backend: Literal["auto", "mamba_ssm", "simple"],
        dropout: float,
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


class _ScaleEncoder(nn.Module):
    def __init__(self, config: TemporalEncoderConfig, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.bidirectional = config.bidirectional

        self.forward_layers = nn.ModuleList(
            [
                ResidualSSMLayer(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    backend=config.backend,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )

        if self.bidirectional:
            self.backward_layers = nn.ModuleList(
                [
                    ResidualSSMLayer(
                        d_model=config.d_model,
                        d_state=config.d_state,
                        backend=config.backend,
                        dropout=config.dropout,
                    )
                    for _ in range(config.n_layers)
                ]
            )
            merged_dim = config.d_model * 2
        else:
            self.backward_layers = None
            merged_dim = config.d_model

        self.merge_proj = nn.Linear(merged_dim, config.d_model)

    def _downsample(self, x: Tensor) -> Tensor:
        if self.scale == 1:
            return x
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=self.scale, stride=self.scale, ceil_mode=True)
        return x.transpose(1, 2)

    @staticmethod
    def _run_stack(x: Tensor, layers: nn.ModuleList) -> Tensor:
        for layer in layers:
            x = layer(x)
        return x

    @staticmethod
    def _upsample_to(x: Tensor, target_len: int) -> Tensor:
        if x.size(1) == target_len:
            return x
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        target_len = x.size(1)
        x_scaled = self._downsample(x)

        fwd = self._run_stack(x_scaled, self.forward_layers)
        branches = [fwd]

        if self.bidirectional and self.backward_layers is not None:
            bwd = torch.flip(x_scaled, dims=[1])
            bwd = self._run_stack(bwd, self.backward_layers)
            bwd = torch.flip(bwd, dims=[1])
            branches.append(bwd)

        merged = self.merge_proj(torch.cat(branches, dim=-1))
        return self._upsample_to(merged, target_len)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        scales: Sequence[int] = (1, 2, 4),
        bidirectional: bool = True,
        backend: Literal["auto", "mamba_ssm", "simple"] = "auto",
        dropout: float = 0.1,
        config: Optional[TemporalEncoderConfig] = None,
    ) -> None:
        super().__init__()

        if config is None:
            if input_dim is None:
                raise ValueError("`input_dim` is required when `config` is not provided.")
            config = TemporalEncoderConfig(
                input_dim=input_dim,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                scales=tuple(scales),
                bidirectional=bidirectional,
                backend=backend,
                dropout=dropout,
            )

        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        self.scale_encoders = nn.ModuleList([_ScaleEncoder(config, scale=s) for s in config.scales])
        self.scale_logits = nn.Parameter(torch.zeros(len(config.scales)))
        self.output_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def _masked_mean(sequence: Tensor, lengths: Optional[Tensor]) -> Tensor:
        if lengths is None:
            return sequence.mean(dim=1)

        if lengths.dim() != 1 or lengths.numel() != sequence.size(0):
            raise ValueError(
                "`lengths` must be a 1D tensor with length equal to batch size. "
                f"Got lengths shape: {tuple(lengths.shape)}"
            )

        batch, seq_len, _ = sequence.shape
        clipped = lengths.to(device=sequence.device).long().clamp(min=1, max=seq_len)
        mask = torch.arange(seq_len, device=sequence.device).unsqueeze(0) < clipped.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1)
        return (sequence * mask).sum(dim=1) / denom

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> dict[str, Tensor] | Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(f"TemporalEncoder expects [B,T,D] or [T,D], got {tuple(x.shape)}")

        hidden = self.input_proj(x)
        per_scale = [encoder(hidden) for encoder in self.scale_encoders]

        weights = torch.softmax(self.scale_logits, dim=0)
        fused = torch.zeros_like(hidden)
        for idx, scale_out in enumerate(per_scale):
            fused = fused + weights[idx] * scale_out

        sequence = self.output_norm(hidden + fused)
        pooled = self._masked_mean(sequence, lengths=lengths)

        if return_dict:
            return {
                "sequence": sequence,
                "pooled": pooled,
                "scale_weights": weights,
            }
        return sequence


__all__ = [
    "TemporalEncoderConfig",
    "SimpleSelectiveSSMBlock",
    "MambaBlockWrapper",
    "TemporalEncoder",
]

