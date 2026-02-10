from __future__ import annotations

from typing import Literal, Sequence

from torch import Tensor, nn

from .backbone import TemporalEncoder, TemporalEncoderConfig
from .heads import MultiTaskHeads
from .scm import SCMEncoder


class CausalGaitFM(nn.Module):
    """Integration scaffold: backbone + SCM + multi-task heads."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        scales: Sequence[int] = (1, 2, 4),
        bidirectional: bool = True,
        backend: Literal["auto", "mamba_ssm", "simple"] = "auto",
        dropout: float = 0.1,
        causal_dim: int = 32,
        domain_dim: int = 16,
        num_fall_classes: int = 3,
        num_frailty_classes: int = 5,
        num_disease_classes: int = 4,
    ) -> None:
        super().__init__()

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
        self.scm = SCMEncoder(input_dim=d_model, causal_dim=causal_dim, domain_dim=domain_dim)
        self.heads = MultiTaskHeads(
            input_dim=causal_dim,
            num_fall_classes=num_fall_classes,
            num_frailty_classes=num_frailty_classes,
            num_disease_classes=num_disease_classes,
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> dict[str, Tensor]:
        backbone_out = self.backbone(x, lengths=lengths, return_dict=True)
        scm_out = self.scm(backbone_out["sequence"])
        head_out = self.heads(scm_out["z_c"])

        return {
            **backbone_out,
            **scm_out,
            **head_out,
        }


__all__ = ["CausalGaitFM"]
