"""Training script for CausalGaitFM.

Supports three evaluation protocols (paper Section 2.1):
  - cross_domain : train on source domains, test on held-out target domain
  - in_domain    : k-fold cross-validation within each dataset
  - loso         : leave-one-subject-out cross-validation

Also supports training with synthetic dummy data for quick debugging.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import signal
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from project.data.dataset import create_dataloaders, load_processed_datasets
from project.model import CausalGaitModel
from project.utils.losses import irm_penalty
from project.utils.metrics import calculate_metrics
from project.utils.visualization import visualize_latent_space

try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    _USE_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import GradScaler as _GradScaler
    from torch.cuda.amp import autocast as _autocast

    _USE_TORCH_AMP = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training configuration aligned with paper Section 4.8."""

    # Data
    input_dim: int = 32
    seq_len: int = 256
    processed_dir: str = "data/processed"
    dataset_names: list[str] = field(default_factory=lambda: [
        "daphnet", "ucihar", "pamap2", "mhealth", "wisdm", "opportunity",
    ])

    # Evaluation protocol
    eval_mode: str = "cross_domain"  # cross_domain | in_domain | loso
    target_domain: str = "daphnet"
    fold: int = 0
    n_folds: int = 5

    # Runtime / device
    device: str = "auto"  # auto|cuda|cpu

    # Training hyperparams (aligned with paper Sec 4.8)
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 15
    num_workers: int = 4

    # DataLoader performance knobs
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # OOM resiliency
    auto_batch: bool = False
    min_batch_size: int = 4

    # Stop / resume control
    resume_from: str | None = None
    run_id: str = "default"
    control_dir: str = "outputs/control"
    check_stop_every: int = 20
    save_every_steps: int = 200
    pipeline_progress_file: str | None = None
    pipeline_autosave_sec: int = 0

    # AMP and CUDA runtime
    use_amp: bool | None = None  # None => auto (enabled on CUDA)
    amp_dtype: str = "fp16"  # fp16|bf16
    allow_tf32: bool = True
    cudnn_benchmark: bool = True

    # Architecture (paper Sec 4.8)
    d_model: int = 128
    d_state: int = 16
    n_layers: int = 4
    scales: tuple[int, ...] = (1, 2, 4)
    bidirectional: bool = True
    backend: str = "auto"
    dropout: float = 0.1
    causal_dim: int = 32
    domain_dim: int = 16
    decoder_n_layers: int = 2
    decoder_layer_type: str = "mamba"

    # Task classes
    num_domains: int = 6
    num_disease_classes: int = 4
    num_fall_classes: int = 3
    num_frailty_classes: int = 5

    # IRM annealing (paper Sec 4.8: annealed over first 500 iterations)
    irm_warmup_iters: int = 500

    # Loss weights (paper Eq. 11)
    beta1_kl: float = 1e-2
    beta2_cls: float = 1.0
    beta4_dag: float = 1e-2
    beta5_hsic: float = 1e-2
    beta6_mt: float = 1.0
    beta7_cf: float = 0.5

    # Logging and output
    output_dir: str = "outputs"
    log_every: int = 20
    enable_tsne: bool = True
    tsne_every_epochs: int = 5
    tsne_max_points: int = 512

    # Dummy data mode (for debugging without real data)
    use_dummy_data: bool = False
    dummy_steps_per_epoch: int = 60
    dummy_val_steps: int = 10

    # Ablation toggles
    use_scm: bool = True
    use_irm: bool = True
    use_counterfactual: bool = True
    use_multitask_uncertainty: bool = True
    use_reconstruction: bool = True

    # Single-task mode (Table 4: single-task vs multi-task comparison)
    # When set, only train on the specified task; others get zero loss weight.
    single_task: str | None = None  # None | "disease" | "fall" | "frailty"


# ============================================================================
# Helpers
# ============================================================================


@dataclass
class RuntimeState:
    amp_enabled: bool
    amp_dtype: torch.dtype
    use_scaler: bool
    scaler: _GradScaler | None
    amp_fallback_triggered: bool = False


@dataclass
class StopController:
    run_id: str
    control_dir: Path
    stop_requested: bool = False
    reason: str = ""

    @property
    def stop_file(self) -> Path:
        return self.control_dir / f"{self.run_id}.stop"

    def request_stop(self, reason: str) -> None:
        self.stop_requested = True
        self.reason = reason

    def poll_stop_file(self) -> bool:
        if self.stop_requested:
            return True
        if self.stop_file.exists():
            self.request_stop("stop_file")
            return True
        return False

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _str2bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}. Use auto|cuda|cpu.")


def _is_oom_exception(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _configure_cuda_backend(cfg: TrainConfig, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)


def _resolve_amp_enabled(cfg: TrainConfig, device: torch.device) -> bool:
    if cfg.use_amp is None:
        return device.type == "cuda"
    return bool(cfg.use_amp) and device.type == "cuda"


def _resolve_amp_dtype(cfg: TrainConfig, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if cfg.amp_dtype == "fp16":
        return torch.float16
    if cfg.amp_dtype == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[warn] bf16 requested but unsupported on this GPU. Falling back to fp16.")
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype: {cfg.amp_dtype}. Use fp16|bf16.")


def _make_grad_scaler(enabled: bool) -> _GradScaler:
    try:
        return _GradScaler(device="cuda", enabled=enabled)
    except TypeError:
        try:
            return _GradScaler("cuda", enabled=enabled)
        except TypeError:
            return _GradScaler(enabled=enabled)


def _autocast_context(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    if _USE_TORCH_AMP:
        return _autocast(device_type="cuda", dtype=dtype, enabled=True)
    return _autocast(enabled=True, dtype=dtype)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return

    def _to_byte_tensor(value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().to(dtype=torch.uint8)
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=torch.uint8)
        if isinstance(value, (list, tuple)):
            try:
                return torch.tensor(value, dtype=torch.uint8)
            except Exception:
                return None
        return None

    try:
        if "python" in state:
            random.setstate(state["python"])
    except Exception as exc:
        print(f"[warn] Failed to restore python RNG state: {type(exc).__name__}: {exc}")

    try:
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except Exception as exc:
        print(f"[warn] Failed to restore numpy RNG state: {type(exc).__name__}: {exc}")

    try:
        if "torch" in state:
            torch_state = _to_byte_tensor(state["torch"])
            if torch_state is not None:
                torch.set_rng_state(torch_state)
            else:
                print("[warn] Skipping torch RNG restore: unsupported checkpoint RNG format.")
    except Exception as exc:
        print(f"[warn] Failed to restore torch RNG state: {type(exc).__name__}: {exc}")

    if torch.cuda.is_available() and state.get("cuda") is not None:
        try:
            cuda_state_raw = state["cuda"]
            cuda_states: list[torch.Tensor] = []
            if isinstance(cuda_state_raw, (list, tuple)):
                for item in cuda_state_raw:
                    item_tensor = _to_byte_tensor(item)
                    if item_tensor is not None:
                        cuda_states.append(item_tensor)
            else:
                item_tensor = _to_byte_tensor(cuda_state_raw)
                if item_tensor is not None:
                    cuda_states.append(item_tensor)

            if cuda_states:
                torch.cuda.set_rng_state_all(cuda_states)
            else:
                print("[warn] Skipping CUDA RNG restore: unsupported checkpoint RNG format.")
        except Exception as exc:
            print(f"[warn] Failed to restore CUDA RNG state: {type(exc).__name__}: {exc}")


def _to_device_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v)
        for k, v in batch.items()
    }


def irm_anneal_weight(iter_idx: int, warmup_iters: int = 500, max_weight: float = 1.0) -> float:
    if warmup_iters <= 0:
        return max_weight
    if warmup_iters == 1:
        return max_weight
    ratio = min(max(float(iter_idx - 1), 0.0) / float(warmup_iters - 1), 1.0)
    return ratio * max_weight


def make_dummy_batch(cfg: TrainConfig, device: torch.device) -> dict[str, Tensor]:
    """Generate synthetic batch for debugging."""
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.input_dim, device=device)
    lengths = torch.randint(
        low=max(2, cfg.seq_len // 2),
        high=cfg.seq_len + 1,
        size=(cfg.batch_size,),
        device=device,
    )
    domain_id = torch.randint(0, cfg.num_domains, (cfg.batch_size,), device=device)
    base_noise = torch.randint(0, 3, (cfg.batch_size,), device=device)
    return {
        "x": x,
        "lengths": lengths,
        "domain_id": domain_id.long(),
        "label_disease": ((domain_id + base_noise) % cfg.num_disease_classes).long(),
        "label_fall": ((2 * domain_id + base_noise) % cfg.num_fall_classes).long(),
        "label_frailty": ((3 * domain_id + base_noise) % cfg.num_frailty_classes).long(),
    }


def compute_domain_irm_penalty(logits: Tensor, targets: Tensor, domain_id: Tensor) -> Tensor:
    penalties = []
    for d in domain_id.unique(sorted=True):
        mask = domain_id == d
        if int(mask.sum().item()) < 2:
            continue
        penalties.append(irm_penalty(logits[mask], targets[mask]))
    if len(penalties) == 0:
        return logits.new_tensor(0.0)
    return torch.stack(penalties).mean()


def _task_ce_losses(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    single_task: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    zero = batch["x"].new_tensor(0.0)

    if single_task is None or single_task == "disease":
        loss_disease = F.cross_entropy(out["disease_logits"], batch["label_disease"])
    else:
        loss_disease = zero

    if single_task is None or single_task == "fall":
        loss_fall = F.cross_entropy(out["fall_logits"], batch["label_fall"])
    else:
        loss_fall = zero

    if single_task is None or single_task == "frailty":
        # Frailty uses ordinal NLL (OrdinalHead outputs log-probs)
        loss_frailty = F.nll_loss(out["frailty_logits"], batch["label_frailty"])
    else:
        loss_frailty = zero

    return loss_disease, loss_fall, loss_frailty


# ============================================================================
# Train & Validation Steps
# ============================================================================


def _compute_losses(
    model: CausalGaitModel,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
    iter_idx: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    out = model(batch["x"], lengths=batch.get("lengths"), sample_domain=True)
    loss_disease, loss_fall, loss_frailty = _task_ce_losses(out, batch, single_task=cfg.single_task)

    if cfg.use_multitask_uncertainty and cfg.single_task is None:
        loss_mt, _ = model.heads.multi_task_uncertainty_loss(
            disease_loss=loss_disease,
            fall_loss=loss_fall,
            frailty_loss=loss_frailty,
        )
    else:
        loss_mt = batch["x"].new_tensor(0.0)

    loss_cls = (loss_disease + loss_fall + loss_frailty) / 3.0

    if cfg.use_irm:
        irm_d = compute_domain_irm_penalty(out["disease_logits"], batch["label_disease"], batch["domain_id"])
        irm_f = compute_domain_irm_penalty(out["fall_logits"], batch["label_fall"], batch["domain_id"])
        irm_r = compute_domain_irm_penalty(out["frailty_logits"], batch["label_frailty"], batch["domain_id"])
        loss_irm = (irm_d + irm_f + irm_r) / 3.0
    else:
        loss_irm = batch["x"].new_tensor(0.0)

    if cfg.use_reconstruction:
        loss_recon = F.mse_loss(out["recon"], out["recon_target"])
    else:
        loss_recon = batch["x"].new_tensor(0.0)

    loss_kl = out["domain_kl_loss"]
    loss_dag = out["dag_loss"]
    loss_hsic = out["hsic_loss"]

    if cfg.use_counterfactual:
        x_cf, _ = model.generate_counterfactuals(
            z_c=out["z_c"].detach(),
            z_d=out["z_d"].detach(),
            batch_domain_ids=batch["domain_id"],
        )
        out_cf = model(x_cf, lengths=batch.get("lengths"), sample_domain=True)
        cf_d, cf_f, cf_r = _task_ce_losses(out_cf, batch)
        loss_cf_cls = (cf_d + cf_f + cf_r) / 3.0
    else:
        loss_cf_cls = batch["x"].new_tensor(0.0)

    beta3_irm = irm_anneal_weight(iter_idx, cfg.irm_warmup_iters) if cfg.use_irm else 0.0
    total_loss = (
        loss_recon
        + cfg.beta1_kl * loss_kl
        + cfg.beta2_cls * loss_cls
        + beta3_irm * loss_irm
        + cfg.beta4_dag * loss_dag
        + cfg.beta5_hsic * loss_hsic
        + cfg.beta6_mt * loss_mt
        + cfg.beta7_cf * loss_cf_cls
    )
    scalars = {
        "total_loss": total_loss,
        "loss_recon": loss_recon,
        "loss_kl": loss_kl,
        "loss_cls": loss_cls,
        "loss_irm": loss_irm,
        "loss_dag": loss_dag,
        "loss_hsic": loss_hsic,
        "loss_mt": loss_mt,
        "loss_cf_cls": loss_cf_cls,
        "beta3_irm": total_loss.new_tensor(beta3_irm),
        "sigma_disease": out["sigma_disease"],
        "sigma_fall": out["sigma_fall"],
        "sigma_frailty": out["sigma_frailty"],
    }
    return total_loss, scalars


def train_step(
    model: CausalGaitModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
    iter_idx: int,
    runtime: RuntimeState,
) -> dict[str, float]:
    """Single training step with all loss components (paper Eq. 11)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_this_step = runtime.amp_enabled
    try:
        with _autocast_context(enabled=amp_this_step, dtype=runtime.amp_dtype):
            total_loss, scalars = _compute_losses(model=model, batch=batch, cfg=cfg, iter_idx=iter_idx)

        if runtime.use_scaler and runtime.scaler is not None and amp_this_step:
            runtime.scaler.scale(total_loss).backward()
            runtime.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            runtime.scaler.step(optimizer)
            runtime.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    except Exception as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        if not amp_this_step:
            raise
        runtime.amp_enabled = False
        runtime.amp_fallback_triggered = True
        print(f"[warn] AMP step failed ({type(exc).__name__}: {exc}). Falling back to FP32.")
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(enabled=False, dtype=runtime.amp_dtype):
            total_loss, scalars = _compute_losses(model=model, batch=batch, cfg=cfg, iter_idx=iter_idx)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return {
        "total": float(scalars["total_loss"].item()),
        "recon": float(scalars["loss_recon"].item()),
        "kl": float(scalars["loss_kl"].item()),
        "cls": float(scalars["loss_cls"].item()),
        "irm": float(scalars["loss_irm"].item()),
        "dag": float(scalars["loss_dag"].item()),
        "hsic": float(scalars["loss_hsic"].item()),
        "mt": float(scalars["loss_mt"].item()),
        "cf_cls": float(scalars["loss_cf_cls"].item()),
        "beta3": float(scalars["beta3_irm"].item()),
        "sigma_disease": float(scalars["sigma_disease"].item()),
        "sigma_fall": float(scalars["sigma_fall"].item()),
        "sigma_frailty": float(scalars["sigma_frailty"].item()),
        "amp_enabled": float(runtime.amp_enabled),
    }


@torch.no_grad()
def validation_step(
    model: CausalGaitModel,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
    runtime: RuntimeState,
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Validation step (no parameter update, no IRM)."""
    model.eval()
    with _autocast_context(enabled=runtime.amp_enabled, dtype=runtime.amp_dtype):
        out = model(batch["x"], lengths=batch.get("lengths"), sample_domain=False)
        loss_disease, loss_fall, loss_frailty = _task_ce_losses(out, batch, single_task=cfg.single_task)
        loss_mt, _ = model.heads.multi_task_uncertainty_loss(
            disease_loss=loss_disease,
            fall_loss=loss_fall,
            frailty_loss=loss_frailty,
        )
        loss_cls = (loss_disease + loss_fall + loss_frailty) / 3.0
        loss_recon = (
            F.mse_loss(out["recon"], out["recon_target"])
            if cfg.use_reconstruction
            else batch["x"].new_tensor(0.0)
        )
        loss_kl = out["domain_kl_loss"]
        loss_dag = out["dag_loss"]
        loss_hsic = out["hsic_loss"]
        total_loss = (
            loss_recon
            + cfg.beta1_kl * loss_kl
            + cfg.beta2_cls * loss_cls
            + cfg.beta4_dag * loss_dag
            + cfg.beta5_hsic * loss_hsic
            + cfg.beta6_mt * loss_mt
        )

    step_metrics = {
        "total": float(total_loss.item()),
        "recon": float(loss_recon.item()),
        "kl": float(loss_kl.item()),
        "cls": float(loss_cls.item()),
        "dag": float(loss_dag.item()),
        "hsic": float(loss_hsic.item()),
        "mt": float(loss_mt.item()),
    }
    y_true = {
        "disease": batch["label_disease"].detach().cpu(),
        "fall": batch["label_fall"].detach().cpu(),
        "frailty": batch["label_frailty"].detach().cpu(),
    }
    y_pred = {
        "disease": out["disease_logits"].detach().cpu(),
        "fall": out["fall_logits"].detach().cpu(),
        "frailty": out["frailty_logits"].detach().cpu(),
    }
    latent = {
        "z_c": out["z_c"].detach().cpu(),
        "z_d": out["z_d"].detach().cpu(),
        "domain_id": batch["domain_id"].detach().cpu(),
        "disease_label": batch["label_disease"].detach().cpu(),
    }
    return step_metrics, y_true, y_pred, latent


# ============================================================================
# Epoch-level routines
# ============================================================================

def _cat_batches(batches: list[Tensor]) -> Tensor:
    if len(batches) == 0:
        raise ValueError("Expected at least one tensor batch.")
    return torch.cat(batches, dim=0)


def _empty_task_metrics() -> dict[str, float]:
    return {
        "disease_acc": 0.0,
        "disease_macro_f1": 0.0,
        "disease_auc": 0.0,
        "fall_acc": 0.0,
        "fall_macro_f1": 0.0,
        "fall_auc": 0.0,
        "fall_ordinal_acc": 0.0,
        "frailty_acc": 0.0,
        "frailty_macro_f1": 0.0,
        "frailty_auc": 0.0,
        "frailty_ordinal_acc": 0.0,
        "frailty_mae": 0.0,
    }


def validate_epoch(
    model: CausalGaitModel,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
    device: torch.device,
    runtime: RuntimeState,
) -> tuple[float, dict[str, float], dict[str, Tensor]]:
    """Run full validation epoch."""
    model.eval()
    val_losses: list[float] = []
    y_true_store: dict[str, list[Tensor]] = {"disease": [], "fall": [], "frailty": []}
    y_pred_store: dict[str, list[Tensor]] = {"disease": [], "fall": [], "frailty": []}
    latent_store: dict[str, list[Tensor]] = {"z_c": [], "z_d": [], "domain_id": [], "disease_label": []}

    if val_loader is not None:
        for batch in val_loader:
            batch = _to_device_batch(batch, device)
            step_metrics, y_true, y_pred, latent = validation_step(model, batch, cfg, runtime)
            val_losses.append(step_metrics["total"])
            for key in y_true_store:
                y_true_store[key].append(y_true[key])
                y_pred_store[key].append(y_pred[key])
            for key in latent_store:
                latent_store[key].append(latent[key])
    else:
        # Dummy data mode
        for _ in range(cfg.dummy_val_steps):
            batch = make_dummy_batch(cfg, device)
            step_metrics, y_true, y_pred, latent = validation_step(model, batch, cfg, runtime)
            val_losses.append(step_metrics["total"])
            for key in y_true_store:
                y_true_store[key].append(y_true[key])
                y_pred_store[key].append(y_pred[key])
            for key in latent_store:
                latent_store[key].append(latent[key])

    if len(val_losses) == 0:
        print("[warn] Validation set is empty; returning default validation metrics.")
        latent_all = {
            "z_c": torch.empty((0, cfg.causal_dim), dtype=torch.float32),
            "z_d": torch.empty((0, cfg.domain_dim), dtype=torch.float32),
            "domain_id": torch.empty((0,), dtype=torch.long),
            "disease_label": torch.empty((0,), dtype=torch.long),
        }
        return float("inf"), _empty_task_metrics(), latent_all

    y_true_all = {k: _cat_batches(v) for k, v in y_true_store.items()}
    y_pred_all = {k: _cat_batches(v) for k, v in y_pred_store.items()}
    task_metrics = calculate_metrics(y_true=y_true_all, y_pred=y_pred_all)
    latent_all = {k: _cat_batches(v) for k, v in latent_store.items()}
    return float(np.mean(val_losses)), task_metrics, latent_all


def _checkpoint_state(
    cfg: TrainConfig,
    model: CausalGaitModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    runtime: RuntimeState,
    epoch: int,
    global_iter: int,
    val_loss: float,
    val_metrics: dict[str, float],
    best_val_loss: float,
    best_metrics: dict[str, float],
    patience_counter: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_iter": global_iter,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": runtime.scaler.state_dict() if runtime.use_scaler and runtime.scaler is not None else None,
        "rng_state": _capture_rng_state(),
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "best_metrics": best_metrics,
        "patience_counter": patience_counter,
        "amp_enabled": runtime.amp_enabled,
        "amp_dtype": cfg.amp_dtype,
        "reason": reason,
        "config": asdict(cfg),
    }


def save_checkpoint(state: dict[str, Any], is_best: bool, filename: str) -> None:
    ckpt_path = Path(filename)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = ckpt_path.with_name("best_model.pth")
        torch.save(state, best_path)


def save_train_state_json(
    output_dir: Path,
    run_id: str,
    epoch: int,
    global_iter: int,
    best_val_loss: float,
    patience_counter: int,
    status: str,
    reason: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "epoch": epoch,
        "global_iter": global_iter,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "status": status,
        "reason": reason,
        "timestamp": int(time.time()),
    }
    path = output_dir / "train_state.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    return path


def maybe_autosave_pipeline_state(
    cfg: TrainConfig,
    epoch: int,
    global_iter: int,
    last_autosave_ts: float,
) -> float:
    interval_sec = int(cfg.pipeline_autosave_sec)
    progress_file = cfg.pipeline_progress_file
    if interval_sec <= 0 or not progress_file:
        return last_autosave_ts

    now = time.time()
    if last_autosave_ts > 0.0 and (now - last_autosave_ts) < interval_sec:
        return last_autosave_ts

    progress_path = Path(progress_file)
    try:
        state: dict[str, Any] = {}
        if progress_path.exists():
            with open(progress_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                state = loaded

        heartbeat = state.get("train_heartbeat")
        if not isinstance(heartbeat, dict):
            heartbeat = {}
        heartbeat.update(
            {
                "epoch": int(epoch),
                "global_iter": int(global_iter),
                "output_dir": str(cfg.output_dir),
                "timestamp": int(now),
            }
        )

        state["train_heartbeat"] = heartbeat
        state["updated_at"] = int(now)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(
            f"[pipeline] autosave heartbeat written "
            f"(epoch={epoch}, iter={global_iter}, interval_sec={interval_sec})"
        )
        return now
    except Exception as exc:
        print(f"[warn] pipeline autosave skipped: {type(exc).__name__}: {exc}")
        return last_autosave_ts


def maybe_visualize_epoch(
    latent_all: dict[str, Tensor],
    cfg: TrainConfig,
    epoch: int,
) -> Path | None:
    if not cfg.enable_tsne:
        return None
    if cfg.tsne_every_epochs <= 0 or epoch % cfg.tsne_every_epochs != 0:
        return None

    z_c = latent_all["z_c"]
    z_d = latent_all["z_d"]
    domain_id = latent_all["domain_id"]
    disease_label = latent_all["disease_label"]

    if z_c.size(0) < 2 or z_d.size(0) < 2:
        print("[warn] t-SNE skipped: not enough validation points.")
        return None

    if cfg.tsne_max_points > 0 and z_c.size(0) > cfg.tsne_max_points:
        keep_idx = torch.randperm(z_c.size(0))[: cfg.tsne_max_points]
        z_c, z_d = z_c[keep_idx], z_d[keep_idx]
        domain_id, disease_label = domain_id[keep_idx], disease_label[keep_idx]

    tsne_path = Path(cfg.output_dir) / f"tsne_epoch_{epoch}.png"
    try:
        return visualize_latent_space(
            z_c=z_c, z_d=z_d, domain_ids=domain_id,
            disease_labels=disease_label, save_path=tsne_path,
        )
    except ImportError as exc:
        print(f"[warn] t-SNE skipped: {exc}")
        return None


# ============================================================================
# Main training loop
# ============================================================================

def _install_stop_signal_handlers(stop_ctrl: StopController) -> dict[int, Any]:
    previous: dict[int, Any] = {}

    def _handler(signum, _frame):
        stop_ctrl.request_stop(f"signal_{signum}")
        print(f"[control] signal {signum} received; stopping at next safe point.")

    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig_name):
            sig = getattr(signal, sig_name)
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _should_stop_now(stop_ctrl: StopController, cfg: TrainConfig, global_iter: int) -> bool:
    if stop_ctrl.stop_requested:
        return True
    if cfg.check_stop_every <= 0:
        return False
    if global_iter % cfg.check_stop_every != 0:
        return False
    return stop_ctrl.poll_stop_file()


def _prepare_stop_controller(cfg: TrainConfig) -> StopController:
    control_dir = Path(cfg.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)
    stop_ctrl = StopController(run_id=cfg.run_id, control_dir=control_dir)
    if stop_ctrl.stop_file.exists():
        print(f"[control] removing stale stop marker: {stop_ctrl.stop_file}")
        stop_ctrl.stop_file.unlink(missing_ok=True)
    return stop_ctrl


def _train_once(cfg: TrainConfig) -> dict[str, float]:
    """Single training attempt (without OOM auto-batch retry)."""
    set_seed(42)
    device = _resolve_device(cfg.device)
    _configure_cuda_backend(cfg, device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_ctrl = _prepare_stop_controller(cfg)
    previous_signal_handlers = _install_stop_signal_handlers(stop_ctrl)

    amp_enabled = _resolve_amp_enabled(cfg, device)
    amp_dtype = _resolve_amp_dtype(cfg, device)
    use_scaler = amp_enabled and device.type == "cuda" and amp_dtype == torch.float16
    runtime = RuntimeState(
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        use_scaler=use_scaler,
        scaler=_make_grad_scaler(enabled=use_scaler),
    )

    print(f"Device: {device}")
    print(f"Eval mode: {cfg.eval_mode}")
    print(f"Use dummy data: {cfg.use_dummy_data}")
    print(
        f"AMP: enabled={runtime.amp_enabled} dtype={cfg.amp_dtype} "
        f"scaler={runtime.use_scaler}"
    )

    # ----- Data -----
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None

    if not cfg.use_dummy_data:
        datasets = load_processed_datasets(
            processed_dir=cfg.processed_dir,
            dataset_names=cfg.dataset_names,
            input_dim=cfg.input_dim,
            n_disease_classes=cfg.num_disease_classes,
            n_fall_classes=cfg.num_fall_classes,
            n_frailty_classes=cfg.num_frailty_classes,
        )
        if len(datasets) == 0:
            print("[WARN] No datasets found. Falling back to dummy data.")
            cfg.use_dummy_data = True
        else:
            loaders = create_dataloaders(
                datasets=datasets,
                mode=cfg.eval_mode,
                target_domain=cfg.target_domain if cfg.eval_mode == "cross_domain" else None,
                fold=cfg.fold,
                n_folds=cfg.n_folds,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                persistent_workers=cfg.persistent_workers,
                prefetch_factor=cfg.prefetch_factor,
            )
            train_loader = loaders["train"]
            val_loader = loaders["val"]
            print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ----- Model -----
    model = CausalGaitModel(
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        n_layers=cfg.n_layers,
        scales=cfg.scales,
        bidirectional=cfg.bidirectional,
        backend=cfg.backend,
        dropout=cfg.dropout,
        causal_dim=cfg.causal_dim,
        domain_dim=cfg.domain_dim,
        num_disease_classes=cfg.num_disease_classes,
        num_fall_classes=cfg.num_fall_classes,
        num_frailty_classes=cfg.num_frailty_classes,
        decoder_n_layers=cfg.decoder_n_layers,
        decoder_layer_type=cfg.decoder_layer_type,
        use_scm=cfg.use_scm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    # ----- Training -----
    global_iter = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics: dict[str, float] = {}

    start_epoch = 1
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if runtime.scaler is not None and runtime.use_scaler and ckpt.get("scaler_state_dict") is not None:
            runtime.scaler.load_state_dict(ckpt["scaler_state_dict"])
        _restore_rng_state(ckpt.get("rng_state", {}))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_iter = int(ckpt.get("global_iter", 0))
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
        patience_counter = int(ckpt.get("patience_counter", 0))
        best_metrics = ckpt.get("best_metrics", ckpt.get("val_metrics", {})) or {}
        print(f"[resume] loaded {resume_path} (start_epoch={start_epoch}, global_iter={global_iter})")

    def _save_last_checkpoint(epoch: int, val_loss: float, val_metrics: dict[str, float], reason: str, is_best: bool) -> None:
        state = _checkpoint_state(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            runtime=runtime,
            epoch=epoch,
            global_iter=global_iter,
            val_loss=val_loss,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
            best_metrics=best_metrics,
            patience_counter=patience_counter,
            reason=reason,
        )
        save_checkpoint(state=state, is_best=is_best, filename=str(output_dir / "last_model.pth"))

    if start_epoch > cfg.num_epochs:
        print("[resume] checkpoint already beyond requested num_epochs. Skipping training loop.")
        save_train_state_json(
            output_dir=output_dir,
            run_id=cfg.run_id,
            epoch=start_epoch - 1,
            global_iter=global_iter,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            status="completed",
            reason="resume_noop",
        )
        _restore_signal_handlers(previous_signal_handlers)
        return best_metrics

    current_epoch = start_epoch - 1
    last_pipeline_autosave_ts = 0.0
    try:
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            current_epoch = epoch
            model.train()
            epoch_losses: list[float] = []
            stop_this_epoch = False

            if cfg.use_dummy_data:
                for _ in range(cfg.dummy_steps_per_epoch):
                    global_iter += 1
                    batch = make_dummy_batch(cfg, device)
                    metrics = train_step(model, optimizer, batch, cfg, global_iter, runtime)
                    epoch_losses.append(metrics["total"])

                    if global_iter == 1 or global_iter % cfg.log_every == 0:
                        _log_train_step(global_iter, metrics)

                    if cfg.save_every_steps > 0 and global_iter % cfg.save_every_steps == 0:
                        _save_last_checkpoint(epoch=epoch, val_loss=float("nan"), val_metrics={}, reason="step_save", is_best=False)

                    last_pipeline_autosave_ts = maybe_autosave_pipeline_state(
                        cfg=cfg,
                        epoch=epoch,
                        global_iter=global_iter,
                        last_autosave_ts=last_pipeline_autosave_ts,
                    )

                    if _should_stop_now(stop_ctrl, cfg, global_iter):
                        stop_this_epoch = True
                        break
            else:
                if train_loader is None:
                    raise RuntimeError("train_loader is None while use_dummy_data is False")
                for batch in train_loader:
                    global_iter += 1
                    batch = _to_device_batch(batch, device)
                    metrics = train_step(model, optimizer, batch, cfg, global_iter, runtime)
                    epoch_losses.append(metrics["total"])

                    if global_iter == 1 or global_iter % cfg.log_every == 0:
                        _log_train_step(global_iter, metrics)

                    if cfg.save_every_steps > 0 and global_iter % cfg.save_every_steps == 0:
                        _save_last_checkpoint(epoch=epoch, val_loss=float("nan"), val_metrics={}, reason="step_save", is_best=False)

                    last_pipeline_autosave_ts = maybe_autosave_pipeline_state(
                        cfg=cfg,
                        epoch=epoch,
                        global_iter=global_iter,
                        last_autosave_ts=last_pipeline_autosave_ts,
                    )

                    if _should_stop_now(stop_ctrl, cfg, global_iter):
                        stop_this_epoch = True
                        break

            if stop_this_epoch:
                print(f"[control] stop requested ({stop_ctrl.reason}). Saving and exiting.")
                _save_last_checkpoint(epoch=epoch, val_loss=float("nan"), val_metrics={}, reason=stop_ctrl.reason, is_best=False)
                save_train_state_json(
                    output_dir=output_dir,
                    run_id=cfg.run_id,
                    epoch=epoch,
                    global_iter=global_iter,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    status="stopped",
                    reason=stop_ctrl.reason,
                )
                return best_metrics

            scheduler.step()
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")

            val_loss, val_task_metrics, latent_all = validate_epoch(
                model=model,
                val_loader=val_loader,
                cfg=cfg,
                device=device,
                runtime=runtime,
            )
            is_best = val_loss < best_val_loss

            if is_best:
                best_val_loss = val_loss
                best_metrics = val_task_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            _save_last_checkpoint(
                epoch=epoch,
                val_loss=val_loss,
                val_metrics=val_task_metrics,
                reason="epoch_end",
                is_best=is_best,
            )
            last_pipeline_autosave_ts = maybe_autosave_pipeline_state(
                cfg=cfg,
                epoch=epoch,
                global_iter=global_iter,
                last_autosave_ts=last_pipeline_autosave_ts,
            )
            tsne_path = maybe_visualize_epoch(latent_all, cfg, epoch)
            _log_epoch(epoch, train_loss, val_loss, val_task_metrics, is_best, tsne_path)

            if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.early_stop_patience})")
                break

        model.eval()
        with torch.no_grad():
            if cfg.use_dummy_data:
                batch = make_dummy_batch(cfg, device)
            else:
                if val_loader is None:
                    raise RuntimeError("val_loader is None while use_dummy_data is False")
                batch = _to_device_batch(next(iter(val_loader)), device)
            out = model(batch["x"], lengths=batch.get("lengths"), sample_domain=False)

        print(
            f"Final shapes | recon={tuple(out['recon'].shape)} z_c={tuple(out['z_c'].shape)} "
            f"z_d={tuple(out['z_d'].shape)} disease={tuple(out['disease_logits'].shape)} "
            f"fall={tuple(out['fall_logits'].shape)} frailty={tuple(out['frailty_logits'].shape)}"
        )
        print(f"Best val metrics: {best_metrics}")
        save_train_state_json(
            output_dir=output_dir,
            run_id=cfg.run_id,
            epoch=current_epoch,
            global_iter=global_iter,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            status="completed",
            reason="finished",
        )
        return best_metrics
    except KeyboardInterrupt:
        stop_ctrl.request_stop("keyboard_interrupt")
        print("[control] KeyboardInterrupt received. Saving checkpoint and exiting gracefully.")
        _save_last_checkpoint(
            epoch=max(current_epoch, 1),
            val_loss=float("nan"),
            val_metrics={},
            reason=stop_ctrl.reason,
            is_best=False,
        )
        save_train_state_json(
            output_dir=output_dir,
            run_id=cfg.run_id,
            epoch=max(current_epoch, 1),
            global_iter=global_iter,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            status="stopped",
            reason=stop_ctrl.reason,
        )
        return best_metrics
    finally:
        _restore_signal_handlers(previous_signal_handlers)


def train(cfg: TrainConfig) -> dict[str, float]:
    """Train entry with optional OOM auto-batch retry."""
    current_batch = cfg.batch_size
    while True:
        attempt_cfg = copy.deepcopy(cfg)
        attempt_cfg.batch_size = current_batch
        try:
            return _train_once(attempt_cfg)
        except RuntimeError as exc:
            if not cfg.auto_batch or not _is_oom_exception(exc):
                raise
            next_batch = max(cfg.min_batch_size, current_batch // 2)
            if next_batch >= current_batch:
                raise
            print(
                f"[auto-batch] CUDA OOM at batch_size={current_batch}; "
                f"retrying with batch_size={next_batch}"
            )
            current_batch = next_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _log_train_step(global_iter: int, m: dict[str, float]) -> None:
    amp_flag = "amp=on" if m.get("amp_enabled", 0.0) > 0.5 else "amp=off"
    print(
        f"iter {global_iter:05d} | total={m['total']:.4f} recon={m['recon']:.4f} "
        f"cls={m['cls']:.4f} irm={m['irm']:.4f} dag={m['dag']:.4f} "
        f"hsic={m['hsic']:.4f} mt={m['mt']:.4f} cf={m['cf_cls']:.4f} "
        f"beta3={m['beta3']:.3f} {amp_flag}"
    )


def _log_epoch(
    epoch: int,
    train_loss: float,
    val_loss: float,
    vm: dict[str, float],
    is_best: bool,
    tsne_path: Path | None,
) -> None:
    best_str = " *BEST*" if is_best else ""
    tsne_str = f" tsne={tsne_path}" if tsne_path else ""
    print(
        f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
        f"disease_acc={vm.get('disease_acc', 0)*100:.1f}% "
        f"fall_acc={vm.get('fall_acc', 0)*100:.1f}% "
        f"frailty_mae={vm.get('frailty_mae', 0):.3f}"
        f"{best_str}{tsne_str}"
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train CausalGaitFM")
    parser.add_argument("--eval-mode", default="cross_domain", choices=["cross_domain", "in_domain", "loso"])
    parser.add_argument("--target-domain", default="daphnet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--use-dummy-data", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--pin-memory", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--persistent-workers", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)

    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--min-batch-size", type=int, default=4)

    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--control-dir", default="outputs/control")
    parser.add_argument("--check-stop-every", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--pipeline-progress-file", default=None)
    parser.add_argument("--pipeline-autosave-sec", type=int, default=0)

    parser.add_argument("--use-amp", type=_str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--allow-tf32", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--cudnn-benchmark", type=_str2bool, nargs="?", const=True, default=True)

    # Ablation flags
    parser.add_argument("--no-scm", action="store_true")
    parser.add_argument("--no-irm", action="store_true")
    parser.add_argument("--no-counterfactual", action="store_true")
    parser.add_argument("--no-multitask-uncertainty", action="store_true")
    parser.add_argument("--no-reconstruction", action="store_true")
    parser.add_argument("--single-task", default=None, choices=["disease", "fall", "frailty"],
                        help="Single-task mode: train only the specified task (Table 4 comparison)")

    args = parser.parse_args()
    cfg = TrainConfig(
        eval_mode=args.eval_mode,
        target_domain=args.target_domain,
        fold=args.fold,
        n_folds=args.n_folds,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        use_dummy_data=args.use_dummy_data,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        device=args.device,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
        auto_batch=args.auto_batch,
        min_batch_size=args.min_batch_size,
        resume_from=args.resume_from,
        run_id=args.run_id,
        control_dir=args.control_dir,
        check_stop_every=args.check_stop_every,
        save_every_steps=args.save_every_steps,
        pipeline_progress_file=args.pipeline_progress_file,
        pipeline_autosave_sec=args.pipeline_autosave_sec,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        allow_tf32=bool(args.allow_tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark),
        use_scm=not args.no_scm,
        use_irm=not args.no_irm,
        use_counterfactual=not args.no_counterfactual,
        use_multitask_uncertainty=not args.no_multitask_uncertainty,
        use_reconstruction=not args.no_reconstruction,
        single_task=args.single_task,
    )
    return cfg


if __name__ == "__main__":
    config = parse_args()
    train(config)
