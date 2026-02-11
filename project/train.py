"""Training script for CausalGaitFM.

Supports three evaluation protocols (paper Section 2.1):
  - cross_domain : train on source domains, test on held-out target domain
  - in_domain    : k-fold cross-validation within each dataset
  - loso         : leave-one-subject-out cross-validation

Also supports training with synthetic dummy data for quick debugging.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from project.model import CausalGaitModel
    from project.utils.losses import irm_penalty
    from project.utils.metrics import calculate_metrics
    from project.utils.visualization import visualize_latent_space
    from project.data.dataset import (
        GaitDataset,
        MultiDomainGaitDataset,
        create_dataloaders,
        load_processed_datasets,
    )
except ModuleNotFoundError:
    from model import CausalGaitModel
    from utils.losses import irm_penalty
    from utils.metrics import calculate_metrics
    from utils.visualization import visualize_latent_space
    from data.dataset import (
        GaitDataset,
        MultiDomainGaitDataset,
        create_dataloaders,
        load_processed_datasets,
    )


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

    # Training hyperparams (aligned with paper Sec 4.8)
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 15
    num_workers: int = 0

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

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    device = batch["label_disease"].device
    zero = torch.tensor(0.0, device=device)

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

def train_step(
    model: CausalGaitModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
    iter_idx: int,
) -> dict[str, float]:
    """Single training step with all loss components (paper Eq. 11)."""
    model.train()
    out = model(batch["x"], lengths=batch.get("lengths"), sample_domain=True)
    loss_disease, loss_fall, loss_frailty = _task_ce_losses(out, batch, single_task=cfg.single_task)

    # Eq. (10): multi-task uncertainty weighting
    if cfg.use_multitask_uncertainty and cfg.single_task is None:
        loss_mt, loss_mt_parts = model.heads.multi_task_uncertainty_loss(
            disease_loss=loss_disease, fall_loss=loss_fall, frailty_loss=loss_frailty,
        )
    else:
        loss_mt = torch.tensor(0.0, device=batch["x"].device)
        loss_mt_parts = {"mt_disease": loss_mt, "mt_fall": loss_mt, "mt_frailty": loss_mt}

    loss_cls = (loss_disease + loss_fall + loss_frailty) / 3.0

    # IRM penalty (per-domain)
    if cfg.use_irm:
        irm_d = compute_domain_irm_penalty(out["disease_logits"], batch["label_disease"], batch["domain_id"])
        irm_f = compute_domain_irm_penalty(out["fall_logits"], batch["label_fall"], batch["domain_id"])
        irm_r = compute_domain_irm_penalty(out["frailty_logits"], batch["label_frailty"], batch["domain_id"])
        loss_irm = (irm_d + irm_f + irm_r) / 3.0
    else:
        loss_irm = torch.tensor(0.0, device=batch["x"].device)

    # Reconstruction loss
    if cfg.use_reconstruction:
        loss_recon = F.mse_loss(out["recon"], out["recon_target"])
    else:
        loss_recon = torch.tensor(0.0, device=batch["x"].device)

    loss_kl = out["domain_kl_loss"]
    loss_dag = out["dag_loss"]
    loss_hsic = out["hsic_loss"]

    # Counterfactual augmentation
    if cfg.use_counterfactual:
        x_cf, _ = model.generate_counterfactuals(
            z_c=out["z_c"].detach(), z_d=out["z_d"].detach(),
            batch_domain_ids=batch["domain_id"],
        )
        out_cf = model(x_cf, lengths=batch.get("lengths"), sample_domain=True)
        cf_d, cf_f, cf_r = _task_ce_losses(out_cf, batch)
        loss_cf_cls = (cf_d + cf_f + cf_r) / 3.0
    else:
        loss_cf_cls = torch.tensor(0.0, device=batch["x"].device)

    beta3_irm = irm_anneal_weight(iter_idx, cfg.irm_warmup_iters) if cfg.use_irm else 0.0

    # Eq. (11): total loss
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

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total": float(total_loss.item()),
        "recon": float(loss_recon.item()),
        "kl": float(loss_kl.item()),
        "cls": float(loss_cls.item()),
        "irm": float(loss_irm.item()),
        "dag": float(loss_dag.item()),
        "hsic": float(loss_hsic.item()),
        "mt": float(loss_mt.item()),
        "cf_cls": float(loss_cf_cls.item()),
        "beta3": float(beta3_irm),
        "sigma_disease": float(out["sigma_disease"].item()),
        "sigma_fall": float(out["sigma_fall"].item()),
        "sigma_frailty": float(out["sigma_frailty"].item()),
    }


@torch.no_grad()
def validation_step(
    model: CausalGaitModel,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """Validation step (no parameter update, no IRM)."""
    model.eval()
    out = model(batch["x"], lengths=batch.get("lengths"), sample_domain=False)
    loss_disease, loss_fall, loss_frailty = _task_ce_losses(out, batch, single_task=cfg.single_task)
    loss_mt, _ = model.heads.multi_task_uncertainty_loss(
        disease_loss=loss_disease, fall_loss=loss_fall, frailty_loss=loss_frailty,
    )
    loss_cls = (loss_disease + loss_fall + loss_frailty) / 3.0
    loss_recon = F.mse_loss(out["recon"], out["recon_target"]) if cfg.use_reconstruction else torch.tensor(0.0)
    loss_kl = out["domain_kl_loss"]
    loss_dag = out["dag_loss"]
    loss_hsic = out["hsic_loss"]

    total_loss = (
        loss_recon + cfg.beta1_kl * loss_kl + cfg.beta2_cls * loss_cls
        + cfg.beta4_dag * loss_dag + cfg.beta5_hsic * loss_hsic + cfg.beta6_mt * loss_mt
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


def validate_epoch(
    model: CausalGaitModel,
    val_loader: DataLoader | None,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, Tensor]]:
    """Run full validation epoch."""
    model.eval()
    val_losses: list[float] = []
    y_true_store: dict[str, list[Tensor]] = {"disease": [], "fall": [], "frailty": []}
    y_pred_store: dict[str, list[Tensor]] = {"disease": [], "fall": [], "frailty": []}
    latent_store: dict[str, list[Tensor]] = {"z_c": [], "z_d": [], "domain_id": [], "disease_label": []}

    if val_loader is not None:
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            step_metrics, y_true, y_pred, latent = validation_step(model, batch, cfg)
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
            step_metrics, y_true, y_pred, latent = validation_step(model, batch, cfg)
            val_losses.append(step_metrics["total"])
            for key in y_true_store:
                y_true_store[key].append(y_true[key])
                y_pred_store[key].append(y_pred[key])
            for key in latent_store:
                latent_store[key].append(latent[key])

    y_true_all = {k: _cat_batches(v) for k, v in y_true_store.items()}
    y_pred_all = {k: _cat_batches(v) for k, v in y_pred_store.items()}
    task_metrics = calculate_metrics(y_true=y_true_all, y_pred=y_pred_all)
    latent_all = {k: _cat_batches(v) for k, v in latent_store.items()}
    return float(np.mean(val_losses)), task_metrics, latent_all


def save_checkpoint(state: dict, is_best: bool, filename: str) -> None:
    ckpt_path = Path(filename)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = ckpt_path.with_name("best_model.pth")
        torch.save(state, best_path)


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

def train(cfg: TrainConfig) -> dict[str, float]:
    """Full training loop. Returns final validation metrics."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Eval mode: {cfg.eval_mode}")
    print(f"Use dummy data: {cfg.use_dummy_data}")

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

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        if cfg.use_dummy_data:
            # Dummy data loop
            for _ in range(cfg.dummy_steps_per_epoch):
                global_iter += 1
                batch = make_dummy_batch(cfg, device)
                metrics = train_step(model, optimizer, batch, cfg, global_iter)
                epoch_losses.append(metrics["total"])

                if global_iter == 1 or global_iter % cfg.log_every == 0:
                    _log_train_step(global_iter, metrics)
        else:
            # Real data loop
            for batch in train_loader:
                global_iter += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                metrics = train_step(model, optimizer, batch, cfg, global_iter)
                epoch_losses.append(metrics["total"])

                if global_iter == 1 or global_iter % cfg.log_every == 0:
                    _log_train_step(global_iter, metrics)

        scheduler.step()
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")

        # ----- Validation -----
        val_loss, val_task_metrics, latent_all = validate_epoch(
            model, val_loader, cfg, device,
        )
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            best_metrics = val_task_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            state={
                "epoch": epoch,
                "global_iter": global_iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_task_metrics,
                "config": cfg.__dict__,
            },
            is_best=is_best,
            filename=str(output_dir / "last_model.pth"),
        )

        tsne_path = maybe_visualize_epoch(latent_all, cfg, epoch)
        _log_epoch(epoch, train_loss, val_loss, val_task_metrics, is_best, tsne_path)

        # Early stopping
        if cfg.early_stop_patience > 0 and patience_counter >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (patience={cfg.early_stop_patience})")
            break

    # ----- Final eval -----
    model.eval()
    with torch.no_grad():
        if cfg.use_dummy_data:
            batch = make_dummy_batch(cfg, device)
        else:
            batch = next(iter(val_loader))
            batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"], lengths=batch.get("lengths"), sample_domain=False)
    print(
        f"Final shapes | recon={tuple(out['recon'].shape)} z_c={tuple(out['z_c'].shape)} "
        f"z_d={tuple(out['z_d'].shape)} disease={tuple(out['disease_logits'].shape)} "
        f"fall={tuple(out['fall_logits'].shape)} frailty={tuple(out['frailty_logits'].shape)}"
    )
    print(f"Best val metrics: {best_metrics}")
    return best_metrics


def _log_train_step(global_iter: int, m: dict[str, float]) -> None:
    print(
        f"iter {global_iter:05d} | total={m['total']:.4f} recon={m['recon']:.4f} "
        f"cls={m['cls']:.4f} irm={m['irm']:.4f} dag={m['dag']:.4f} "
        f"hsic={m['hsic']:.4f} mt={m['mt']:.4f} cf={m['cf_cls']:.4f} "
        f"beta3={m['beta3']:.3f}"
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
    parser.add_argument("--num-workers", type=int, default=0)
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
