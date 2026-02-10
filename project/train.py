from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from project.model import CausalGaitModel
    from project.utils.losses import irm_penalty
    from project.utils.metrics import calculate_metrics
    from project.utils.visualization import visualize_latent_space
except ModuleNotFoundError:
    from model import CausalGaitModel
    from utils.losses import irm_penalty
    from utils.metrics import calculate_metrics
    from utils.visualization import visualize_latent_space


@dataclass
class TrainConfig:
    input_dim: int = 32
    seq_len: int = 256
    batch_size: int = 16
    total_iters: int = 600
    steps_per_epoch: int = 60
    val_steps: int = 10
    log_every: int = 20

    num_domains: int = 4
    num_disease_classes: int = 4
    num_fall_classes: int = 3
    num_frailty_classes: int = 5

    lr: float = 1e-4
    irm_warmup_iters: int = 500

    beta1_kl: float = 1e-2
    beta2_cls: float = 1.0
    beta4_dag: float = 1e-2
    beta5_hsic: float = 1e-2
    beta6_mt: float = 1.0
    beta7_cf: float = 0.5

    output_dir: str = "outputs"
    enable_tsne: bool = True
    tsne_every_epochs: int = 1
    tsne_max_points: int = 512


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


def make_dummy_batch(
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, Tensor]:
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.input_dim, device=device)
    lengths = torch.randint(
        low=max(2, cfg.seq_len // 2),
        high=cfg.seq_len + 1,
        size=(cfg.batch_size,),
        device=device,
    )

    domain_id = torch.randint(0, cfg.num_domains, (cfg.batch_size,), device=device)
    base_noise = torch.randint(0, 3, (cfg.batch_size,), device=device)

    label_disease = (domain_id + base_noise) % cfg.num_disease_classes
    label_fall = (2 * domain_id + base_noise) % cfg.num_fall_classes
    label_frailty = (3 * domain_id + base_noise) % cfg.num_frailty_classes

    return {
        "x": x,
        "lengths": lengths,
        "domain_id": domain_id.long(),
        "label_disease": label_disease.long(),
        "label_fall": label_fall.long(),
        "label_frailty": label_frailty.long(),
    }


def compute_domain_irm_penalty(logits: Tensor, targets: Tensor, domain_id: Tensor) -> Tensor:
    penalties = []
    unique_domains = domain_id.unique(sorted=True)
    for d in unique_domains:
        mask = domain_id == d
        if int(mask.sum().item()) < 2:
            continue
        penalties.append(irm_penalty(logits[mask], targets[mask]))

    if len(penalties) == 0:
        return logits.new_tensor(0.0)
    return torch.stack(penalties).mean()


def _task_ce_losses(out: dict[str, Tensor], batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    loss_disease = F.cross_entropy(out["disease_logits"], batch["label_disease"])
    loss_fall = F.cross_entropy(out["fall_logits"], batch["label_fall"])
    loss_frailty = F.cross_entropy(out["frailty_logits"], batch["label_frailty"])
    return loss_disease, loss_fall, loss_frailty


def train_step(
    model: CausalGaitModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
    iter_idx: int,
) -> dict[str, float]:
    """
    Train step with real + counterfactual data.
    """
    out = model(batch["x"], lengths=batch["lengths"], sample_domain=True)
    loss_disease, loss_fall, loss_frailty = _task_ce_losses(out=out, batch=batch)

    # Eq. (10): dynamic multi-task weighting with learnable uncertainty sigma_t.
    loss_mt, loss_mt_parts = model.heads.multi_task_uncertainty_loss(
        disease_loss=loss_disease,
        fall_loss=loss_fall,
        frailty_loss=loss_frailty,
    )
    loss_cls = (loss_disease + loss_fall + loss_frailty) / 3.0

    irm_disease = compute_domain_irm_penalty(
        logits=out["disease_logits"],
        targets=batch["label_disease"],
        domain_id=batch["domain_id"],
    )
    irm_fall = compute_domain_irm_penalty(
        logits=out["fall_logits"],
        targets=batch["label_fall"],
        domain_id=batch["domain_id"],
    )
    irm_frailty = compute_domain_irm_penalty(
        logits=out["frailty_logits"],
        targets=batch["label_frailty"],
        domain_id=batch["domain_id"],
    )
    loss_irm = (irm_disease + irm_fall + irm_frailty) / 3.0

    # Decoder-based reconstruction from z=[z_c, z_d] back to x.
    loss_recon = F.mse_loss(out["recon"], out["recon_target"])
    loss_kl = out["domain_kl_loss"]
    loss_dag = out["dag_loss"]
    loss_hsic = out["hsic_loss"]

    # Counterfactual augmentation (domain intervention):
    # keep z_c fixed, shuffle z_d by domain, decode x_cf, reuse original labels.
    x_cf, _ = model.generate_counterfactuals(
        z_c=out["z_c"].detach(),
        z_d=out["z_d"].detach(),
        batch_domain_ids=batch["domain_id"],
    )
    out_cf = model(x_cf, lengths=batch["lengths"], sample_domain=True)
    cf_disease, cf_fall, cf_frailty = _task_ce_losses(out=out_cf, batch=batch)
    loss_cf_cls = (cf_disease + cf_fall + cf_frailty) / 3.0

    beta3_irm = irm_anneal_weight(
        iter_idx=iter_idx,
        warmup_iters=cfg.irm_warmup_iters,
        max_weight=1.0,
    )

    # Eq. (11) + counterfactual regularizer:
    # L_total = L_recon + b1*L_KL + b2*L_cls + b3*L_IRM + b4*L_DAG + b5*L_HSIC + b6*L_MT + b7*L_CF
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
        "mt_disease": float(loss_mt_parts["mt_disease"].item()),
        "mt_fall": float(loss_mt_parts["mt_fall"].item()),
        "mt_frailty": float(loss_mt_parts["mt_frailty"].item()),
    }


@torch.no_grad()
def validation_step(
    model: CausalGaitModel,
    batch: dict[str, Tensor],
    cfg: TrainConfig,
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    """
    Validation step (no parameter update). IRM is skipped to keep validation in no-grad mode.
    """
    out = model(batch["x"], lengths=batch["lengths"], sample_domain=False)
    loss_disease, loss_fall, loss_frailty = _task_ce_losses(out=out, batch=batch)
    loss_mt, _ = model.heads.multi_task_uncertainty_loss(
        disease_loss=loss_disease,
        fall_loss=loss_fall,
        frailty_loss=loss_frailty,
    )
    loss_cls = (loss_disease + loss_fall + loss_frailty) / 3.0
    loss_recon = F.mse_loss(out["recon"], out["recon_target"])
    loss_kl = out["domain_kl_loss"]
    loss_dag = out["dag_loss"]
    loss_hsic = out["hsic_loss"]

    x_cf, _ = model.generate_counterfactuals(
        z_c=out["z_c"],
        z_d=out["z_d"],
        batch_domain_ids=batch["domain_id"],
    )
    out_cf = model(x_cf, lengths=batch["lengths"], sample_domain=False)
    cf_disease, cf_fall, cf_frailty = _task_ce_losses(out=out_cf, batch=batch)
    loss_cf_cls = (cf_disease + cf_fall + cf_frailty) / 3.0

    total_loss = (
        loss_recon
        + cfg.beta1_kl * loss_kl
        + cfg.beta2_cls * loss_cls
        + cfg.beta4_dag * loss_dag
        + cfg.beta5_hsic * loss_hsic
        + cfg.beta6_mt * loss_mt
        + cfg.beta7_cf * loss_cf_cls
    )

    step_metrics = {
        "total": float(total_loss.item()),
        "recon": float(loss_recon.item()),
        "kl": float(loss_kl.item()),
        "cls": float(loss_cls.item()),
        "dag": float(loss_dag.item()),
        "hsic": float(loss_hsic.item()),
        "mt": float(loss_mt.item()),
        "cf_cls": float(loss_cf_cls.item()),
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


def save_checkpoint(state: dict[str, object], is_best: bool, filename: str) -> None:
    ckpt_path = Path(filename)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = ckpt_path.with_name("best_model.pth")
        torch.save(state, best_path)


def _cat_batches(batches: list[Tensor]) -> Tensor:
    if len(batches) == 0:
        raise ValueError("Expected at least one tensor batch.")
    return torch.cat(batches, dim=0)


def validate_epoch(
    model: CausalGaitModel,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, dict[str, float], dict[str, Tensor]]:
    if cfg.val_steps <= 0:
        raise ValueError("`val_steps` must be positive.")

    model.eval()
    val_losses: list[float] = []
    y_true_store: dict[str, list[Tensor]] = {"disease": [], "fall": [], "frailty": []}
    y_pred_store: dict[str, list[Tensor]] = {"disease": [], "fall": [], "frailty": []}
    latent_store: dict[str, list[Tensor]] = {
        "z_c": [],
        "z_d": [],
        "domain_id": [],
        "disease_label": [],
    }

    for _ in range(cfg.val_steps):
        batch = make_dummy_batch(cfg=cfg, device=device)
        step_metrics, y_true, y_pred, latent = validation_step(model=model, batch=batch, cfg=cfg)
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


def maybe_visualize_epoch(
    latent_all: dict[str, Tensor],
    cfg: TrainConfig,
    epoch: int,
) -> Path | None:
    if not cfg.enable_tsne:
        return None
    if cfg.tsne_every_epochs <= 0:
        return None
    if epoch % cfg.tsne_every_epochs != 0:
        return None

    z_c = latent_all["z_c"]
    z_d = latent_all["z_d"]
    domain_id = latent_all["domain_id"]
    disease_label = latent_all["disease_label"]

    if cfg.tsne_max_points > 0 and z_c.size(0) > cfg.tsne_max_points:
        keep_idx = torch.randperm(z_c.size(0))[: cfg.tsne_max_points]
        z_c = z_c[keep_idx]
        z_d = z_d[keep_idx]
        domain_id = domain_id[keep_idx]
        disease_label = disease_label[keep_idx]

    tsne_path = Path(cfg.output_dir) / f"tsne_epoch_{epoch}.png"
    try:
        return visualize_latent_space(
            z_c=z_c,
            z_d=z_d,
            domain_ids=domain_id,
            disease_labels=disease_label,
            save_path=tsne_path,
        )
    except ImportError as exc:
        print(f"[warn] t-SNE skipped: {exc}")
        return None


def train(cfg: TrainConfig) -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CausalGaitModel(
        input_dim=cfg.input_dim,
        seq_len=cfg.seq_len,
        d_model=128,
        d_state=16,
        n_layers=4,
        scales=(1, 2, 4),
        bidirectional=True,
        backend="auto",
        causal_dim=32,
        domain_dim=16,
        num_disease_classes=cfg.num_disease_classes,
        num_fall_classes=cfg.num_fall_classes,
        num_frailty_classes=cfg.num_frailty_classes,
        decoder_n_layers=2,
        decoder_layer_type="mamba",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.steps_per_epoch <= 0:
        raise ValueError("`steps_per_epoch` must be positive.")
    num_epochs = max(1, math.ceil(cfg.total_iters / cfg.steps_per_epoch))

    global_iter = 0
    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses: list[float] = []

        for _ in range(cfg.steps_per_epoch):
            if global_iter >= cfg.total_iters:
                break
            global_iter += 1

            batch = make_dummy_batch(cfg=cfg, device=device)
            metrics = train_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                cfg=cfg,
                iter_idx=global_iter,
            )
            train_losses.append(metrics["total"])

            if global_iter == 1 or global_iter % cfg.log_every == 0 or global_iter == cfg.total_iters:
                print(
                    "iter "
                    f"{global_iter:04d}/{cfg.total_iters:04d} | "
                    f"total={metrics['total']:.5f} | "
                    f"recon={metrics['recon']:.5f} kl={metrics['kl']:.5f} "
                    f"cls={metrics['cls']:.5f} irm={metrics['irm']:.5f} "
                    f"dag={metrics['dag']:.5f} hsic={metrics['hsic']:.5f} "
                    f"mt={metrics['mt']:.5f} cf_cls={metrics['cf_cls']:.5f} | "
                    f"beta3={metrics['beta3']:.3f} | "
                    f"sigma(d/f/r)=({metrics['sigma_disease']:.4f}/"
                    f"{metrics['sigma_fall']:.4f}/"
                    f"{metrics['sigma_frailty']:.4f}) | "
                    f"mt_parts(d/f/r)=({metrics['mt_disease']:.5f}/"
                    f"{metrics['mt_fall']:.5f}/"
                    f"{metrics['mt_frailty']:.5f})"
                )

        if len(train_losses) == 0:
            break
        train_loss = float(np.mean(train_losses))

        val_loss, val_task_metrics, latent_all = validate_epoch(model=model, cfg=cfg, device=device)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

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

        tsne_path = maybe_visualize_epoch(latent_all=latent_all, cfg=cfg, epoch=epoch)
        save_msg = " | Saving Best Model..." if is_best else ""
        tsne_msg = f" | t-SNE={tsne_path}" if tsne_path is not None else ""
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc (Disease): {val_task_metrics['disease_acc'] * 100.0:.2f}% | "
            f"Val Macro-F1 (Disease): {val_task_metrics['disease_macro_f1']:.4f} | "
            f"Val Acc (Fall): {val_task_metrics['fall_acc'] * 100.0:.2f}% | "
            f"Val MAE (Frailty): {val_task_metrics['frailty_mae']:.4f}"
            f"{save_msg}{tsne_msg}"
        )

        if global_iter >= cfg.total_iters:
            break

    model.eval()
    with torch.no_grad():
        batch = make_dummy_batch(cfg=cfg, device=device)
        out = model(batch["x"], lengths=batch["lengths"], sample_domain=False)
    print(
        "eval shapes | "
        f"recon={tuple(out['recon'].shape)} "
        f"target={tuple(out['recon_target'].shape)} "
        f"z_c={tuple(out['z_c'].shape)} "
        f"z_d={tuple(out['z_d'].shape)} "
        f"disease={tuple(out['disease_logits'].shape)} "
        f"fall={tuple(out['fall_logits'].shape)} "
        f"frailty={tuple(out['frailty_logits'].shape)}"
    )


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
