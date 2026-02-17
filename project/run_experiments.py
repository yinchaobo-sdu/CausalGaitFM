"""Experiment runner for ablation studies, baseline comparisons, and full evaluation.

Usage:
  # Run cross-domain evaluation for CausalGaitFM (all target domains)
  python -m project.run_experiments --experiment cross_domain

  # Run ablation study (paper Figure 2)
  python -m project.run_experiments --experiment ablation

  # Run all baseline comparisons (paper Table 2)
  python -m project.run_experiments --experiment baselines

  # Run in-domain 5-fold CV (paper Table 3)
  python -m project.run_experiments --experiment in_domain

  # Run with dummy data for debugging
  python -m project.run_experiments --experiment ablation --use-dummy-data
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from project.train import TrainConfig, train, set_seed
from project.baselines.erm import ERMModel
from project.baselines.dann import DANNModel
from project.baselines.coral import CORALModel
from project.baselines.irm_baseline import IRMBaselineModel
from project.baselines.groupdro import GroupDROModel
from project.baselines.miro import MIROModel
from project.baselines.domainbed import DomainBedModel
from project.baselines.cnn_lstm import CNNLSTMModel
from project.baselines.transformer import TransformerModel
from project.baselines.beta_vae import BetaVAE
from project.baselines.causal_vae import CausalVAE
from project.baselines.st_gcn import STGCNModel
from project.data.dataset import load_processed_datasets, create_dataloaders
from project.utils.metrics import calculate_metrics


def _stop_file(base_cfg: TrainConfig) -> Path:
    return Path(base_cfg.control_dir) / f"{base_cfg.run_id}.stop"


def _stop_requested(base_cfg: TrainConfig) -> bool:
    return _stop_file(base_cfg).exists()


def _str2bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


ProgressCallback = Callable[[dict[str, Any]], None]


def _emit_progress(
    progress_cb: ProgressCallback | None,
    stage_name: str,
    event: str,
    sub_done: int,
    sub_total: int,
    sub_name: str = "",
) -> None:
    if progress_cb is None:
        return
    progress_cb(
        {
            "stage_name": stage_name,
            "event": event,
            "sub_done": int(sub_done),
            "sub_total": int(sub_total),
            "sub_name": sub_name,
        }
    )


def _empty_task_metrics(skipped: bool = False) -> dict[str, float]:
    metrics = {
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
    if skipped:
        metrics["skipped"] = 1.0
    return metrics


def _domain_sample_count(processed_dir: str, domain_name: str) -> int:
    npz_path = Path(processed_dir) / f"{domain_name.lower()}.npz"
    if not npz_path.exists():
        return 0
    try:
        with np.load(npz_path) as data:
            if "X" in data:
                return int(data["X"].shape[0])
            if "y" in data:
                return int(data["y"].shape[0])
    except Exception as exc:
        print(f"[warn] failed to inspect {npz_path}: {type(exc).__name__}: {exc}")
    return 0


# ============================================================================
# Experiment: CausalGaitFM Cross-Domain (paper Table 2, last row)
# ============================================================================

def run_cross_domain(
    base_cfg: TrainConfig,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, dict[str, float]]:
    """Train on 5 source domains, test on each held-out target domain."""
    all_domains = base_cfg.dataset_names
    total = len(all_domains)
    results = {}
    domain_counts = {
        name: _domain_sample_count(base_cfg.processed_dir, name)
        for name in all_domains
    }
    _emit_progress(progress_cb, "cross_domain", "start", 0, total, "")

    for target in all_domains:
        if _stop_requested(base_cfg):
            print("[control] stop requested; terminating cross-domain loop.")
            break

        target_count = domain_counts.get(target, 0)
        source_count = sum(v for k, v in domain_counts.items() if k != target)
        if target_count <= 0:
            print(f"[warn] Cross-domain target={target} has 0 samples. Skipping this target.")
            results[target] = _empty_task_metrics(skipped=True)
            _emit_progress(progress_cb, "cross_domain", "step_done", len(results), total, target)
            continue
        if source_count <= 0:
            print(f"[warn] Cross-domain source data is empty when target={target}. Skipping this target.")
            results[target] = _empty_task_metrics(skipped=True)
            _emit_progress(progress_cb, "cross_domain", "step_done", len(results), total, target)
            continue

        print(f"\n{'='*60}")
        print(f"Cross-domain: target={target}")
        print(f"{'='*60}")

        cfg = deepcopy(base_cfg)
        cfg.eval_mode = "cross_domain"
        cfg.target_domain = target
        cfg.output_dir = f"{base_cfg.output_dir}/cross_domain/{target}"

        metrics = train(cfg)
        results[target] = metrics
        _emit_progress(progress_cb, "cross_domain", "step_done", len(results), total, target)
        if _stop_requested(base_cfg):
            print("[control] stop requested; cross-domain loop ended after current target.")
            break

    # Summary
    print(f"\n{'='*60}")
    print("Cross-domain results summary:")
    for target, m in results.items():
        if m.get("skipped", 0.0) > 0.5:
            print(f"  {target}: skipped (empty target/source domain)")
            continue
        acc = m.get("disease_acc", 0) * 100
        print(f"  {target}: disease_acc={acc:.1f}%")

    valid_metrics = [m for m in results.values() if m.get("skipped", 0.0) <= 0.5]
    if valid_metrics:
        avg_acc = np.mean([m.get("disease_acc", 0) for m in valid_metrics]) * 100
        print(f"  Average: {avg_acc:.1f}%")
    else:
        print("  No completed targets.")
    _emit_progress(progress_cb, "cross_domain", "done", len(results), total, "")
    _save_results(results, Path(base_cfg.output_dir) / "cross_domain" / "results.json")
    return results


# ============================================================================
# Experiment: Ablation Study (paper Figure 2, Section 2.5)
# ============================================================================

ABLATION_CONFIGS = {
    "vanilla_mamba": {
        "use_scm": False, "use_irm": False,
        "use_counterfactual": False, "use_multitask_uncertainty": False,
        "use_reconstruction": False,
    },
    "+SCM": {
        "use_scm": True, "use_irm": False,
        "use_counterfactual": False, "use_multitask_uncertainty": False,
        "use_reconstruction": True,
    },
    "+SCM+CF": {
        "use_scm": True, "use_irm": False,
        "use_counterfactual": True, "use_multitask_uncertainty": False,
        "use_reconstruction": True,
    },
    "+SCM+CF+IRM": {
        "use_scm": True, "use_irm": True,
        "use_counterfactual": True, "use_multitask_uncertainty": False,
        "use_reconstruction": True,
    },
    "full_model": {
        "use_scm": True, "use_irm": True,
        "use_counterfactual": True, "use_multitask_uncertainty": True,
        "use_reconstruction": True,
    },
}


def run_ablation(
    base_cfg: TrainConfig,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, dict[str, float]]:
    """Progressive ablation study (paper Section 2.5, Figure 2)."""
    results = {}
    total = len(ABLATION_CONFIGS)
    _emit_progress(progress_cb, "ablation", "start", 0, total, "")

    for ablation_name, toggles in ABLATION_CONFIGS.items():
        if _stop_requested(base_cfg):
            print("[control] stop requested; terminating ablation loop.")
            break

        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_name}")
        print(f"  Toggles: {toggles}")
        print(f"{'='*60}")

        cfg = deepcopy(base_cfg)
        cfg.output_dir = f"{base_cfg.output_dir}/ablation/{ablation_name}"
        for key, val in toggles.items():
            setattr(cfg, key, val)

        metrics = train(cfg)
        results[ablation_name] = metrics
        _emit_progress(progress_cb, "ablation", "step_done", len(results), total, ablation_name)
        if _stop_requested(base_cfg):
            print("[control] stop requested; ablation loop ended after current config.")
            break

    # Summary
    print(f"\n{'='*60}")
    print("Ablation results summary:")
    if not results:
        print("  No completed ablation configs.")
        _emit_progress(progress_cb, "ablation", "done", 0, total, "")
        _save_results(results, Path(base_cfg.output_dir) / "ablation" / "results.json")
        return results
    for name, m in results.items():
        acc = m.get("disease_acc", 0) * 100
        f1 = m.get("disease_macro_f1", 0)
        print(f"  {name}: acc={acc:.1f}%, f1={f1:.4f}")

    _emit_progress(progress_cb, "ablation", "done", len(results), total, "")
    _save_results(results, Path(base_cfg.output_dir) / "ablation" / "results.json")
    return results


# ============================================================================
# Experiment: Baseline Comparisons (paper Table 2)
# ============================================================================

def _train_baseline(
    model_class,
    model_kwargs: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    label_key: str = "label_disease",
) -> dict[str, float]:
    """Generic training loop for baseline models."""
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc = 0.0
    global_iter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            global_iter += 1
            batch = {
                k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            loss_dict = model.compute_loss(
                x=batch["x"],
                targets=batch[label_key],
                domain_ids=batch.get("domain_id", torch.zeros(batch["x"].size(0), dtype=torch.long, device=device)),
                lengths=batch.get("lengths"),
                iter_idx=global_iter,
            )
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                if hasattr(model, 'backbone'):
                    logits = model(batch["x"], lengths=batch.get("lengths"))
                    if isinstance(logits, tuple):
                        logits = logits[0]
                else:
                    logits = model(batch["x"], lengths=batch.get("lengths"))

                preds = logits.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_targets.append(batch[label_key].cpu())

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        acc = (preds_cat == targets_cat).float().mean().item()
        best_acc = max(best_acc, acc)

        if epoch % 10 == 0 or epoch == num_epochs:
            avg_loss = np.mean(epoch_losses)
            print(f"  epoch {epoch:03d} loss={avg_loss:.4f} val_acc={acc*100:.1f}%")

    return {"accuracy": best_acc}


def run_baselines(
    base_cfg: TrainConfig,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, dict[str, float]]:
    """Run all baseline models for comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    results = {}
    target = base_cfg.target_domain

    if not base_cfg.use_dummy_data:
        datasets = load_processed_datasets(
            processed_dir=base_cfg.processed_dir,
            dataset_names=base_cfg.dataset_names,
            input_dim=base_cfg.input_dim,
        )
        loaders = create_dataloaders(
            datasets=datasets, mode="cross_domain",
            target_domain=target, batch_size=base_cfg.batch_size,
            num_workers=base_cfg.num_workers,
            pin_memory=base_cfg.pin_memory,
            persistent_workers=base_cfg.persistent_workers,
            prefetch_factor=base_cfg.prefetch_factor,
        )
        train_loader, val_loader = loaders["train"], loaders["val"]
    else:
        # Use dummy loaders
        train_loader = val_loader = None

    if train_loader is None:
        print("[WARN] No data for baselines. Skipping.")
        _emit_progress(progress_cb, "baselines", "done", 0, 0, "")
        return results

    baseline_configs = {
        "ERM": (ERMModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "DANN": (DANNModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes, "num_domains": base_cfg.num_domains}),
        "CORAL": (CORALModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "IRM": (IRMBaselineModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "DomainBed": (DomainBedModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes, "num_domains": base_cfg.num_domains}),
        "GroupDRO": (GroupDROModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes, "num_domains": base_cfg.num_domains}),
        "MIRO": (MIROModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "CNN-LSTM": (CNNLSTMModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "Transformer": (TransformerModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "beta-VAE": (BetaVAE, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "CausalVAE": (CausalVAE, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
        "ST-GCN": (STGCNModel, {"input_dim": base_cfg.input_dim, "num_classes": base_cfg.num_disease_classes}),
    }
    total = len(baseline_configs)
    _emit_progress(progress_cb, "baselines", "start", 0, total, "")

    for name, (cls, kwargs) in baseline_configs.items():
        if _stop_requested(base_cfg):
            print("[control] stop requested; terminating baselines loop.")
            break

        print(f"\n{'='*60}")
        print(f"Baseline: {name} (target={target})")
        print(f"{'='*60}")
        metrics = _train_baseline(
            model_class=cls, model_kwargs=kwargs,
            train_loader=train_loader, val_loader=val_loader,
            num_epochs=base_cfg.num_epochs, lr=base_cfg.lr, device=device,
        )
        results[name] = metrics
        _emit_progress(progress_cb, "baselines", "step_done", len(results), total, name)
        print(f"  -> {name}: accuracy={metrics['accuracy']*100:.1f}%")
        if _stop_requested(base_cfg):
            print("[control] stop requested; baselines loop ended after current model.")
            break

    _emit_progress(progress_cb, "baselines", "done", len(results), total, "")
    _save_results(results, Path(base_cfg.output_dir) / "baselines" / "results.json")
    return results


# ============================================================================
# Experiment: In-Domain 5-Fold CV (paper Table 3)
# ============================================================================

def run_in_domain(
    base_cfg: TrainConfig,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, dict[str, float]]:
    """In-domain 5-fold cross-validation."""
    results = {}
    total = int(base_cfg.n_folds)
    _emit_progress(progress_cb, "in_domain", "start", 0, total, "")

    for fold in range(base_cfg.n_folds):
        if _stop_requested(base_cfg):
            print("[control] stop requested; terminating in-domain loop.")
            break

        print(f"\n{'='*60}")
        print(f"In-domain: fold {fold+1}/{base_cfg.n_folds}")
        print(f"{'='*60}")

        cfg = deepcopy(base_cfg)
        cfg.eval_mode = "in_domain"
        cfg.fold = fold
        cfg.output_dir = f"{base_cfg.output_dir}/in_domain/fold_{fold}"

        metrics = train(cfg)
        results[f"fold_{fold}"] = metrics
        _emit_progress(progress_cb, "in_domain", "step_done", len(results), total, f"fold_{fold}")
        if _stop_requested(base_cfg):
            print("[control] stop requested; in-domain loop ended after current fold.")
            break

    # Average metrics
    if not results:
        _emit_progress(progress_cb, "in_domain", "done", 0, total, "")
        _save_results(results, Path(base_cfg.output_dir) / "in_domain" / "results.json")
        return results
    all_metrics = list(results.values())
    avg = {}
    for key in all_metrics[0]:
        avg[key] = float(np.mean([m.get(key, 0) for m in all_metrics]))
    results["average"] = avg

    print(f"\n{'='*60}")
    print("In-domain CV results:")
    for fold_name, m in results.items():
        acc = m.get("disease_acc", 0) * 100
        print(f"  {fold_name}: disease_acc={acc:.1f}%")

    # Exclude "average" from progress done counter.
    _emit_progress(progress_cb, "in_domain", "done", len(results) - 1, total, "")
    _save_results(results, Path(base_cfg.output_dir) / "in_domain" / "results.json")
    return results


# ============================================================================
# Experiment: Single-Task vs Multi-Task (paper Table 4)
# ============================================================================

def run_single_task(
    base_cfg: TrainConfig,
    progress_cb: ProgressCallback | None = None,
    multi_task_resume_from: str | None = None,
) -> dict[str, dict[str, float]]:
    """Compare single-task training vs multi-task training (paper Table 4).

    Trains the model separately for each task (disease, fall, frailty)
    and compares with the full multi-task model.
    """
    results = {}
    total = 4
    done = 0
    _emit_progress(progress_cb, "single_task", "start", 0, total, "")

    # Multi-task (baseline comparison)
    print(f"\n{'='*60}")
    print("Single-task experiment: Multi-Task (all tasks)")
    print(f"{'='*60}")
    cfg_mt = deepcopy(base_cfg)
    cfg_mt.single_task = None
    cfg_mt.output_dir = f"{base_cfg.output_dir}/single_task/multi_task"
    if multi_task_resume_from:
        cfg_mt.resume_from = multi_task_resume_from
        print(f"[single_task] multi_task resume_from={multi_task_resume_from}")
    results["multi_task"] = train(cfg_mt)
    done += 1
    _emit_progress(progress_cb, "single_task", "step_done", done, total, "multi_task")
    if _stop_requested(base_cfg):
        print("[control] stop requested; single-task loop ended after multi-task run.")
        _emit_progress(progress_cb, "single_task", "done", done, total, "")
        _save_results(results, Path(base_cfg.output_dir) / "single_task" / "results.json")
        return results

    # Single tasks
    for task in ("disease", "fall", "frailty"):
        if _stop_requested(base_cfg):
            print("[control] stop requested; terminating single-task loop.")
            break

        print(f"\n{'='*60}")
        print(f"Single-task experiment: {task} only")
        print(f"{'='*60}")
        cfg_st = deepcopy(base_cfg)
        cfg_st.single_task = task
        cfg_st.output_dir = f"{base_cfg.output_dir}/single_task/{task}"
        results[task] = train(cfg_st)
        done += 1
        _emit_progress(progress_cb, "single_task", "step_done", done, total, task)
        if _stop_requested(base_cfg):
            print("[control] stop requested; single-task loop ended after current task.")
            break

    # Summary
    print(f"\n{'='*60}")
    print("Single-task vs Multi-task results:")
    for name, m in results.items():
        disease_acc = m.get("disease_acc", 0) * 100
        fall_acc = m.get("fall_acc", 0) * 100
        frailty_mae = m.get("frailty_mae", 0)
        frailty_oa = m.get("frailty_ordinal_acc", 0) * 100
        print(
            f"  {name}: disease_acc={disease_acc:.1f}% fall_acc={fall_acc:.1f}% "
            f"frailty_mae={frailty_mae:.3f} frailty_OA={frailty_oa:.1f}%"
        )

    _emit_progress(progress_cb, "single_task", "done", done, total, "")
    _save_results(results, Path(base_cfg.output_dir) / "single_task" / "results.json")
    return results


# ============================================================================
# Utils
# ============================================================================

def _save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: convert(vv) for kk, vv in v.items()}
        else:
            serializable[k] = convert(v)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run CausalGaitFM experiments")
    parser.add_argument(
        "--experiment", required=True,
        choices=["cross_domain", "ablation", "baselines", "in_domain", "single_task", "all"],
        help="Which experiment to run",
    )
    parser.add_argument("--target-domain", default="daphnet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--use-dummy-data", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Training device: auto|cuda|cpu")

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

    parser.add_argument("--use-amp", type=_str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--allow-tf32", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--cudnn-benchmark", type=_str2bool, nargs="?", const=True, default=True)
    args = parser.parse_args()

    cfg = TrainConfig(
        target_domain=args.target_domain,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        input_dim=args.input_dim,
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
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        allow_tf32=bool(args.allow_tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark),
    )

    experiments = {
        "cross_domain": run_cross_domain,
        "ablation": run_ablation,
        "baselines": run_baselines,
        "in_domain": run_in_domain,
        "single_task": run_single_task,
    }

    if args.experiment == "all":
        for name, fn in experiments.items():
            print(f"\n\n{'#'*70}")
            print(f"# Running experiment: {name}")
            print(f"{'#'*70}")
            fn(cfg)
    else:
        experiments[args.experiment](cfg)


if __name__ == "__main__":
    main()
