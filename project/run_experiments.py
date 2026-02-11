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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
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
except ModuleNotFoundError:
    from train import TrainConfig, train, set_seed
    from baselines.erm import ERMModel
    from baselines.dann import DANNModel
    from baselines.coral import CORALModel
    from baselines.irm_baseline import IRMBaselineModel
    from baselines.groupdro import GroupDROModel
    from baselines.miro import MIROModel
    from baselines.domainbed import DomainBedModel
    from baselines.cnn_lstm import CNNLSTMModel
    from baselines.transformer import TransformerModel
    from baselines.beta_vae import BetaVAE
    from baselines.causal_vae import CausalVAE
    from baselines.st_gcn import STGCNModel
    from data.dataset import load_processed_datasets, create_dataloaders
    from utils.metrics import calculate_metrics


# ============================================================================
# Experiment: CausalGaitFM Cross-Domain (paper Table 2, last row)
# ============================================================================

def run_cross_domain(base_cfg: TrainConfig) -> dict[str, dict[str, float]]:
    """Train on 5 source domains, test on each held-out target domain."""
    all_domains = base_cfg.dataset_names
    results = {}

    for target in all_domains:
        print(f"\n{'='*60}")
        print(f"Cross-domain: target={target}")
        print(f"{'='*60}")

        cfg = deepcopy(base_cfg)
        cfg.eval_mode = "cross_domain"
        cfg.target_domain = target
        cfg.output_dir = f"{base_cfg.output_dir}/cross_domain/{target}"

        metrics = train(cfg)
        results[target] = metrics

    # Summary
    print(f"\n{'='*60}")
    print("Cross-domain results summary:")
    for target, m in results.items():
        acc = m.get("disease_acc", 0) * 100
        print(f"  {target}: disease_acc={acc:.1f}%")

    avg_acc = np.mean([m.get("disease_acc", 0) for m in results.values()]) * 100
    print(f"  Average: {avg_acc:.1f}%")
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


def run_ablation(base_cfg: TrainConfig) -> dict[str, dict[str, float]]:
    """Progressive ablation study (paper Section 2.5, Figure 2)."""
    results = {}

    for ablation_name, toggles in ABLATION_CONFIGS.items():
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

    # Summary
    print(f"\n{'='*60}")
    print("Ablation results summary:")
    for name, m in results.items():
        acc = m.get("disease_acc", 0) * 100
        f1 = m.get("disease_macro_f1", 0)
        print(f"  {name}: acc={acc:.1f}%, f1={f1:.4f}")

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
            batch = {k: v.to(device) for k, v in batch.items()}
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
                batch = {k: v.to(device) for k, v in batch.items()}
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


def run_baselines(base_cfg: TrainConfig) -> dict[str, dict[str, float]]:
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
        )
        train_loader, val_loader = loaders["train"], loaders["val"]
    else:
        # Use dummy loaders
        train_loader = val_loader = None

    if train_loader is None:
        print("[WARN] No data for baselines. Skipping.")
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

    for name, (cls, kwargs) in baseline_configs.items():
        print(f"\n{'='*60}")
        print(f"Baseline: {name} (target={target})")
        print(f"{'='*60}")
        metrics = _train_baseline(
            model_class=cls, model_kwargs=kwargs,
            train_loader=train_loader, val_loader=val_loader,
            num_epochs=base_cfg.num_epochs, lr=base_cfg.lr, device=device,
        )
        results[name] = metrics
        print(f"  -> {name}: accuracy={metrics['accuracy']*100:.1f}%")

    _save_results(results, Path(base_cfg.output_dir) / "baselines" / "results.json")
    return results


# ============================================================================
# Experiment: In-Domain 5-Fold CV (paper Table 3)
# ============================================================================

def run_in_domain(base_cfg: TrainConfig) -> dict[str, dict[str, float]]:
    """In-domain 5-fold cross-validation."""
    results = {}

    for fold in range(base_cfg.n_folds):
        print(f"\n{'='*60}")
        print(f"In-domain: fold {fold+1}/{base_cfg.n_folds}")
        print(f"{'='*60}")

        cfg = deepcopy(base_cfg)
        cfg.eval_mode = "in_domain"
        cfg.fold = fold
        cfg.output_dir = f"{base_cfg.output_dir}/in_domain/fold_{fold}"

        metrics = train(cfg)
        results[f"fold_{fold}"] = metrics

    # Average metrics
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

    _save_results(results, Path(base_cfg.output_dir) / "in_domain" / "results.json")
    return results


# ============================================================================
# Experiment: Single-Task vs Multi-Task (paper Table 4)
# ============================================================================

def run_single_task(base_cfg: TrainConfig) -> dict[str, dict[str, float]]:
    """Compare single-task training vs multi-task training (paper Table 4).

    Trains the model separately for each task (disease, fall, frailty)
    and compares with the full multi-task model.
    """
    results = {}

    # Multi-task (baseline comparison)
    print(f"\n{'='*60}")
    print("Single-task experiment: Multi-Task (all tasks)")
    print(f"{'='*60}")
    cfg_mt = deepcopy(base_cfg)
    cfg_mt.single_task = None
    cfg_mt.output_dir = f"{base_cfg.output_dir}/single_task/multi_task"
    results["multi_task"] = train(cfg_mt)

    # Single tasks
    for task in ("disease", "fall", "frailty"):
        print(f"\n{'='*60}")
        print(f"Single-task experiment: {task} only")
        print(f"{'='*60}")
        cfg_st = deepcopy(base_cfg)
        cfg_st.single_task = task
        cfg_st.output_dir = f"{base_cfg.output_dir}/single_task/{task}"
        results[task] = train(cfg_st)

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
