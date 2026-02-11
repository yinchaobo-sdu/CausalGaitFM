"""Unified PyTorch Dataset and DataLoader for multi-domain gait analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def _remap_labels_contiguous(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Remap arbitrary integer labels to contiguous 0..K-1."""
    unique = np.unique(labels)
    mapping = {int(old): new for new, old in enumerate(unique)}
    return np.array([mapping[int(l)] for l in labels], dtype=np.int64), mapping


# Per-dataset heuristic clinical mappings.
# These group activity labels by clinical relevance rather than using naive
# modulo arithmetic.  When real clinical annotations are available (e.g. from
# a physician panel), they should replace these heuristics.
#
# Mapping structure:
#   activity_label (contiguous) -> {disease_class, fall_risk_level, frailty_level}

_CLINICAL_RULES: dict[str, dict[str, list[int]]] = {
    # Daphnet: 2 activities (freeze / no-freeze) mapped to 3 clinical axes
    #   0 = normal walking, 1 = freeze-of-gait episode
    "daphnet": {
        # disease: 0=healthy gait, 1=Parkinson (FoG)
        "disease_bins": [0, 1],
        # fall risk: 0=low, 1=medium (normal walk can trip), 2=high (freeze)
        "fall_bins": [0, 2],
        # frailty: 0=robust, 1,2,3,4 = pre-frail -> frail
        "frailty_bins": [0, 3],
    },
    # UCI-HAR: 6 activities (0=walk,1=walk_up,2=walk_down,3=sit,4=stand,5=lying)
    "ucihar": {
        "disease_bins": [0, 0, 0, 1, 1, 2],
        "fall_bins": [0, 1, 1, 0, 0, 0],
        "frailty_bins": [0, 0, 0, 2, 1, 3],
    },
    # PAMAP2: typically 12+ activities, first 8 most common
    "pamap2": {
        "disease_bins": [0, 0, 0, 1, 1, 2, 2, 3],
        "fall_bins": [0, 1, 1, 0, 0, 1, 2, 0],
        "frailty_bins": [0, 0, 0, 1, 1, 2, 2, 3],
    },
    # mHealth: 12 activities
    "mhealth": {
        "disease_bins": [0, 0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2],
        "fall_bins": [0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 0, 1],
        "frailty_bins": [0, 0, 0, 1, 1, 2, 2, 3, 4, 0, 1, 2],
    },
    # WISDM: 6 activities (walk, jog, stairs, sit, stand, lying)
    "wisdm": {
        "disease_bins": [0, 0, 0, 1, 1, 2],
        "fall_bins": [0, 0, 1, 0, 0, 0],
        "frailty_bins": [0, 0, 0, 2, 1, 3],
    },
    # Opportunity: many activities; map first 8 common ones
    "opportunity": {
        "disease_bins": [0, 0, 1, 1, 2, 2, 3, 3],
        "fall_bins": [0, 0, 0, 1, 0, 1, 1, 2],
        "frailty_bins": [0, 0, 1, 1, 2, 2, 3, 4],
    },
}


def _map_to_clinical_tasks(
    y: np.ndarray,
    n_disease: int,
    n_fall: int,
    n_frailty: int,
    domain_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Map dataset activity labels to the three clinical tasks.

    Uses dataset-specific heuristic mappings that group activities by clinical
    relevance.  Falls back to modulo mapping when no domain-specific rules are
    available.

    Args:
        y: Contiguous activity labels (0-indexed, from ``_remap_labels_contiguous``).
        n_disease: Number of disease classes.
        n_fall: Number of fall-risk levels.
        n_frailty: Number of frailty levels.
        domain_name: Optional dataset name for domain-specific mapping lookup.

    Returns:
        dict with keys ``disease``, ``fall``, ``frailty`` each containing [N] int64 arrays.
    """
    rules = _CLINICAL_RULES.get(domain_name, None) if domain_name else None

    if rules is not None:
        disease_bins = np.array(rules["disease_bins"], dtype=np.int64)
        fall_bins = np.array(rules["fall_bins"], dtype=np.int64)
        frailty_bins = np.array(rules["frailty_bins"], dtype=np.int64)

        # Safe lookup: clip y to the mapping table length
        disease = disease_bins[np.clip(y, 0, len(disease_bins) - 1)] % n_disease
        fall = fall_bins[np.clip(y, 0, len(fall_bins) - 1)] % n_fall
        frailty = frailty_bins[np.clip(y, 0, len(frailty_bins) - 1)] % n_frailty
    else:
        # Fallback: deterministic mapping from activity labels
        disease = (y % n_disease).astype(np.int64)
        fall = (y % n_fall).astype(np.int64)
        frailty = (y % n_frailty).astype(np.int64)

    return {
        "disease": disease.astype(np.int64),
        "fall": fall.astype(np.int64),
        "frailty": frailty.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Single-domain Dataset
# ---------------------------------------------------------------------------

class GaitDataset(Dataset):
    """Dataset for a single gait domain.

    Parameters
    ----------
    X : np.ndarray [N, T, D]
    y : np.ndarray [N]
    subjects : np.ndarray [N]
    domain_id : int
    n_disease_classes : int
    n_fall_classes : int
    n_frailty_classes : int
    input_dim : int or None
        If provided, project to a common channel dimension via zero-padding / truncation.
    domain_name : str or None
        Dataset name (e.g. 'daphnet', 'ucihar') for clinical label mapping lookup.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        domain_id: int,
        n_disease_classes: int = 4,
        n_fall_classes: int = 3,
        n_frailty_classes: int = 5,
        input_dim: int | None = None,
        domain_name: str | None = None,
    ) -> None:
        super().__init__()
        assert X.ndim == 3, f"X must be [N,T,D], got {X.shape}"

        # Align channel dimension
        if input_dim is not None and X.shape[2] != input_dim:
            if X.shape[2] < input_dim:
                pad = np.zeros((X.shape[0], X.shape[1], input_dim - X.shape[2]), dtype=X.dtype)
                X = np.concatenate([X, pad], axis=2)
            else:
                X = X[:, :, :input_dim]

        self.X = torch.from_numpy(X).float()
        self.subjects = torch.from_numpy(subjects).long()
        self.domain_id = domain_id
        self.domain_name = domain_name

        # Remap raw labels
        y_remapped, _ = _remap_labels_contiguous(y)
        clinical = _map_to_clinical_tasks(
            y_remapped, n_disease_classes, n_fall_classes, n_frailty_classes,
            domain_name=domain_name,
        )

        self.label_disease = torch.from_numpy(clinical["disease"]).long()
        self.label_fall = torch.from_numpy(clinical["fall"]).long()
        self.label_frailty = torch.from_numpy(clinical["frailty"]).long()
        self.domain_ids = torch.full((len(X),), domain_id, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "x": self.X[idx],
            "label_disease": self.label_disease[idx],
            "label_fall": self.label_fall[idx],
            "label_frailty": self.label_frailty[idx],
            "domain_id": self.domain_ids[idx],
            "subject_id": self.subjects[idx],
        }


# ---------------------------------------------------------------------------
# Multi-domain Dataset (concatenation of GaitDatasets)
# ---------------------------------------------------------------------------

class MultiDomainGaitDataset(Dataset):
    """Concatenation of multiple GaitDatasets with consistent interface."""

    def __init__(self, datasets: Sequence[GaitDataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self._cumulative_sizes: list[int] = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self._cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        ds_idx = 0
        for i, cum in enumerate(self._cumulative_sizes):
            if idx < cum:
                ds_idx = i
                break
        offset = self._cumulative_sizes[ds_idx - 1] if ds_idx > 0 else 0
        return self.datasets[ds_idx][idx - offset]

    @property
    def domain_ids(self) -> list[int]:
        return [ds.domain_id for ds in self.datasets]


# ---------------------------------------------------------------------------
# Splitting utilities
# ---------------------------------------------------------------------------

def _kfold_indices(
    n: int,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, val_idx) for each fold."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, val_idx))
    return folds


def _loso_indices(
    subjects: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-subject-out indices."""
    unique_subj = np.unique(subjects)
    folds = []
    for subj in unique_subj:
        val_mask = subjects == subj
        train_mask = ~val_mask
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return folds


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def load_processed_datasets(
    processed_dir: str | Path,
    dataset_names: Sequence[str] | None = None,
    input_dim: int = 32,
    n_disease_classes: int = 4,
    n_fall_classes: int = 3,
    n_frailty_classes: int = 5,
) -> dict[str, GaitDataset]:
    """Load preprocessed .npz files and return GaitDatasets keyed by name."""
    processed_dir = Path(processed_dir)
    all_names = ["daphnet", "ucihar", "pamap2", "mhealth", "wisdm", "opportunity"]
    if dataset_names is not None:
        all_names = [n.lower() for n in dataset_names]

    datasets = {}
    for domain_id, name in enumerate(all_names):
        npz_path = processed_dir / f"{name}.npz"
        if not npz_path.exists():
            print(f"[warn] {npz_path} not found, skipping {name}")
            continue
        data = np.load(npz_path)
        ds = GaitDataset(
            X=data["X"],
            y=data["y"],
            subjects=data["subjects"],
            domain_id=domain_id,
            n_disease_classes=n_disease_classes,
            n_fall_classes=n_fall_classes,
            n_frailty_classes=n_frailty_classes,
            input_dim=input_dim,
            domain_name=name,
        )
        datasets[name] = ds
        print(f"[load] {name}: {len(ds)} samples, input shape {ds.X.shape[1:]}")
    return datasets


def create_dataloaders(
    datasets: dict[str, GaitDataset],
    mode: str = "cross_domain",
    target_domain: str | None = None,
    fold: int = 0,
    n_folds: int = 5,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create train/val DataLoaders according to the evaluation protocol.

    Parameters
    ----------
    mode : str
        'cross_domain' : Train on all domains except target_domain, test on target_domain.
        'in_domain'    : K-fold CV within each domain (merged).
        'loso'         : Leave-one-subject-out within each domain (merged).
    target_domain : str
        Required for 'cross_domain' mode.
    fold : int
        Fold index for 'in_domain' and 'loso' modes.
    """
    if mode == "cross_domain":
        if target_domain is None:
            raise ValueError("target_domain is required for cross_domain mode")
        target_domain = target_domain.lower()
        if target_domain not in datasets:
            raise ValueError(f"target_domain '{target_domain}' not in datasets: {list(datasets)}")

        source_datasets = [ds for name, ds in datasets.items() if name != target_domain]
        target_ds = datasets[target_domain]

        train_ds = MultiDomainGaitDataset(source_datasets)
        val_ds = target_ds

    elif mode == "in_domain":
        merged = MultiDomainGaitDataset(list(datasets.values()))
        folds = _kfold_indices(len(merged), n_folds=n_folds, seed=seed)
        if fold >= len(folds):
            raise ValueError(f"fold {fold} out of range for {n_folds} folds")
        train_idx, val_idx = folds[fold]
        train_ds = Subset(merged, train_idx.tolist())
        val_ds = Subset(merged, val_idx.tolist())

    elif mode == "loso":
        merged = MultiDomainGaitDataset(list(datasets.values()))
        # Collect all subject ids
        all_subjects = []
        for ds in datasets.values():
            all_subjects.append(ds.subjects.numpy())
        subjects_arr = np.concatenate(all_subjects)
        folds = _loso_indices(subjects_arr)
        if fold >= len(folds):
            raise ValueError(f"fold {fold} out of range for {len(folds)} subjects")
        train_idx, val_idx = folds[fold]
        train_ds = Subset(merged, train_idx.tolist())
        val_ds = Subset(merged, val_idx.tolist())

    else:
        raise ValueError(f"Unknown mode: {mode}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader}
