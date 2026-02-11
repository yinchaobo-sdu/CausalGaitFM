from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import Tensor


ArrayLike = Tensor | np.ndarray | list[int] | list[float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(values: ArrayLike, name: str) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)

    if values.ndim == 0:
        values = values.reshape(1)
    return values


def _to_class_labels(values: ArrayLike, name: str) -> np.ndarray:
    arr = _to_numpy(values, name=name)
    if arr.ndim == 1:
        return np.rint(arr).astype(np.int64)
    if arr.ndim == 2:
        return arr.argmax(axis=1).astype(np.int64)
    raise ValueError(f"`{name}` must be 1D class ids or 2D logits/probabilities, got shape {arr.shape}")


def _to_probs(values: ArrayLike, name: str) -> np.ndarray:
    """Convert logits/log-probs to proper probabilities [N, C]."""
    arr = _to_numpy(values, name=name)
    if arr.ndim == 1:
        return None  # cannot compute AUC from class labels alone
    if arr.ndim != 2:
        return None
    # Apply softmax if values look like logits (not already probabilities)
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.1):
        # Check if log-probs (sum of exp ~ 1)
        exp_sums = np.exp(arr).sum(axis=1)
        if np.allclose(exp_sums, 1.0, atol=0.1):
            arr = np.exp(arr)
        else:
            # Standard logits -> softmax
            exp_arr = np.exp(arr - arr.max(axis=1, keepdims=True))
            arr = exp_arr / exp_arr.sum(axis=1, keepdims=True)
    return arr


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0

    classes = np.unique(np.concatenate([y_true, y_pred], axis=0))
    if classes.size == 0:
        return 0.0

    f1_scores: list[float] = []
    for cls in classes:
        true_pos = float(np.logical_and(y_true == cls, y_pred == cls).sum())
        false_pos = float(np.logical_and(y_true != cls, y_pred == cls).sum())
        false_neg = float(np.logical_and(y_true == cls, y_pred != cls).sum())
        denom = 2.0 * true_pos + false_pos + false_neg
        f1_scores.append(0.0 if denom == 0.0 else (2.0 * true_pos) / denom)
    return float(np.mean(f1_scores))


def _auc_roc(y_true: np.ndarray, y_probs: np.ndarray | None) -> float:
    """Compute macro-averaged AUC-ROC. Returns 0.0 if not computable."""
    if y_probs is None or y_true.size == 0:
        return 0.0
    try:
        from sklearn.metrics import roc_auc_score
        n_classes = y_probs.shape[1]
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return 0.0
        # One-hot encode true labels
        y_onehot = np.zeros_like(y_probs)
        for i, label in enumerate(y_true):
            if 0 <= label < n_classes:
                y_onehot[i, label] = 1.0
        return float(roc_auc_score(y_onehot, y_probs, average="macro", multi_class="ovr"))
    except Exception:
        return 0.0


def _ordinal_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 1) -> float:
    """Ordinal accuracy: fraction of predictions within +-tolerance of true label.

    Paper Table 4 uses this metric for fall risk and frailty (ordinal tasks).
    """
    if y_true.size == 0:
        return 0.0
    within_tol = np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64)) <= tolerance
    return float(within_tol.mean())


def _resolve_key(mapping: Mapping[str, ArrayLike], aliases: tuple[str, ...], mapping_name: str) -> ArrayLike:
    for key in aliases:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing key in `{mapping_name}`. Expected one of: {aliases}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_metrics(
    y_true: Mapping[str, ArrayLike],
    y_pred: Mapping[str, ArrayLike],
) -> dict[str, float]:
    """Calculate task metrics (paper Tables 2-4):

    - Disease: Accuracy, Macro-F1, AUC-ROC
    - Fall risk: Accuracy, Macro-F1, AUC-ROC, Ordinal Accuracy
    - Frailty: Accuracy, Macro-F1, AUC-ROC, Ordinal Accuracy, MAE
    """
    # --- Extract raw predictions (before argmax) for AUC ---
    disease_pred_raw = _resolve_key(y_pred, ("disease", "disease_logits", "pred_disease"), "y_pred")
    fall_pred_raw = _resolve_key(y_pred, ("fall", "fall_logits", "pred_fall"), "y_pred")
    frailty_pred_raw = _resolve_key(y_pred, ("frailty", "frailty_logits", "pred_frailty"), "y_pred")

    disease_probs = _to_probs(disease_pred_raw, "disease_probs")
    fall_probs = _to_probs(fall_pred_raw, "fall_probs")
    frailty_probs = _to_probs(frailty_pred_raw, "frailty_probs")

    # --- Class labels ---
    disease_true = _to_class_labels(
        _resolve_key(y_true, ("disease", "label_disease"), "y_true"), name="disease_true",
    )
    disease_pred = _to_class_labels(disease_pred_raw, name="disease_pred")
    fall_true = _to_class_labels(
        _resolve_key(y_true, ("fall", "label_fall"), "y_true"), name="fall_true",
    )
    fall_pred = _to_class_labels(fall_pred_raw, name="fall_pred")
    frailty_true = _to_class_labels(
        _resolve_key(y_true, ("frailty", "label_frailty"), "y_true"), name="frailty_true",
    )
    frailty_pred = _to_class_labels(frailty_pred_raw, name="frailty_pred")

    if disease_true.shape != disease_pred.shape:
        raise ValueError(f"Disease shape mismatch: true {disease_true.shape}, pred {disease_pred.shape}")
    if fall_true.shape != fall_pred.shape:
        raise ValueError(f"Fall shape mismatch: true {fall_true.shape}, pred {fall_pred.shape}")
    if frailty_true.shape != frailty_pred.shape:
        raise ValueError(f"Frailty shape mismatch: true {frailty_true.shape}, pred {frailty_pred.shape}")

    return {
        # Disease (Table 2, 3)
        "disease_acc": _accuracy(disease_true, disease_pred),
        "disease_macro_f1": _macro_f1(disease_true, disease_pred),
        "disease_auc": _auc_roc(disease_true, disease_probs),
        # Fall risk (Table 4)
        "fall_acc": _accuracy(fall_true, fall_pred),
        "fall_macro_f1": _macro_f1(fall_true, fall_pred),
        "fall_auc": _auc_roc(fall_true, fall_probs),
        "fall_ordinal_acc": _ordinal_accuracy(fall_true, fall_pred),
        # Frailty (Table 4)
        "frailty_acc": _accuracy(frailty_true, frailty_pred),
        "frailty_macro_f1": _macro_f1(frailty_true, frailty_pred),
        "frailty_auc": _auc_roc(frailty_true, frailty_probs),
        "frailty_ordinal_acc": _ordinal_accuracy(frailty_true, frailty_pred),
        "frailty_mae": float(np.mean(np.abs(frailty_pred - frailty_true))) if frailty_true.size else 0.0,
    }


__all__ = ["calculate_metrics", "_ordinal_accuracy", "_auc_roc"]
