from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from torch import Tensor


ArrayLike = Tensor | np.ndarray | list[int] | list[float]


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


def _resolve_key(mapping: Mapping[str, ArrayLike], aliases: tuple[str, ...], mapping_name: str) -> ArrayLike:
    for key in aliases:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing key in `{mapping_name}`. Expected one of: {aliases}")


def calculate_metrics(
    y_true: Mapping[str, ArrayLike],
    y_pred: Mapping[str, ArrayLike],
) -> dict[str, float]:
    """
    Calculate task metrics:
      - Disease: Accuracy, Macro-F1
      - Fall risk: Accuracy, Macro-F1
      - Frailty (ordinal classes): Accuracy, Macro-F1, MAE
    """
    disease_true = _to_class_labels(
        _resolve_key(y_true, ("disease", "label_disease"), "y_true"),
        name="disease_true",
    )
    disease_pred = _to_class_labels(
        _resolve_key(y_pred, ("disease", "disease_logits", "pred_disease"), "y_pred"),
        name="disease_pred",
    )
    fall_true = _to_class_labels(
        _resolve_key(y_true, ("fall", "label_fall"), "y_true"),
        name="fall_true",
    )
    fall_pred = _to_class_labels(
        _resolve_key(y_pred, ("fall", "fall_logits", "pred_fall"), "y_pred"),
        name="fall_pred",
    )
    frailty_true = _to_class_labels(
        _resolve_key(y_true, ("frailty", "label_frailty"), "y_true"),
        name="frailty_true",
    )
    frailty_pred = _to_class_labels(
        _resolve_key(y_pred, ("frailty", "frailty_logits", "pred_frailty"), "y_pred"),
        name="frailty_pred",
    )

    if disease_true.shape != disease_pred.shape:
        raise ValueError(f"Disease shape mismatch: true {disease_true.shape}, pred {disease_pred.shape}")
    if fall_true.shape != fall_pred.shape:
        raise ValueError(f"Fall shape mismatch: true {fall_true.shape}, pred {fall_pred.shape}")
    if frailty_true.shape != frailty_pred.shape:
        raise ValueError(f"Frailty shape mismatch: true {frailty_true.shape}, pred {frailty_pred.shape}")

    return {
        "disease_acc": _accuracy(disease_true, disease_pred),
        "disease_macro_f1": _macro_f1(disease_true, disease_pred),
        "fall_acc": _accuracy(fall_true, fall_pred),
        "fall_macro_f1": _macro_f1(fall_true, fall_pred),
        "frailty_acc": _accuracy(frailty_true, frailty_pred),
        "frailty_macro_f1": _macro_f1(frailty_true, frailty_pred),
        "frailty_mae": float(np.mean(np.abs(frailty_pred - frailty_true))) if frailty_true.size else 0.0,
    }


__all__ = ["calculate_metrics"]
