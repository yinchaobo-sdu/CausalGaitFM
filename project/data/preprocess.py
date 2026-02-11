"""Preprocessing pipelines for each dataset -> unified numpy arrays.

Each dataset is converted to:
  - X: np.ndarray of shape [N, T, D]  (N windows, T timesteps, D channels)
  - y: np.ndarray of shape [N]        (activity / disease label)
  - subjects: np.ndarray of shape [N] (subject id per window)
  - domain_id: int                     (unique id for the dataset as a domain)

All signals are resampled / windowed to a common window length (default 256)
and z-score normalized per sensor channel.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

# Default sliding window parameters
DEFAULT_WINDOW = 256
DEFAULT_STEP = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sliding_windows(
    data: np.ndarray,
    window: int,
    step: int,
) -> np.ndarray:
    """Create sliding windows from [T, D] -> [N, window, D]."""
    if data.shape[0] < window:
        # pad with zeros
        pad = np.zeros((window - data.shape[0], data.shape[1]), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=0)
        return data[np.newaxis]
    starts = np.arange(0, data.shape[0] - window + 1, step)
    return np.array([data[s : s + window] for s in starts])


def _z_normalize(X: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel across the dataset. X: [N, T, D]."""
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Per-dataset processors
# ---------------------------------------------------------------------------

def _find_file(directory: Path, pattern: str) -> list[Path]:
    """Recursively find files matching a glob pattern."""
    return sorted(directory.rglob(pattern))


def preprocess_daphnet(
    raw_dir: Path,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> dict[str, np.ndarray]:
    """Daphnet Freezing of Gait: 10 PD patients, 3 IMUs (ankle, thigh, trunk)."""
    data_files = _find_file(raw_dir, "*.txt")
    if not data_files:
        data_files = _find_file(raw_dir, "S*.txt")

    all_X, all_y, all_subj = [], [], []
    for fpath in data_files:
        try:
            raw = np.loadtxt(fpath)
        except Exception:
            continue
        if raw.ndim != 2 or raw.shape[1] < 2:
            continue

        # Columns: timestamp, 9 accel channels, label (last column)
        signals = raw[:, 1:-1].astype(np.float32)
        labels = raw[:, -1].astype(np.int64)

        # Remove unlabeled (0) segments, keep 1=no freeze, 2=freeze
        valid = labels > 0
        signals = signals[valid]
        labels = labels[valid] - 1  # -> 0=no freeze, 1=freeze

        if signals.shape[0] < window:
            continue

        # Extract subject id from filename (e.g., S01R01.txt -> 1)
        match = re.search(r"S(\d+)", fpath.stem)
        subj_id = int(match.group(1)) if match else 0

        wins = _sliding_windows(signals, window, step)
        # Majority label per window
        lab_wins = _sliding_windows(labels.reshape(-1, 1), window, step).squeeze(-1)
        win_labels = np.array(
            [np.bincount(w.astype(int)).argmax() for w in lab_wins]
        )

        all_X.append(wins)
        all_y.append(win_labels)
        all_subj.append(np.full(len(wins), subj_id, dtype=np.int64))

    if not all_X:
        return {"X": np.zeros((0, window, 9), dtype=np.float32),
                "y": np.zeros(0, dtype=np.int64),
                "subjects": np.zeros(0, dtype=np.int64)}

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)
    X = _z_normalize(X)
    return {"X": X.astype(np.float32), "y": y, "subjects": subjects}


def preprocess_ucihar(
    raw_dir: Path,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> dict[str, np.ndarray]:
    """UCI HAR: 30 subjects, smartphone accelerometer + gyroscope."""
    # Try to find the Inertial Signals directory
    inertial_dirs = list(raw_dir.rglob("Inertial Signals"))

    if inertial_dirs:
        # Use raw inertial signals
        all_X, all_y, all_subj = [], [], []
        for split in ["train", "test"]:
            split_dir = None
            for d in inertial_dirs:
                if split in str(d).lower():
                    split_dir = d
                    break
            if split_dir is None:
                continue

            signal_files = sorted(split_dir.glob("*.txt"))
            if not signal_files:
                continue

            signals_list = []
            for sf in signal_files:
                data = np.loadtxt(sf)
                signals_list.append(data)

            if not signals_list:
                continue

            # Each file is [N_samples, 128], stack channels
            X_split = np.stack(signals_list, axis=-1).astype(np.float32)  # [N, 128, C]

            # Labels
            label_file = list(split_dir.parent.glob(f"y_{split}*.txt"))
            if label_file:
                y_split = np.loadtxt(label_file[0]).astype(np.int64) - 1  # 1-indexed -> 0-indexed
            else:
                continue

            subj_file = list(split_dir.parent.glob(f"subject_{split}*.txt"))
            if subj_file:
                subj_split = np.loadtxt(subj_file[0]).astype(np.int64)
            else:
                subj_split = np.zeros(len(y_split), dtype=np.int64)

            # Windows are already 128 samples; resample if needed
            if X_split.shape[1] != window:
                from scipy.signal import resample
                X_split = resample(X_split, window, axis=1).astype(np.float32)

            all_X.append(X_split)
            all_y.append(y_split)
            all_subj.append(subj_split)

        if all_X:
            X = np.concatenate(all_X, axis=0)
            y = np.concatenate(all_y, axis=0)
            subjects = np.concatenate(all_subj, axis=0)
            X = _z_normalize(X)
            return {"X": X.astype(np.float32), "y": y, "subjects": subjects}

    # Fallback: use pre-processed X_train.txt / X_test.txt
    all_X, all_y, all_subj = [], [], []
    for split in ["train", "test"]:
        x_files = list(raw_dir.rglob(f"X_{split}.txt"))
        y_files = list(raw_dir.rglob(f"y_{split}.txt"))
        s_files = list(raw_dir.rglob(f"subject_{split}.txt"))
        if not x_files or not y_files:
            continue
        X_data = np.loadtxt(x_files[0]).astype(np.float32)
        y_data = np.loadtxt(y_files[0]).astype(np.int64) - 1
        s_data = np.loadtxt(s_files[0]).astype(np.int64) if s_files else np.zeros(len(y_data), dtype=np.int64)
        # X_data is [N, 561] features; reshape into pseudo-windows
        n_features = X_data.shape[1]
        n_channels = min(n_features, 9)  # Use first 9 features as channels
        X_reshaped = X_data[:, :n_channels].reshape(-1, 1, n_channels)
        X_reshaped = np.repeat(X_reshaped, window, axis=1)
        all_X.append(X_reshaped)
        all_y.append(y_data)
        all_subj.append(s_data)

    if not all_X:
        return {"X": np.zeros((0, window, 9), dtype=np.float32),
                "y": np.zeros(0, dtype=np.int64),
                "subjects": np.zeros(0, dtype=np.int64)}
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)
    X = _z_normalize(X)
    return {"X": X.astype(np.float32), "y": y, "subjects": subjects}


def preprocess_pamap2(
    raw_dir: Path,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> dict[str, np.ndarray]:
    """PAMAP2: 9 subjects, 3 IMUs, 12+ activities."""
    data_files = _find_file(raw_dir, "subject10*.dat")
    if not data_files:
        data_files = _find_file(raw_dir, "*.dat")

    all_X, all_y, all_subj = [], [], []
    for fpath in data_files:
        try:
            raw = np.loadtxt(fpath)
        except Exception:
            try:
                raw = np.genfromtxt(fpath, invalid_raise=False)
            except Exception:
                continue

        if raw.ndim != 2 or raw.shape[1] < 3:
            continue

        # Col 0=timestamp, col 1=activity_id, col 2=heart_rate, cols 3+=IMU data
        labels = raw[:, 1].astype(np.int64)
        signals = raw[:, 3:].astype(np.float32)

        # Replace NaN with 0
        signals = np.nan_to_num(signals, nan=0.0)

        # Remove transient activities (label 0)
        valid = labels > 0
        signals = signals[valid]
        labels = labels[valid]

        # Remap labels to 0-indexed
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int64)

        match = re.search(r"subject(\d+)", fpath.stem, re.IGNORECASE)
        subj_id = int(match.group(1)) if match else 0

        if signals.shape[0] < window:
            continue

        wins = _sliding_windows(signals, window, step)
        lab_wins = _sliding_windows(labels.reshape(-1, 1), window, step).squeeze(-1)
        win_labels = np.array(
            [np.bincount(w.astype(int), minlength=1).argmax() for w in lab_wins]
        )

        all_X.append(wins)
        all_y.append(win_labels)
        all_subj.append(np.full(len(wins), subj_id, dtype=np.int64))

    if not all_X:
        return {"X": np.zeros((0, window, 36), dtype=np.float32),
                "y": np.zeros(0, dtype=np.int64),
                "subjects": np.zeros(0, dtype=np.int64)}

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)
    X = _z_normalize(X)
    return {"X": X.astype(np.float32), "y": y, "subjects": subjects}


def preprocess_mhealth(
    raw_dir: Path,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> dict[str, np.ndarray]:
    """MHEALTH: 10 subjects, body-worn sensors, 12 activities."""
    data_files = _find_file(raw_dir, "mHealth_subject*.log")
    if not data_files:
        data_files = _find_file(raw_dir, "*.log")

    all_X, all_y, all_subj = [], [], []
    for fpath in data_files:
        try:
            raw = np.loadtxt(fpath)
        except Exception:
            continue
        if raw.ndim != 2 or raw.shape[1] < 2:
            continue

        # Last column is the label, rest are sensor channels
        signals = raw[:, :-1].astype(np.float32)
        labels = raw[:, -1].astype(np.int64)

        # Remove label 0 (no activity)
        valid = labels > 0
        signals = signals[valid]
        labels = labels[valid] - 1  # 0-indexed

        match = re.search(r"subject(\d+)", fpath.stem, re.IGNORECASE)
        subj_id = int(match.group(1)) if match else 0

        if signals.shape[0] < window:
            continue

        wins = _sliding_windows(signals, window, step)
        lab_wins = _sliding_windows(labels.reshape(-1, 1), window, step).squeeze(-1)
        win_labels = np.array(
            [np.bincount(w.astype(int), minlength=1).argmax() for w in lab_wins]
        )

        all_X.append(wins)
        all_y.append(win_labels)
        all_subj.append(np.full(len(wins), subj_id, dtype=np.int64))

    if not all_X:
        return {"X": np.zeros((0, window, 23), dtype=np.float32),
                "y": np.zeros(0, dtype=np.int64),
                "subjects": np.zeros(0, dtype=np.int64)}

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)
    X = _z_normalize(X)
    return {"X": X.astype(np.float32), "y": y, "subjects": subjects}


def preprocess_wisdm(
    raw_dir: Path,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> dict[str, np.ndarray]:
    """WISDM: 29 subjects, smartphone accelerometer, 6 activities."""
    data_files = _find_file(raw_dir, "WISDM_ar_v1.1_raw.txt")
    if not data_files:
        data_files = _find_file(raw_dir, "*.txt")

    all_X, all_y, all_subj = [], [], []

    activity_map = {
        "Walking": 0, "Jogging": 1, "Sitting": 2,
        "Standing": 3, "Upstairs": 4, "Downstairs": 5,
    }

    for fpath in data_files:
        try:
            lines = fpath.read_text(errors="replace").strip().split("\n")
        except Exception:
            continue

        subjects_data: dict[int, dict] = {}
        for line in lines:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                subj_id = int(parts[0])
                activity = parts[1].strip()
                x_val = float(parts[3])
                y_val = float(parts[4])
                z_val = float(parts[5].rstrip(";"))
            except (ValueError, IndexError):
                continue

            if activity not in activity_map:
                continue
            label = activity_map[activity]

            if subj_id not in subjects_data:
                subjects_data[subj_id] = {"signals": [], "labels": []}
            subjects_data[subj_id]["signals"].append([x_val, y_val, z_val])
            subjects_data[subj_id]["labels"].append(label)

        for subj_id, sdata in subjects_data.items():
            signals = np.array(sdata["signals"], dtype=np.float32)
            labels = np.array(sdata["labels"], dtype=np.int64)
            if signals.shape[0] < window:
                continue

            wins = _sliding_windows(signals, window, step)
            lab_wins = _sliding_windows(labels.reshape(-1, 1), window, step).squeeze(-1)
            win_labels = np.array(
                [np.bincount(w.astype(int), minlength=6).argmax() for w in lab_wins]
            )

            all_X.append(wins)
            all_y.append(win_labels)
            all_subj.append(np.full(len(wins), subj_id, dtype=np.int64))

    if not all_X:
        return {"X": np.zeros((0, window, 3), dtype=np.float32),
                "y": np.zeros(0, dtype=np.int64),
                "subjects": np.zeros(0, dtype=np.int64)}

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)
    X = _z_normalize(X)
    return {"X": X.astype(np.float32), "y": y, "subjects": subjects}


def preprocess_opportunity(
    raw_dir: Path,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> dict[str, np.ndarray]:
    """Opportunity: 4 subjects, 72 body-worn + ambient sensors, 17+ activities."""
    data_files = _find_file(raw_dir, "S*-ADL*.dat")
    if not data_files:
        data_files = _find_file(raw_dir, "*.dat")

    all_X, all_y, all_subj = [], [], []
    for fpath in data_files:
        try:
            raw = np.genfromtxt(fpath, filling_values=0.0)
        except Exception:
            continue
        if raw.ndim != 2 or raw.shape[1] < 3:
            continue

        # Use locomotion label (column 244, 0-indexed=243) or last column
        label_col = min(243, raw.shape[1] - 1)
        signals = raw[:, 1:label_col].astype(np.float32)
        labels = raw[:, label_col].astype(np.int64)

        # Replace NaN
        signals = np.nan_to_num(signals, nan=0.0)

        # Remove null class (0)
        valid = labels > 0
        signals = signals[valid]
        labels = labels[valid]

        # Remap to 0-indexed
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            continue
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int64)

        match = re.search(r"S(\d+)", fpath.stem)
        subj_id = int(match.group(1)) if match else 0

        if signals.shape[0] < window:
            continue

        wins = _sliding_windows(signals, window, step)
        lab_wins = _sliding_windows(labels.reshape(-1, 1), window, step).squeeze(-1)
        win_labels = np.array(
            [np.bincount(w.astype(int), minlength=1).argmax() for w in lab_wins]
        )

        all_X.append(wins)
        all_y.append(win_labels)
        all_subj.append(np.full(len(wins), subj_id, dtype=np.int64))

    if not all_X:
        return {"X": np.zeros((0, window, 113), dtype=np.float32),
                "y": np.zeros(0, dtype=np.int64),
                "subjects": np.zeros(0, dtype=np.int64)}

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)
    X = _z_normalize(X)
    return {"X": X.astype(np.float32), "y": y, "subjects": subjects}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREPROCESSORS = {
    "daphnet": preprocess_daphnet,
    "ucihar": preprocess_ucihar,
    "pamap2": preprocess_pamap2,
    "mhealth": preprocess_mhealth,
    "wisdm": preprocess_wisdm,
    "opportunity": preprocess_opportunity,
}


def preprocess_dataset(
    name: str,
    raw_dir: Path,
    processed_dir: Path,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Preprocess a single dataset and save to disk."""
    name = name.lower()
    out_path = processed_dir / f"{name}.npz"

    if out_path.exists() and not force:
        print(f"[skip] {name}: already preprocessed at {out_path}")
        data = np.load(out_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    if name not in PREPROCESSORS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(PREPROCESSORS)}")

    print(f"[preprocess] {name} ...")
    result = PREPROCESSORS[name](raw_dir / name, window=window, step=step)
    print(
        f"  -> X: {result['X'].shape}, y: {result['y'].shape}, "
        f"subjects: {result['subjects'].shape}, "
        f"n_classes: {len(np.unique(result['y']))}"
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **result)
    return result


def preprocess_all_datasets(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
    force: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Preprocess all six datasets."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    results = {}
    for name in PREPROCESSORS:
        try:
            results[name] = preprocess_dataset(
                name, raw_dir, processed_dir, window=window, step=step, force=force
            )
        except Exception as exc:
            print(f"[ERROR] Failed to preprocess {name}: {exc}")
    return results


if __name__ == "__main__":
    preprocess_all_datasets()
