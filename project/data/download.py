"""Download scripts for the six benchmark gait / activity recognition datasets."""

from __future__ import annotations

import io
import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------
DATASET_URLS: dict[str, str] = {
    "daphnet": (
        "https://archive.ics.uci.edu/static/public/245/daphnet+freezing+of+gait.zip"
    ),
    "ucihar": (
        "https://archive.ics.uci.edu/static/public/240/"
        "human+activity+recognition+using+smartphones.zip"
    ),
    "pamap2": (
        "https://archive.ics.uci.edu/static/public/231/"
        "pamap2+physical+activity+monitoring.zip"
    ),
    "mhealth": (
        "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
    ),
    "wisdm": (
        "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/"
        "WISDM_ar_latest.tar.gz"
    ),
    "opportunity": (
        "https://archive.ics.uci.edu/static/public/226/"
        "opportunity+activity+recognition.zip"
    ),
}

# Mirror / alternative URLs (some UCI URLs change over time)
FALLBACK_URLS: dict[str, list[str]] = {
    "daphnet": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00245/daphnet_dataset.zip",
    ],
    "ucihar": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
    ],
    "pamap2": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
    ],
    "mhealth": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
    ],
    "wisdm": [],
    "opportunity": [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip",
    ],
}


def _download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [warn] Failed to download {url}: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return True


def _extract(archive_path: Path, extract_dir: Path) -> None:
    """Extract zip or tar.gz archive."""
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        import tarfile
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_dataset(name: str, raw_dir: Path) -> Path:
    """Download a single dataset into *raw_dir/<name>*. Returns the dataset folder."""
    name = name.lower()
    if name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(DATASET_URLS)}")

    dest_dir = raw_dir / name
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[skip] {name}: already downloaded at {dest_dir}")
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = DATASET_URLS[name]
    suffix = ".tar.gz" if url.endswith(".tar.gz") else ".zip"
    archive_path = raw_dir / f"{name}{suffix}"

    # Try primary URL, then fallbacks
    urls_to_try = [url] + FALLBACK_URLS.get(name, [])
    success = False
    for u in urls_to_try:
        print(f"[download] {name} from {u}")
        if _download_file(u, archive_path):
            success = True
            break

    if not success:
        print(f"[ERROR] Could not download {name} from any URL.")
        print(f"  Please download manually and place files in {dest_dir}")
        return dest_dir

    print(f"[extract] {archive_path.name} -> {dest_dir}")
    _extract(archive_path, dest_dir)
    archive_path.unlink(missing_ok=True)
    return dest_dir


def download_all_datasets(raw_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Download all six benchmark datasets. Returns mapping name -> folder."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in DATASET_URLS:
        results[name] = download_dataset(name, raw_dir)
    return results


if __name__ == "__main__":
    download_all_datasets()
