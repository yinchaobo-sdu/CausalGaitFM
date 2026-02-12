"""Data loading and preprocessing for CausalGaitFM."""

from .dataset import GaitDataset, MultiDomainGaitDataset, create_dataloaders


def download_all_datasets(*args, **kwargs):
    """Lazily import download utilities to avoid eager optional deps at package import."""
    from .download import download_all_datasets as _download_all_datasets
    return _download_all_datasets(*args, **kwargs)


def preprocess_all_datasets(*args, **kwargs):
    """Lazily import preprocess utilities to avoid import-time dependency coupling."""
    from .preprocess import preprocess_all_datasets as _preprocess_all_datasets
    return _preprocess_all_datasets(*args, **kwargs)

__all__ = [
    "GaitDataset",
    "MultiDomainGaitDataset",
    "create_dataloaders",
    "download_all_datasets",
    "preprocess_all_datasets",
]
