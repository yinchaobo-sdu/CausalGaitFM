from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


ArrayLike = Tensor | np.ndarray


def _to_numpy_2d(values: ArrayLike, name: str) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(f"`{name}` must be 2D [N, D], got shape {values.shape}")
    return values


def _to_numpy_1d(values: ArrayLike, name: str) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(f"`{name}` must be 1D [N], got shape {values.shape}")
    return values.astype(np.int64)


def _run_tsne(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ImportError("visualize_latent_space requires `scikit-learn`.") from exc

    n = x.shape[0]
    if n < 2:
        raise ValueError("t-SNE needs at least 2 points.")

    perplexity = min(30, max(5, n // 5))
    perplexity = min(perplexity, n - 1)
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=random_state,
    )
    return tsne.fit_transform(x)


def _scatter_by_label(ax, embedding: np.ndarray, labels: np.ndarray, title: str, cmap_name: str) -> None:
    import matplotlib.pyplot as plt

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap(cmap_name, len(unique_labels))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=20,
            alpha=0.8,
            color=cmap(idx),
            label=str(int(label)),
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(title="Class", fontsize=8, title_fontsize=9, loc="best")


def visualize_latent_space(
    z_c: ArrayLike,
    z_d: ArrayLike,
    domain_ids: ArrayLike,
    disease_labels: ArrayLike,
    save_path: str | Path,
) -> Path:
    """
    Save side-by-side t-SNE plots for latent representations:
      1) z_c colored by disease labels
      2) z_c colored by domain ids
      3) z_d colored by domain ids
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ImportError("visualize_latent_space requires `matplotlib`.") from exc

    z_c_np = _to_numpy_2d(z_c, name="z_c")
    z_d_np = _to_numpy_2d(z_d, name="z_d")
    domain_np = _to_numpy_1d(domain_ids, name="domain_ids")
    disease_np = _to_numpy_1d(disease_labels, name="disease_labels")

    n = z_c_np.shape[0]
    if z_d_np.shape[0] != n or domain_np.shape[0] != n or disease_np.shape[0] != n:
        raise ValueError("z_c, z_d, domain_ids, and disease_labels must share the same first dimension.")

    emb_zc = _run_tsne(z_c_np, random_state=42)
    emb_zd = _run_tsne(z_d_np, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _scatter_by_label(axes[0], emb_zc, disease_np, "Zc clustered by disease", cmap_name="tab10")
    _scatter_by_label(axes[1], emb_zc, domain_np, "Zc mixed across domains", cmap_name="tab20")
    _scatter_by_label(axes[2], emb_zd, domain_np, "Zd clustered by domain", cmap_name="tab20")
    fig.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = ["visualize_latent_space"]
