"""DCI-D (Disentanglement, Completeness, Informativeness) metric.

Reference: Eastwood & Williams, "A framework for the quantitative evaluation of
disentangled representations", ICLR 2018.

Used in Figure 2c to evaluate the quality of causal-domain factor separation
across different latent dimensions (dc/ds grid search).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _entropy(probs: NDArray) -> float:
    """Shannon entropy of a probability distribution."""
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs + 1e-12)))


def _compute_importance_matrix(
    z: NDArray,
    factors: NDArray,
    n_trees: int = 50,
    max_depth: int = 6,
    seed: int = 42,
) -> NDArray:
    """Compute importance matrix R[i,j] = importance of latent dim j for predicting factor i.

    Uses sklearn GradientBoosting for each factor separately.
    Returns R: [n_factors, z_dim].
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        # Fallback: use simple correlation-based importance
        return _correlation_importance(z, factors)

    n_factors = factors.shape[1] if factors.ndim == 2 else 1
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)

    z_dim = z.shape[1]
    R = np.zeros((n_factors, z_dim))

    for i in range(n_factors):
        y = factors[:, i].astype(int)
        if len(np.unique(y)) < 2:
            continue
        clf = GradientBoostingClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=seed,
        )
        clf.fit(z, y)
        R[i] = clf.feature_importances_

    return R


def _correlation_importance(z: NDArray, factors: NDArray) -> NDArray:
    """Fallback: absolute correlation as importance proxy."""
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    n_factors = factors.shape[1]
    z_dim = z.shape[1]
    R = np.zeros((n_factors, z_dim))
    for i in range(n_factors):
        for j in range(z_dim):
            R[i, j] = abs(np.corrcoef(z[:, j], factors[:, i])[0, 1])
    # Replace NaN with 0
    R = np.nan_to_num(R)
    return R


def disentanglement_score(R: NDArray) -> float:
    """DCI Disentanglement: each latent dim should be important for at most one factor.

    D = 1 - H(p_j) / log(K) averaged over latent dims,
    where p_j is the normalized importance of dim j across factors.
    """
    n_factors, z_dim = R.shape
    if n_factors <= 1 or z_dim == 0:
        return 1.0  # trivially disentangled

    max_ent = np.log(n_factors)
    if max_ent < 1e-12:
        return 1.0

    scores = []
    for j in range(z_dim):
        col = R[:, j]
        total = col.sum()
        if total < 1e-12:
            continue
        p_j = col / total
        scores.append(1.0 - _entropy(p_j) / max_ent)

    return float(np.mean(scores)) if scores else 0.0


def completeness_score(R: NDArray) -> float:
    """DCI Completeness: each factor should be captured by at most one latent dim.

    C = 1 - H(p_i) / log(L) averaged over factors.
    """
    n_factors, z_dim = R.shape
    if z_dim <= 1 or n_factors == 0:
        return 1.0

    max_ent = np.log(z_dim)
    if max_ent < 1e-12:
        return 1.0

    scores = []
    for i in range(n_factors):
        row = R[i]
        total = row.sum()
        if total < 1e-12:
            continue
        p_i = row / total
        scores.append(1.0 - _entropy(p_i) / max_ent)

    return float(np.mean(scores)) if scores else 0.0


def informativeness_score(
    z: NDArray,
    factors: NDArray,
    seed: int = 42,
) -> float:
    """DCI Informativeness: prediction accuracy of factors from latent (averaged)."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return 0.0

    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)

    scores = []
    for i in range(factors.shape[1]):
        y = factors[:, i].astype(int)
        if len(np.unique(y)) < 2:
            scores.append(1.0)
            continue
        clf = GradientBoostingClassifier(
            n_estimators=50, max_depth=6, random_state=seed,
        )
        cv_scores = cross_val_score(clf, z, y, cv=min(5, len(np.unique(y))),
                                     scoring="accuracy")
        scores.append(float(cv_scores.mean()))

    return float(np.mean(scores)) if scores else 0.0


def compute_dci(
    z: NDArray,
    factors: NDArray,
    n_trees: int = 50,
    max_depth: int = 6,
    seed: int = 42,
) -> dict[str, float]:
    """Compute full DCI metrics.

    Args:
        z: Latent representations [N, z_dim].
        factors: Ground-truth generative factors [N, n_factors] or [N,].
        n_trees: Number of trees for GBT importance.
        max_depth: Max depth for GBT.
        seed: Random seed.

    Returns:
        Dictionary with keys: disentanglement, completeness, informativeness.
    """
    z = np.asarray(z, dtype=np.float64)
    factors = np.asarray(factors)
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)

    R = _compute_importance_matrix(z, factors, n_trees=n_trees, max_depth=max_depth, seed=seed)

    return {
        "dci_disentanglement": disentanglement_score(R),
        "dci_completeness": completeness_score(R),
        "dci_informativeness": informativeness_score(z, factors, seed=seed),
    }


def compute_dci_from_model(
    z_c: NDArray,
    z_d: NDArray,
    disease_labels: NDArray,
    domain_ids: NDArray,
) -> dict[str, float]:
    """Convenience function for CausalGaitFM evaluation (Figure 2c).

    Evaluates disentanglement quality:
    - z_c (causal) should be informative about disease labels
    - z_d (domain) should be informative about domain ids
    - z_c should NOT encode domain info (and vice versa)

    Returns DCI metrics for both causal and domain factors.
    """
    z_c = np.asarray(z_c, dtype=np.float64)
    z_d = np.asarray(z_d, dtype=np.float64)
    disease_labels = np.asarray(disease_labels).ravel()
    domain_ids = np.asarray(domain_ids).ravel()

    # Combine latent dims
    z_all = np.concatenate([z_c, z_d], axis=1)
    factors = np.stack([disease_labels, domain_ids], axis=1)

    dci = compute_dci(z_all, factors)

    # Also compute per-subspace metrics for detailed analysis
    results = {
        "dci_d": dci["dci_disentanglement"],
        "dci_c": dci["dci_completeness"],
        "dci_i": dci["dci_informativeness"],
    }

    return results


__all__ = ["compute_dci", "compute_dci_from_model", "disentanglement_score",
           "completeness_score", "informativeness_score"]
