"""Embedding quality measures, computed on a random subset of the embedded points.

Local structure:
* trustworthiness: are the embedding's k nearest neighbours also close in
  the original space? (penalises false neighbours). The tuning criterion.
* continuity: are the original k nearest neighbours still close in the
  embedding? (penalises torn neighbourhoods); trustworthiness with the two
  spaces swapped.
* knn_accuracy: 5-fold cross-validated accuracy of a 5-NN classifier on the
  embedding coordinates. Uses labels, so it is reported but never used to
  choose hyperparameters.

Global structure:
* distance_correlation: Spearman rank correlation between all pairwise
  distances in the original space and in the embedding, on a smaller subset
  (as in Kobak & Linderman, 2019). Local measures cannot see whether far-apart
  groups are placed sensibly relative to each other; this can.

A dataset without labels has no knn_accuracy (nan); a dataset can add its own
measures (``datasets.Dataset.extra_metrics``), computed on the global subset.

All of these compare rankings, so they cannot see points piled onto the same
spot; ``stacking`` measures that, for pictures.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

METRICS = ("trustworthiness", "continuity", "knn_accuracy", "distance_correlation")


def quality(
    X: np.ndarray, Z: np.ndarray, y: np.ndarray | None, *,
    eval_size: int = 5000, global_size: int = 1500, k: int = 10, seed: int = 0, extra=None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(X.shape[0], min(eval_size, X.shape[0]), replace=False)
    Xs, Zs = X[idx], Z[idx]
    g = idx[: min(global_size, idx.size)]
    knn = float("nan") if y is None else float(cross_val_score(KNeighborsClassifier(5), Zs, y[idx], cv=5).mean())
    return {
        "trustworthiness": float(trustworthiness(Xs, Zs, n_neighbors=k)),
        "continuity": float(trustworthiness(Zs, Xs, n_neighbors=k)),
        "knn_accuracy": knn,
        "distance_correlation": float(spearmanr(pdist(X[g]), pdist(Z[g])).statistic),
        **(extra(X[g], Z[g]) if extra else {}),
    }


def stacking(Z: np.ndarray, grid: int = 1000) -> dict[str, float]:
    """How piled up an embedding looks: the squares of a grid x grid raster over it that hold points, and
    that count per point (``visible_share``: 1 if every point has a square of its own, small if points pile up).
    A picture shows a pile as one dot; rank-based measures cannot see it."""
    span = np.ptp(Z, axis=0)
    squares = np.floor((Z - Z.min(axis=0)) / np.where(span > 0, span / grid, 1.0)).astype(np.int64)
    occupied = len(np.unique(squares, axis=0))
    return {"occupied_squares": int(occupied), "visible_share": float(occupied / len(Z))}
