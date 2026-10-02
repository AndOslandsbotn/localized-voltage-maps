"""Embedding quality measures, computed on a random subset of the points.

* trustworthiness: are the embedding's k nearest neighbours also close in the
  original space? (penalises false neighbours)
* continuity: are the original k nearest neighbours still close in the
  embedding? (penalises torn-apart neighbourhoods). Computed as
  trustworthiness with the two spaces swapped.
* knn_accuracy: 5-fold cross-validated accuracy of a 5-NN classifier on the
  embedding coordinates, i.e. how well classes stay separated.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def quality(X: np.ndarray, Z: np.ndarray, y: np.ndarray, *, eval_size: int = 5000, k: int = 10, seed: int = 0) -> dict:
    n = X.shape[0]
    idx = np.random.default_rng(seed + 1).choice(n, min(eval_size, n), replace=False)
    Xs, Zs, ys = X[idx], Z[idx], y[idx]
    return {
        "trustworthiness": float(trustworthiness(Xs, Zs, n_neighbors=k)),
        "continuity": float(trustworthiness(Zs, Xs, n_neighbors=k)),
        "knn_accuracy": float(cross_val_score(KNeighborsClassifier(5), Zs, ys, cv=5).mean()),
    }
