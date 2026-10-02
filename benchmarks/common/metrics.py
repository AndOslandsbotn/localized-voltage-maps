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
    X: np.ndarray, Z: np.ndarray, y: np.ndarray, *,
    eval_size: int = 5000, global_size: int = 1500, k: int = 10, seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(X.shape[0], min(eval_size, X.shape[0]), replace=False)
    Xs, Zs, ys = X[idx], Z[idx], y[idx]
    g = idx[: min(global_size, idx.size)]
    return {
        "trustworthiness": float(trustworthiness(Xs, Zs, n_neighbors=k)),
        "continuity": float(trustworthiness(Zs, Xs, n_neighbors=k)),
        "knn_accuracy": float(cross_val_score(KNeighborsClassifier(5), Zs, ys, cv=5).mean()),
        "distance_correlation": float(spearmanr(pdist(X[g]), pdist(Z[g])).statistic),
    }
