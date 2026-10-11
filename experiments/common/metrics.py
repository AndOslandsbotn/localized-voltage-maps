from __future__ import annotations

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def quality(X: np.ndarray, Z: np.ndarray, labels: np.ndarray | None, *, seed: int, size: int = 5000,
            k: int = 10) -> dict[str, float]:
    """Local quality of an embedding Z of X, on a random ``size`` of the points:
    - trustworthiness: are the embedding's k nearest neighbours also near in the data (no false neighbours)?
    - continuity: are the data's k nearest neighbours still near in the embedding (no torn neighbourhoods)?
    - kNN accuracy: 5-fold cross-validated accuracy of a 5-nearest-neighbour classifier on Z (needs labels).
    - density preservation: do dense regions stay dense and sparse ones sparse (``density_preservation``)?
    - density slope: is the density contrast kept at its size (1), exaggerated (> 1) or flattened (< 1)?
    """
    rows = np.random.default_rng(seed + 1).choice(len(X), min(size, len(X)), replace=False)
    Xs, Zs = X[rows], Z[rows]
    scores = {"trustworthiness": float(trustworthiness(Xs, Zs, n_neighbors=k)),
              "continuity": float(trustworthiness(Zs, Xs, n_neighbors=k)),
              "density_preservation": density_preservation(X, Z, seed=seed, size=size),
              "density_slope": density_slope(X, Z, seed=seed, size=size)}
    if labels is not None:
        scores["knn_accuracy"] = float(cross_val_score(KNeighborsClassifier(5), Zs, labels[rows], cv=5).mean())
    return scores


def room_per_digit(X: np.ndarray, Z: np.ndarray, labels: np.ndarray, *, seed: int,
                   size: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Per digit: its data density (-log of its median distance to the 10th nearest point) and its room, the
    median over its points of (10-NN distance in the embedding) / (10-NN distance in the data), relative to all
    points' median."""
    rows = np.random.default_rng(seed + 1).choice(len(X), min(size, len(X)), replace=False)
    in_data = NearestNeighbors(n_neighbors=11).fit(X[rows]).kneighbors()[0][:, -1]
    in_embedding = NearestNeighbors(n_neighbors=11).fit(Z[rows]).kneighbors()[0][:, -1]
    ratio = in_embedding / in_data
    ratio /= np.median(ratio)
    digits = labels[rows]
    density = np.array([-np.log(np.median(in_data[digits == d])) for d in range(int(labels.max()) + 1)])
    room = np.array([np.median(ratio[digits == d]) for d in range(int(labels.max()) + 1)])
    return density, room


def density_preservation(X: np.ndarray, Z: np.ndarray, *, seed: int, size: int = 5000, k: int = 30) -> float:
    """Density preservation (Narayan, Berger & Cho, 2021, densMAP): the Pearson correlation of the log local radii
    in the data and in the embedding, on a random ``size`` of the points; a point's local radius is its mean distance
    to its k nearest neighbours (densMAP weights by the method's affinities instead)."""
    rows = np.random.default_rng(seed + 1).choice(len(X), min(size, len(X)), replace=False)
    in_data = NearestNeighbors(n_neighbors=k).fit(X[rows]).kneighbors()[0].mean(axis=1)
    in_embedding = NearestNeighbors(n_neighbors=k).fit(Z[rows]).kneighbors()[0].mean(axis=1)
    return float(np.corrcoef(np.log(in_data), np.log(in_embedding))[0, 1])


def density_slope(X: np.ndarray, Z: np.ndarray, *, seed: int, size: int = 5000, k: int = 30) -> float:
    """The least-squares slope of log local radius in the embedding against log local radius in the data, on the
    same points and radii as ``density_preservation``: 1 keeps the density contrast at its size."""
    rows = np.random.default_rng(seed + 1).choice(len(X), min(size, len(X)), replace=False)
    in_data = NearestNeighbors(n_neighbors=k).fit(X[rows]).kneighbors()[0].mean(axis=1)
    in_embedding = NearestNeighbors(n_neighbors=k).fit(Z[rows]).kneighbors()[0].mean(axis=1)
    x, y = np.log(in_data), np.log(in_embedding)
    if not np.all(np.isfinite(y)):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])
