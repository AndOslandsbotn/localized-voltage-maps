"""Voronoi cells of a region: k-means centroids fitted on the region's sample."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus


def assign_cells(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid (Voronoi cell) for each row of X."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2; ||x||^2 is the same for every c, so skip it.
    d2 = (centroids * centroids).sum(axis=1) - 2.0 * (X @ centroids.T)
    return np.argmin(d2, axis=1)


def fit_cells(
    sample: np.ndarray,
    n_cells: int,
    *,
    batch_size: int = 1024,
    max_iter: int = 100,
    n_local_trials: int = 2,
    seed: int | None = 0,
) -> np.ndarray:
    """Fit ``n_cells`` k-means centroids to one region's sample.

    k-means++ seeding followed by MiniBatchKMeans, both on the sample only.
    Depends on nothing but its arguments, so regions can be fitted in parallel.
    A region with fewer sample points than ``n_cells`` gets one cell per point.

    Returns
    -------
    centroids : (min(n_cells, len(sample)), d) ndarray
    """
    sample = np.asarray(sample, dtype=np.float64)
    k = min(n_cells, sample.shape[0])
    if k == 0:
        raise ValueError("cannot fit cells to an empty sample")
    init, _ = kmeans_plusplus(sample, n_clusters=k, random_state=seed, n_local_trials=n_local_trials)
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        init=init,
        n_init=1,
        batch_size=min(batch_size, sample.shape[0]),
        max_iter=max_iter,
        random_state=seed,
    )
    kmeans.fit(sample)
    return kmeans.cluster_centers_
