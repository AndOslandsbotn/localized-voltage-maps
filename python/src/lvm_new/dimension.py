from __future__ import annotations

import numpy as np

from lvm_new.config import DimensionConfig
from lvm_new.config.config import FixedDimensionOptions, MleOptions
from lvm_new.precision import DATA_DTYPE, SOLVE_DTYPE


def estimate_dimension(sample: np.ndarray, *, config: DimensionConfig, device: str, seed: int) -> float:
    """The region's intrinsic dimension d̂ (unrounded), by the strategy in ``config``."""
    return float(_STRATEGIES[config.strategy](sample, options=config.options, sample_size=config.sample_size,
                                              device=device, seed=seed))


def _mle(sample: np.ndarray, *, options: MleOptions, sample_size: int, device: str, seed: int) -> float:
    """Levina & Bickel (2005), "Maximum likelihood estimation of intrinsic dimension", NIPS 17.

    Each point's estimate is (k - 1) / sum_j log(T_k / T_j) over its neighbour distances T_1 <= ... <= T_k;
    scikit-dimension combines them by the harmonic mean (MacKay & Ghahramani, 2005).
    """
    import skdim

    X = _subsample(sample, sample_size, seed)
    return skdim.id.MLE().fit(X, precomputed_knn_arrays=_nearest_neighbours(X, options.k, device=device)).dimension_


def _twonn(sample: np.ndarray, *, options: None, sample_size: int, device: str, seed: int) -> float:
    """Facco, d'Errico, Rodriguez & Laio (2017), "Estimating the intrinsic dimension of datasets by a minimal
    neighborhood information", Scientific Reports 7:12140. From the ratio of each point's second to first neighbour
    distance (scikit-dimension, on the CPU)."""
    import skdim

    return skdim.id.TwoNN().fit(_subsample(sample, sample_size, seed)).dimension_


def _fixed(sample: np.ndarray, *, options: FixedDimensionOptions, sample_size: int, device: str, seed: int) -> float:
    return options.d


def _subsample(sample: np.ndarray, size: int, seed: int) -> np.ndarray:
    if len(sample) <= size:
        return sample
    return sample[np.random.default_rng(seed).choice(len(sample), size, replace=False)]


def _nearest_neighbours(X: np.ndarray, k: int, *, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Distances and indices of each point's k nearest other points, nearest first."""
    if device == "cpu":
        from sklearn.neighbors import NearestNeighbors

        return NearestNeighbors(n_neighbors=k).fit(X).kneighbors()      # no query: each point is left out
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    # cuML computes distances in float32 as |x|^2 + |y|^2 - 2 x.y: far from the origin this rounds badly enough to
    # pick wrong neighbours. Distances don't change when all points shift, so search on the centred points.
    centred = (X - X.mean(axis=0, dtype=SOLVE_DTYPE)).astype(DATA_DTYPE)
    idx = cp.asnumpy(NearestNeighbors(n_neighbors=k + 1).fit(centred).kneighbors(centred, return_distance=False))
    # Drop each point itself by index: with duplicate points it needn't come first.
    own = idx == np.arange(len(X))[:, None]
    idx = np.take_along_axis(idx, np.argsort(own, axis=1, kind="stable"), axis=1)[:, :k]
    # Close neighbours can still round to distance 0 in float32, and the estimator takes logarithms of distance
    # ratios: recompute the distances in SOLVE_DTYPE, 2000 points at a time.
    points, neighbours = cp.asarray(X, dtype=SOLVE_DTYPE), cp.asarray(idx)
    dist = cp.empty(idx.shape, dtype=SOLVE_DTYPE)
    for start in range(0, len(X), 2000):
        rows = slice(start, start + 2000)
        dist[rows] = cp.linalg.norm(points[neighbours[rows]] - points[rows, None, :], axis=2)
    dist = cp.asnumpy(dist)
    order = np.argsort(dist, axis=1)
    return np.take_along_axis(dist, order, axis=1), np.take_along_axis(idx, order, axis=1)


_STRATEGIES = {
    "mle": _mle,
    "twonn": _twonn,
    "fixed": _fixed,
}
