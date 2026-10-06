"""Intrinsic dimension of a region (config section ``dimension``).

Estimated once per fit from the region's sample (before the cells are
built), so it is measured rather than assumed; in the hierarchy each region
gets its own estimate. It sets how many landmarks are needed: a position in
d dimensions is pinned down by its distances to d + 1 landmarks in general
position (``landmarks.count: dimension``).

The estimators come from scikit-dimension. Their cost is the exact
k-nearest-neighbour search, so on ``cuda`` that search runs in cuML and its
result is handed to the estimator. Configurable choice point ``dimension``;
see ``lvm.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lvm_old.config import DimensionConfig, FixedDimensionConfig, MleDimensionConfig, TwoNnDimensionConfig
from lvm_old.strategies import resolve


@dataclass(frozen=True)
class Dimension:
    d: float   # estimated intrinsic dimension (not rounded)


def choose_dimension(sample: np.ndarray, *, config: DimensionConfig, seed: int = 0, device: str = "cuda") -> Dimension:
    """Intrinsic dimension of the data in ``sample`` with the strategy in ``config.strategy``."""
    strategy, options = resolve(_STRATEGIES, config)
    return Dimension(float(strategy(np.asarray(sample), options=options, seed=seed, device=device)))


def _subsample(sample: np.ndarray, size: int, seed: int) -> np.ndarray:
    if sample.shape[0] <= size:
        return sample
    return sample[np.random.default_rng(seed).choice(sample.shape[0], size, replace=False)]


def _nearest_neighbours(X: np.ndarray, k: int, *, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Distances and indices of each point's ``k`` nearest other points, nearest first."""
    if device == "cuda":
        import cupy as cp
        from cuml.neighbors import NearestNeighbors

        X32 = np.ascontiguousarray(X, dtype=np.float32)
        idx = cp.asnumpy(NearestNeighbors(n_neighbors=k + 1).fit(X32).kneighbors(X32, return_distance=False))
        # Drop each point itself (by index: with duplicates it need not come first) ...
        own = idx == np.arange(len(X))[:, None]
        idx = np.take_along_axis(idx, np.argsort(own, axis=1, kind="stable"), axis=1)[:, :k]
        # ... and recompute the distances in float64: cuML's float32 |x|^2 + |y|^2 - 2 x.y cancels to 0 for
        # neighbours that are close relative to their norms, and the estimator takes log(T_k / T_j).
        G, I = cp.asarray(X, dtype=cp.float64), cp.asarray(idx)
        dist = cp.empty(idx.shape, dtype=cp.float64)
        for start in range(0, len(X), 2000):                          # bounded memory: 2000 x k x features
            rows = slice(start, start + 2000)
            dist[rows] = cp.linalg.norm(G[I[rows]] - G[rows, None, :], axis=2)
        dist = cp.asnumpy(dist)
        order = np.argsort(dist, axis=1)
        return np.take_along_axis(dist, order, axis=1), np.take_along_axis(idx, order, axis=1)
    from sklearn.neighbors import NearestNeighbors

    dist, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors()   # no query: each point is left out
    return dist, idx


def _mle(sample: np.ndarray, *, options: MleDimensionConfig, seed: int, device: str) -> float:
    """Levina & Bickel (2005) maximum-likelihood estimate from k-nearest-neighbour distances.

    Each point's estimate is (k - 1) / sum_j log(T_k / T_j) over its neighbour
    distances T_1 <= ... <= T_k; scikit-dimension combines them by the harmonic
    mean (MacKay & Ghahramani).
    """
    import skdim

    X = _subsample(sample, options.sample_size, seed)
    knn = _nearest_neighbours(X, options.k, device=device)
    return skdim.id.MLE().fit(X, precomputed_knn_arrays=knn).dimension_


def _twonn(sample: np.ndarray, *, options: TwoNnDimensionConfig, seed: int, device: str) -> float:
    """Facco et al. (2017): from the ratio of each point's second to first neighbour distance."""
    import skdim

    return skdim.id.TwoNN().fit(_subsample(sample, options.sample_size, seed)).dimension_


def _fixed(sample: np.ndarray, *, options: FixedDimensionConfig, seed: int, device: str) -> float:
    return options.d


_STRATEGIES: dict[str, Callable[..., float]] = {
    "mle": _mle,
    "twonn": _twonn,
    "fixed": _fixed,
}
