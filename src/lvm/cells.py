"""Voronoi cells of a region (k-means centroids) and point-to-cell distances.

k-means is a configurable choice point (config ``cells.kmeans``; see
``lvm.strategies``) backed by existing libraries: RAPIDS cuML on the GPU,
FAISS or scikit-learn on the CPU. Distances and nearest-cell lookups also use
library routines -- scikit-learn for NumPy arrays, ``torch.cdist`` for torch
tensors -- so the per-point steps can run on whichever device holds the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from lvm.config import (CumlKMeansConfig, FaissKMeansConfig, KMeansConfig, RefineConfig, SklearnKMeansConfig,
                        StreamRefineConfig)
from lvm.strategies import resolve

if TYPE_CHECKING:
    from lvm.stream import ChunkSource


@dataclass(frozen=True)
class Cells:
    centroids: np.ndarray   # (k, d) float64


# --- distances ------------------------------------------------------------

def sq_distances(X, C, *, dtype=np.float64):
    """Squared Euclidean distances (len(X), len(C)).

    NumPy inputs use scikit-learn's ``euclidean_distances`` (BLAS-based, exact
    zeros on the diagonal when X is C); torch tensors use ``torch.cdist`` on
    their own device. ``dtype`` applies to NumPy inputs only; tensors keep theirs.
    """
    if _is_torch(X):
        return _torch().cdist(X, C).square()
    from sklearn.metrics.pairwise import euclidean_distances

    X = np.asarray(X, dtype=dtype)
    C = X if C is X else np.asarray(C, dtype=dtype)
    return euclidean_distances(X, C, squared=True)


def assign_cells(X: np.ndarray, centroids: np.ndarray, *, device: str | None = None) -> np.ndarray:
    """Index of the nearest centroid (Voronoi cell) for each row of X, as a NumPy array.

    ``device=None`` or "cpu" uses scikit-learn's chunked
    ``pairwise_distances_argmin``; another device (e.g. "cuda") copies the
    chunk there and uses ``torch.cdist``.
    """
    if device in (None, "cpu"):
        from sklearn.metrics import pairwise_distances_argmin

        return pairwise_distances_argmin(np.asarray(X, dtype=np.float32), np.asarray(centroids, dtype=np.float32))
    torch = _torch()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    Ct = torch.as_tensor(np.asarray(centroids), dtype=torch.float32, device=device)
    return torch.cdist(Xt, Ct).argmin(dim=1).cpu().numpy()


# --- k-means (config cells.kmeans) -----------------------------------------

def fit_cells(sample: np.ndarray, *, config: KMeansConfig, n_cells: int, seed: int | None = 0) -> Cells:
    """Fit ``n_cells`` k-means centroids to one region's sample, with the strategy in ``config.strategy``.

    Depends on nothing but its arguments, so regions can be fitted in
    parallel. A region with at most ``n_cells`` sample points gets one cell
    per point.
    """
    sample = np.asarray(sample)                   # float32 or float64 as given: no copy of a large sample
    if sample.shape[0] == 0:
        raise ValueError("cannot fit cells to an empty sample")
    if sample.shape[0] <= n_cells:
        return Cells(sample.copy())
    strategy, options = resolve(_KMEANS_STRATEGIES, config)
    centroids = strategy(sample, options=options, n_cells=n_cells, seed=0 if seed is None else seed)
    return Cells(np.asarray(centroids, dtype=np.float64))


def _cuml(sample: np.ndarray, *, options: CumlKMeansConfig, n_cells: int, seed: int) -> np.ndarray:
    """RAPIDS cuML k-means on the GPU (float32)."""
    from cuml.cluster import KMeans

    km = KMeans(n_clusters=n_cells, init=options.init, max_iter=options.max_iter, tol=options.tol,
                n_init=1, random_state=seed)
    return np.asarray(km.fit(np.asarray(sample, dtype=np.float32)).cluster_centers_)   # no copy if float32


def _faiss(sample: np.ndarray, *, options: FaissKMeansConfig, n_cells: int, seed: int) -> np.ndarray:
    """FAISS k-means on the CPU (float32, multi-threaded)."""
    import faiss

    # max_points_per_centroid: train on the whole sample, don't let FAISS subsample it.
    km = faiss.Kmeans(sample.shape[1], n_cells, niter=options.niter, seed=seed,
                      max_points_per_centroid=sample.shape[0], verbose=False)
    km.train(np.ascontiguousarray(sample, dtype=np.float32))
    return km.centroids


def _sklearn(sample: np.ndarray, *, options: SklearnKMeansConfig, n_cells: int, seed: int) -> np.ndarray:
    """scikit-learn k-means (k-means++ init, Lloyd) on the CPU: the reference implementation."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_cells, n_init=1, max_iter=options.max_iter, tol=options.tol, random_state=seed)
    return km.fit(sample.astype(np.float32)).cluster_centers_


_KMEANS_STRATEGIES: dict[str, Callable[..., np.ndarray]] = {
    "cuml": _cuml,
    "faiss": _faiss,
    "sklearn": _sklearn,
}


# --- refinement over the stream (config cells.refine) -------------------------

def refine_cells(
    source: ChunkSource, sample: np.ndarray, centroids: np.ndarray, *, config: RefineConfig, skip: int = 0,
    device: str = "cpu",
) -> Cells:
    """Centroids fitted to the whole stream, starting from ``centroids`` (k-means on ``sample``).

    ``skip``: points at the start of each scan's first pass that are ``sample``
    itself (a shuffled stream's sample is its prefix), so they are not counted
    twice. Strategy ``config.strategy``; none returns ``centroids`` unchanged.
    """
    strategy, options = resolve(_REFINE_STRATEGIES, config)
    return Cells(np.asarray(strategy(source, sample, centroids, options=options, skip=skip, device=device),
                            dtype=np.float64))


def _none(source, sample, centroids, *, options: None, skip: int, device: str) -> np.ndarray:
    return centroids


def _stream(source, sample, centroids, *, options: StreamRefineConfig, skip: int, device: str) -> np.ndarray:
    """Mini-batch k-means (Sculley 2010), one chunk of the stream per step, in torch on ``device``.

    Each centroid moves to the running mean of every point assigned to it so far
    (per-centroid step 1 / count), starting with the sample's points at the
    seed centroids. Only one chunk and the centroids are held, whatever the
    size of the data. Distances in float32, the running means in float64.
    """
    torch = _torch()
    C = torch.tensor(np.asarray(centroids), dtype=torch.float64, device=device)   # a copy: updated in place
    counts = torch.as_tensor(np.bincount(assign_cells(sample, centroids, device=device), minlength=C.shape[0]),
                             dtype=torch.float64, device=device)
    for p in range(options.passes):
        to_skip, seen = (skip if p == 0 else 0), 0
        for chunk in source():
            if to_skip:
                drop = min(to_skip, chunk.shape[0])
                chunk, to_skip = chunk[drop:], to_skip - drop
            if options.max_points is not None:
                chunk = chunk[: options.max_points - seen]
            if chunk.shape[0] == 0:
                if options.max_points is not None and seen >= options.max_points:
                    break
                continue
            X = torch.as_tensor(np.asarray(chunk), dtype=torch.float32, device=device)
            C32 = C.to(torch.float32)
            label = ((C32 * C32).sum(dim=1)[None, :] - 2.0 * (X @ C32.T)).argmin(dim=1)   # |x|^2 is the same for all
            n_c = torch.bincount(label, minlength=C.shape[0]).to(torch.float64)
            sums = torch.zeros_like(C32).index_add_(0, label, X).to(torch.float64)
            counts += n_c
            C += (sums - n_c[:, None] * C) / counts.clamp(min=1.0)[:, None]
            seen += chunk.shape[0]
    return C.cpu().numpy()


_REFINE_STRATEGIES: dict[str, Callable[..., np.ndarray]] = {
    "none": _none,
    "stream": _stream,
}


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _torch():
    import torch

    return torch
