"""Voronoi cells of a region: k-means centroids fitted on the region's sample."""

from __future__ import annotations

import numpy as np
import torch


def sq_distances(X: np.ndarray, C: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    """Squared Euclidean distances (len(X), len(C)) as ||x||^2 - 2 x.c + ||c||^2.

    One matrix multiply (BLAS), which is >10x faster than scipy's cdist for
    the point-to-cell distances the pipeline needs. Rounding can make a
    distance slightly negative; those are clipped to 0. ``dtype=np.float32``
    halves the time again and is accurate enough for kernels and nearest cells.
    """
    X = np.asarray(X, dtype=dtype)
    C = np.asarray(C, dtype=dtype)
    d2 = X @ C.T
    d2 *= -2.0
    d2 += (X * X).sum(axis=1)[:, None]
    d2 += (C * C).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0, out=d2)


def assign_cells(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid (Voronoi cell) for each row of X."""
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2; ||x||^2 is the same for every c, so skip it.
    d2 = (centroids * centroids).sum(axis=1) - 2.0 * (X @ centroids.T)
    return np.argmin(d2, axis=1)


def fit_cells(
    sample: np.ndarray,
    n_cells: int,
    *,
    max_iter: int = 30,
    tol: float = 1e-4,
    n_local_trials: int = 2,
    init_sample_size: int | None = 20000,
    seed: int | None = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Fit ``n_cells`` k-means centroids to one region's sample.

    k-means++ seeding (with ``n_local_trials`` candidates per step, as in
    scikit-learn) on a random subset of ``init_sample_size`` points (None: the
    whole sample), followed by full-batch Lloyd iterations on the whole
    sample, all in torch on ``device`` in float32. Seeding is sequential (one
    centre per step), so seeding from a subset is what keeps it cheap; Lloyd
    then refines the centres on every sample point. Lloyd stops after ``max_iter`` iterations or once
    no centroid moves more than ``tol`` times the sample's spread. Depends on
    nothing but its arguments, so regions can be fitted in parallel. A region
    with fewer sample points than ``n_cells`` gets one cell per point.

    Returns
    -------
    centroids : (min(n_cells, len(sample)), d) float64 ndarray
    """
    sample = np.asarray(sample)
    k = min(n_cells, sample.shape[0])
    if k == 0:
        raise ValueError("cannot fit cells to an empty sample")
    X = torch.as_tensor(sample, dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(0 if seed is None else seed)

    x_sq = (X * X).sum(dim=1)
    if init_sample_size is not None and init_sample_size < X.shape[0]:
        init = torch.randperm(X.shape[0], generator=gen, device=device)[: max(init_sample_size, k)]
        C = _kmeans_plus_plus(X[init], x_sq[init], k, n_local_trials, gen)
    else:
        C = _kmeans_plus_plus(X, x_sq, k, n_local_trials, gen)
    scale = tol * float(X.var(dim=0).sum())
    for _ in range(max_iter):
        labels = _nearest(X, x_sq, C)
        sums = torch.zeros_like(C).index_add_(0, labels, X)
        counts = torch.bincount(labels, minlength=k).to(X.dtype)
        # A cluster that lost all its points keeps its old centroid.
        new_C = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], C)
        shift = float(((new_C - C) ** 2).sum(dim=1).max())
        C = new_C
        if shift <= scale:
            break
    return C.double().cpu().numpy()


def _nearest(X: torch.Tensor, x_sq: torch.Tensor, C: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Nearest centroid per row, in row chunks so the distance matrix stays small."""
    c_sq = (C * C).sum(dim=1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        out[s : s + chunk] = (c_sq[None, :] - 2.0 * X[s : s + chunk] @ C.T).argmin(dim=1)
    return out


def _kmeans_plus_plus(
    X: torch.Tensor, x_sq: torch.Tensor, k: int, n_local_trials: int, gen: torch.Generator
) -> torch.Tensor:
    """k-means++ seeding with ``n_local_trials`` candidates per step (greedy k-means++).

    Everything stays on the device: no Python ints are read back inside the
    loop, so the GPU never waits for the host between steps.
    """
    n = X.shape[0]
    C = torch.empty((k, X.shape[1]), dtype=X.dtype, device=X.device)
    first = torch.randint(n, (1,), generator=gen, device=X.device)
    C[0] = X[first[0]]
    # Squared distance from every point to its nearest chosen centre.
    closest = (x_sq - 2.0 * (X @ C[0]) + x_sq[first[0]]).clamp_(min=0.0)
    for i in range(1, k):
        # Candidates drawn with probability proportional to closest; keep the
        # one that lowers the total potential the most.
        cand = torch.multinomial(closest, n_local_trials, replacement=True, generator=gen)
        d_cand = (x_sq[None, :] - 2.0 * (X[cand] @ X.T) + x_sq[cand][:, None]).clamp_(min=0.0)
        new_closest = torch.minimum(closest[None, :], d_cand)
        best = new_closest.sum(dim=1).argmin()
        C[i] = X[cand[best]]
        closest = new_closest[best]
    return C
