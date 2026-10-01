"""Mass-weighted grounded resistor graph over one region's cells.

The graph nodes are the region's Voronoi cells (k-means centroids c_i) with
streamed masses p_i. Following Def. 6 of Structure_from_Voltage.pdf, with the
empirical measure replaced by the cell masses:

    W_ij  = p_i p_j k(c_i, c_j)        (edge weights, i != j)
    rho_i = rho_g p_i                  (weight of node i's edge to ground)

so the grounded Laplacian is ``L_rho = diag(W 1 + rho_g p) - W``.

Configurable choice points (see ``lvm.strategies``): the kernel radius
(``choose_radius``, config ``graph.radius``) and the kernel
(``choose_kernel``, config ``graph.kernel``). Every function here depends only
on its arguments, so regions can be handled in parallel. ``rho_g`` is kept
out of the kernel so step 7's scaling search can rebuild ``L_rho`` for many
values of ``rho_g`` from the same K and p.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lvm.cells import sq_distances
from lvm.config import CentroidSpacingConfig, KernelConfig, KnnRadiusConfig, RadiusConfig
from lvm.strategies import resolve


@dataclass(frozen=True)
class Radius:
    r: float


@dataclass(frozen=True)
class Kernel:
    K: np.ndarray   # (m, n) kernel; between cells (m = n) the diagonal is zero


# --- kernel radius (config graph.radius) ------------------------------------

def choose_radius(centroids: np.ndarray, *, config: RadiusConfig) -> Radius:
    """Kernel radius r for a region with the strategy named in ``config.strategy``.

    Every strategy returns r = 0.0 for a region with a single cell (it has no
    neighbours to reach).
    """
    strategy, options = resolve(_RADIUS_STRATEGIES, config)
    return strategy(np.asarray(centroids, dtype=np.float64), options=options)


def _knn(centroids: np.ndarray, *, options: KnnRadiusConfig) -> Radius:
    """r = median over cells of the distance to their k-th nearest centroid.

    A cell at the median then has about k neighbours within r, whatever the
    dimension. A fixed multiple of the nearest-neighbour distance can't
    promise that: in high dimensions the nearest and the typical distances
    become similar, and the ball reaches almost every cell. k is capped at the
    number of other cells.
    """
    if options.k < 1:
        raise ValueError(f"k must be >= 1, got {options.k}")
    if centroids.shape[0] < 2:
        return Radius(0.0)
    k = min(options.k, centroids.shape[0] - 1)
    d2 = sq_distances(centroids, centroids)
    np.fill_diagonal(d2, np.inf)
    kth = np.partition(d2, k - 1, axis=1)[:, k - 1]
    return Radius(float(np.sqrt(np.median(kth))))


def _centroid_spacing(centroids: np.ndarray, *, options: CentroidSpacingConfig) -> Radius:
    """r = multiplier * median nearest-neighbour centroid distance.

    Fine in low dimensions; in high ones it connects almost every cell, see
    ``_knn``.
    """
    if centroids.shape[0] < 2:
        return Radius(0.0)
    d2 = sq_distances(centroids, centroids)
    np.fill_diagonal(d2, np.inf)
    return Radius(float(options.multiplier * np.sqrt(np.median(d2.min(axis=1)))))


_RADIUS_STRATEGIES: dict[str, Callable[..., Radius]] = {
    "knn": _knn,
    "centroid_spacing": _centroid_spacing,
}


# --- kernel (config graph.kernel) -------------------------------------------

def choose_kernel(sq_dist: np.ndarray, *, r: float, config: KernelConfig, exclude_self: bool = False) -> Kernel:
    """Kernel from squared distances with the strategy named in ``config.strategy``.

    ``sq_dist`` is (m, n), e.g. ``cells.sq_distances(points, centroids)`` from
    data points to cells, or ``sq_distances(centroids, centroids)`` between
    cells. Taking distances rather than coordinates lets a caller compute them
    once and reuse them (the pipeline also needs each point's nearest cell).
    ``r`` is the kernel radius, e.g. ``choose_radius(...).r``.
    ``exclude_self=True`` (cells to cells) zeroes the diagonal: a self-loop
    carries no current.
    """
    strategy, options = resolve(_KERNEL_STRATEGIES, config)
    K = strategy(np.asarray(sq_dist), options=options, r=r)
    if exclude_self:
        if K.shape[0] != K.shape[1]:
            raise ValueError(f"exclude_self needs a square distance matrix, got {K.shape}")
        np.fill_diagonal(K, 0.0)
    return Kernel(K)


def _radial(sq_dist: np.ndarray, *, options: None, r: float) -> np.ndarray:
    """K_ij = 1{d_ij <= r}."""
    return (sq_dist <= r * r).astype(np.float64)


_KERNEL_STRATEGIES: dict[str, Callable[..., np.ndarray]] = {
    "radial": _radial,
}


# --- grounded Laplacian -----------------------------------------------------

def grounded_laplacian(K: np.ndarray, p: np.ndarray, rho_g: float) -> np.ndarray:
    """Grounded Laplacian L_rho = diag(W 1 + rho_g p) - W with W = p p^T * K.

    Config ``graph.edge_weighting: mass`` and ``ground.weighting: mass``.

    Row i of ``L_rho @ v = 0`` divided by p_i reads
    ``v_i = sum_j K_ij p_j v_j / (rho_g + sum_j K_ij p_j)``: a free cell's
    voltage is a mass-weighted average of its neighbours', and its own mass
    cancels. So a low-mass cell gets a well-defined voltage while barely
    influencing the others. A cell with p_i = 0 has an all-zero row, which
    makes L_rho singular; ``voltage.solve_grounded_voltage_maps`` solves on
    the cells with p_i > 0 and fills zero-mass cells in afterwards.
    """
    K = np.asarray(K, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] != p.shape[0]:
        raise ValueError(f"K must be (n, n) and p (n,), got {K.shape} and {p.shape}")
    if rho_g <= 0:
        raise ValueError(f"rho_g must be > 0, got {rho_g}")
    if np.any(p < 0):
        raise ValueError("cell masses p must be non-negative")
    W = np.outer(p, p) * K
    return np.diag(W.sum(axis=1) + rho_g * p) - W
