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
from array_api_compat import array_namespace

from lvm_old.config import (
    AdaptivePerCellRadiusConfig, CentroidSpacingConfig, GaussianKernelConfig, KernelConfig, KnnRadiusConfig,
    PointKnnRadiusConfig, RadiusConfig,
)
from lvm_old.strategies import resolve


@dataclass(frozen=True)
class Radius:
    r: float                              # one radius for the region (adaptive_per_cell: the median of per_cell)
    per_cell: np.ndarray | None = None    # (n,) each cell's own radius (adaptive_per_cell), else None

    def between_cells(self):
        """Radius for the cell-to-cell kernel: r, or the (n, n) matrix max(r_i, r_j) (symmetric kNN graph)."""
        return self.r if self.per_cell is None else np.maximum.outer(self.per_cell, self.per_cell)

    def to_cells(self):
        """Radius for a points-to-cells kernel: r, or (n,) each cell's own radius (one per column)."""
        return self.r if self.per_cell is None else self.per_cell


@dataclass(frozen=True)
class Kernel:
    K: np.ndarray   # (m, n) kernel; between cells (m = n) the diagonal is zero


# --- kernel radius (config graph.radius) ------------------------------------

def choose_radius(centroids: np.ndarray, *, config: RadiusConfig, points: np.ndarray | None = None) -> Radius:
    """Kernel radius r for a region with the strategy named in ``config.strategy``.

    ``points`` are data points of the region (its sample), needed by
    ``point_knn``. Every strategy returns r = 0.0 for a region with a single
    cell (it has no neighbours to reach).
    """
    strategy, options = resolve(_RADIUS_STRATEGIES, config)
    return strategy(np.asarray(centroids, dtype=np.float64), options=options, points=points)


def _knn(centroids: np.ndarray, *, options: KnnRadiusConfig, points=None) -> Radius:
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
    return Radius(float(np.median(_neighbour_distances(centroids, k)[:, k - 1])))


def _point_knn(centroids: np.ndarray, *, options: PointKnnRadiusConfig, points=None) -> Radius:
    """r = median over data points of the distance to their k-th nearest centroid.

    A typical point then has about k cells within r. In high dimensions a
    point sits farther from every centroid than centroids sit from each other
    (averaging removes each point's own deviation), so ``_knn``'s r leaves
    about half of the points with no cell within reach; this one doesn't.
    """
    if points is None:
        raise ValueError("the point_knn radius needs the region's sample points")
    if centroids.shape[0] < 2:
        return Radius(0.0)
    from sklearn.neighbors import NearestNeighbors

    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] > options.sample_size:
        points = points[np.random.default_rng(0).choice(points.shape[0], options.sample_size, replace=False)]
    k = min(options.k, centroids.shape[0])
    dist, _ = NearestNeighbors(n_neighbors=k).fit(centroids).kneighbors(points)
    return Radius(float(np.median(dist[:, k - 1])))


def _centroid_spacing(centroids: np.ndarray, *, options: CentroidSpacingConfig, points=None) -> Radius:
    """r = multiplier * median nearest-neighbour centroid distance.

    Fine in low dimensions; in high ones it connects almost every cell, see
    ``_knn``.
    """
    if centroids.shape[0] < 2:
        return Radius(0.0)
    return Radius(float(options.multiplier * np.median(_neighbour_distances(centroids, 1)[:, 0])))


def _adaptive_per_cell(centroids: np.ndarray, *, options: AdaptivePerCellRadiusConfig, points=None) -> Radius:
    """Each cell its own radius: r_i = distance to its k-th nearest centroid.

    With r_ij = max(r_i, r_j) (``Radius.between_cells``) two cells are joined
    when either is among the other's k nearest, so every cell has at least k
    edges wherever it lies: like UMAP's per-point scale, it removes the
    dependence of connectivity on how spread out a part of the data is. One
    global r (``_knn``) leaves cells in spread-out parts with few edges, and
    voltage then has to detour along thin chains.
    """
    if options.k < 1:
        raise ValueError(f"k must be >= 1, got {options.k}")
    if centroids.shape[0] < 2:
        return Radius(0.0)
    k = min(options.k, centroids.shape[0] - 1)
    # The k-th neighbour lies exactly on r_i; the kernel's distances come from another routine, so rounding could
    # drop it in one direction only (an asymmetric K). A relative margin far above rounding keeps it in.
    per_cell = _neighbour_distances(centroids, k)[:, k - 1] * (1.0 + 1e-9)
    return Radius(float(np.median(per_cell)), per_cell)


def _neighbour_distances(centroids: np.ndarray, k: int) -> np.ndarray:
    """(n, k) distances from each centroid to its k nearest other centroids, ascending."""
    from sklearn.neighbors import NearestNeighbors

    # Ask for k + 1: each centroid's nearest neighbour in its own set is itself.
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(centroids).kneighbors(centroids)
    return dist[:, 1:]


_RADIUS_STRATEGIES: dict[str, Callable[..., Radius]] = {
    "knn": _knn,
    "point_knn": _point_knn,
    "centroid_spacing": _centroid_spacing,
    "adaptive_per_cell": _adaptive_per_cell,
}


# --- kernel (config graph.kernel) -------------------------------------------

def choose_kernel(sq_dist: np.ndarray, *, r, config: KernelConfig, exclude_self: bool = False) -> Kernel:
    """Kernel from squared distances with the strategy named in ``config.strategy``.

    ``sq_dist`` is an (m, n) NumPy array or torch tensor (the kernel comes back
    as the same type, on the same device), e.g. ``cells.sq_distances(points, centroids)`` from
    data points to cells, or ``sq_distances(centroids, centroids)`` between
    cells. Taking distances rather than coordinates lets a caller compute them
    once and reuse them (the pipeline also needs each point's nearest cell).
    ``r`` is the kernel radius: a number, or an array broadcastable to
    ``sq_dist`` (a radius per pair or per column), e.g.
    ``choose_radius(...).between_cells()`` or ``.to_cells()``.
    ``exclude_self=True`` (cells to cells) zeroes the diagonal: a self-loop
    carries no current.
    """
    strategy, options = resolve(_KERNEL_STRATEGIES, config)
    sq_dist = sq_dist if hasattr(sq_dist, "shape") else np.asarray(sq_dist)
    if not np.isscalar(r) and not isinstance(sq_dist, np.ndarray):          # per-cell radii, onto the tensor's device
        import torch

        r = torch.as_tensor(np.asarray(r), dtype=sq_dist.dtype, device=sq_dist.device)
    K = strategy(sq_dist, options=options, r=r)
    if exclude_self:
        if K.shape[0] != K.shape[1]:
            raise ValueError(f"exclude_self needs a square distance matrix, got {tuple(K.shape)}")
        if isinstance(K, np.ndarray):
            np.fill_diagonal(K, 0.0)
        else:
            K.fill_diagonal_(0.0)
    return Kernel(K)


def _radial(sq_dist, *, options: None, r: float):
    """K_ij = 1{d_ij <= r}, in the dtype and on the device of ``sq_dist`` (NumPy or torch)."""
    xp = array_namespace(sq_dist)
    return xp.astype(sq_dist <= r * r, sq_dist.dtype)


def _tapered(sq_dist, *, options: None, r: float):
    """K_ij = (1 - d_ij^2 / r^2)_+: falls smoothly to 0 at r, with the radial kernel's support."""
    xp = array_namespace(sq_dist)
    return xp.clip(1.0 - sq_dist / (r * r), 0.0, None)


def _gaussian(sq_dist, *, options: GaussianKernelConfig, r: float):
    """K_ij = exp(-d_ij^2 / (2 s^2)) with s = sigma * r, set to 0 beyond cutoff * s."""
    xp = array_namespace(sq_dist)
    s2 = (options.sigma * r) ** 2
    inside = xp.astype(sq_dist <= options.cutoff**2 * s2, sq_dist.dtype)
    return xp.exp(-sq_dist / (2.0 * s2)) * inside


_KERNEL_STRATEGIES: dict[str, Callable[..., np.ndarray]] = {
    "radial": _radial,
    "tapered": _tapered,
    "gaussian": _gaussian,
}


def adaptive_knn_kernel(sq_dist, *, k: int, sharpness: float = 1.0):
    """Points to cells: weights on each point's k nearest cells, 0 elsewhere (NumPy or torch).

    w_i = exp(-sharpness (d_i^2 - d_1^2) / (2 s^2)) over the point's k nearest
    cells, with d_1 its nearest and s^2 the mean of d_i^2 - d_1^2 over those k. In
    high dimensions every point is about equally far from all nearby
    centroids (its own deviation from the cell mean adds to every distance);
    subtracting d_1^2 removes that shared part, leaving how much closer the
    point is to one cell than another. The nearest cell gets weight 1.
    """
    k = min(k, sq_dist.shape[1])
    if isinstance(sq_dist, np.ndarray):
        idx = np.argpartition(sq_dist, k - 1, axis=1)[:, :k]
        d2 = np.take_along_axis(sq_dist, idx, axis=1)
        excess = d2 - d2.min(axis=1, keepdims=True)
        s2 = excess.mean(axis=1, keepdims=True)
        w = np.exp(-sharpness * excess / (2.0 * np.where(s2 > 0, s2, 1.0)))
        K = np.zeros_like(sq_dist)
        np.put_along_axis(K, idx, w, axis=1)
        return K
    import torch

    d2, idx = torch.topk(sq_dist, k, dim=1, largest=False)
    excess = d2 - d2[:, :1]
    s2 = excess.mean(dim=1, keepdim=True)
    w = torch.exp(-sharpness * excess / (2.0 * torch.where(s2 > 0, s2, torch.ones_like(s2))))
    return torch.zeros_like(sq_dist).scatter_(1, idx, w)


def connect_components(K: np.ndarray, sq_dist: np.ndarray) -> np.ndarray:
    """K with every connected piece of the graph joined to the largest one by its shortest link (weight 1).

    ``sq_dist`` are the squared distances between the same nodes. Pieces are
    joined one at a time, each through its closest pair of nodes to the
    growing main piece, so the result is connected with as few, and as short,
    added edges as possible. K itself is not changed.
    """
    from scipy.sparse.csgraph import connected_components

    K = np.array(K, copy=True)
    D = np.asarray(sq_dist, dtype=np.float64)
    n_pieces, label = connected_components(K > 0, directed=False)
    if n_pieces == 1:
        return K
    main = label == np.bincount(label).argmax()
    for piece in sorted(set(label[~main].tolist()), key=lambda c: -np.sum(label == c)):
        inside = np.flatnonzero(label == piece)
        sub = D[np.ix_(inside, np.flatnonzero(main))]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = inside[i], np.flatnonzero(main)[j]
        K[a, b] = K[b, a] = 1.0
        main[inside] = True
    return K


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
