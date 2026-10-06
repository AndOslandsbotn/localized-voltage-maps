from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from lvm_new.config import GraphConfig
from lvm_new.config.config import GaussianKernelOptions, KernelConfig, PointKnnOptions
from lvm_new.compute import TORCH_DATA_DTYPE, TORCH_SOLVE_DTYPE


def build_graph(centroids: torch.Tensor, sample: torch.Tensor, *, config: GraphConfig, device: str,
                seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(kernel, radius): the (n_cells, n_cells) kernel between cells (zero diagonal) and each cell's radius, on the
    device."""
    n_cells = len(centroids)
    if n_cells < 2:
        empty = torch.zeros(n_cells, dtype=TORCH_SOLVE_DTYPE, device=device)
        return torch.zeros((n_cells, n_cells), dtype=TORCH_SOLVE_DTYPE, device=device), empty

    mean = centroids.mean(dim=0)
    centred = (centroids - mean).to(TORCH_DATA_DTYPE)
    sq_dist = torch.cdist(centred, centred).square().to(TORCH_SOLVE_DTYPE)
    sq_dist = (sq_dist + sq_dist.T) / 2
    sq_dist.fill_diagonal_(0.0)

    radius = config.radius
    match radius.strategy:
        case "adaptive_per_cell":
            sq_radius = _adaptive_per_cell(sq_dist, radius.adaptive_per_cell.k)
        case "knn":
            sq_radius = _knn(sq_dist, radius.knn.k)
        case "point_knn":
            sq_radius = _point_knn(sample, centred, mean, options=radius.point_knn, seed=seed)
        case "centroid_spacing":
            sq_radius = _centroid_spacing(sq_dist, radius.centroid_spacing.multiplier)
    pair = torch.maximum(sq_radius[:, None], sq_radius[None, :])
    kernel = _kernel(sq_dist, pair, config.kernel)
    kernel.fill_diagonal_(0.0)                                      # no self-loops
    if config.connect:
        _connect(kernel, sq_dist)
    return kernel, sq_radius.sqrt()


def _kth_neighbour(sq_dist: torch.Tensor, k: int) -> torch.Tensor:
    """(n,) each cell's squared distance to its k-th nearest other cell (itself, at distance 0, comes first)."""
    k = min(k, len(sq_dist) - 1)
    return torch.topk(sq_dist, k + 1, dim=1, largest=False).values[:, k]


def _one_radius(radius: torch.Tensor, n_cells: int) -> torch.Tensor:
    """(n_cells,) one squared radius for every cell, from a radius (a distance)."""
    return torch.full((n_cells,), float(radius) ** 2, dtype=TORCH_SOLVE_DTYPE, device=radius.device)


def _adaptive_per_cell(sq_dist: torch.Tensor, k: int) -> torch.Tensor:
    """Each cell its own radius, the distance to its k-th nearest cell (returned squared): with r_ij = max(r_i, r_j)
    two cells are joined when either is among the other's k nearest, the symmetric kNN graph (von Luxburg, 2007, "A
    tutorial on spectral clustering"), so every cell has at least k edges wherever it lies."""
    return _kth_neighbour(sq_dist, k)


def _knn(sq_dist: torch.Tensor, k: int) -> torch.Tensor:
    """One radius: the median over cells of the distance to their k-th nearest cell."""
    return _one_radius(torch.quantile(_kth_neighbour(sq_dist, k).sqrt(), 0.5), len(sq_dist))


def _point_knn(sample: torch.Tensor, centred: torch.Tensor, mean: torch.Tensor, *, options: PointKnnOptions,
               seed: int) -> torch.Tensor:
    """One radius: the median over data points (a subsample of the sample) of the distance to their k-th nearest cell.
    In high dimensions points sit farther from every centroid than centroids sit from each other, so this radius is
    larger than knn's."""
    points = sample
    if len(sample) > options.sample_size:
        rows = np.random.default_rng(seed).choice(len(sample), options.sample_size, replace=False)
        points = sample[torch.as_tensor(rows, device=sample.device)]
    points = (points.to(TORCH_SOLVE_DTYPE) - mean).to(TORCH_DATA_DTYPE)
    k = min(options.k, len(centred))
    kth = torch.topk(torch.cdist(points, centred), k, dim=1, largest=False).values[:, k - 1].to(TORCH_SOLVE_DTYPE)
    return _one_radius(torch.quantile(kth, 0.5), len(centred))


def _centroid_spacing(sq_dist: torch.Tensor, multiplier: float) -> torch.Tensor:
    """One radius: multiplier x the median distance to the nearest cell. Fine in low dimensions; in high ones it
    connects almost every cell (distances concentrate)."""
    return _one_radius(multiplier * torch.quantile(_kth_neighbour(sq_dist, 1).sqrt(), 0.5), len(sq_dist))


def _kernel(sq_dist: torch.Tensor, sq_radius: torch.Tensor, config: KernelConfig) -> torch.Tensor:
    """The kernel between cells from their squared distances and squared pair radii, by the strategy in ``config``."""
    match config.strategy:
        case "radial":
            return _radial(sq_dist, sq_radius)
        case "tapered":
            return _tapered(sq_dist, sq_radius)
        case "gaussian":
            return _gaussian(sq_dist, sq_radius, options=config.gaussian)


def _radial(sq_dist: torch.Tensor, sq_radius: torch.Tensor) -> torch.Tensor:
    """K_ij = 1 if d_ij <= r_ij, else 0 (compared on squares)."""
    return (sq_dist <= sq_radius).to(TORCH_SOLVE_DTYPE)


def _tapered(sq_dist: torch.Tensor, sq_radius: torch.Tensor) -> torch.Tensor:
    """K_ij = (1 - d_ij^2 / r_ij^2)+: falls smoothly to 0 at r_ij."""
    return torch.clamp(1.0 - sq_dist / sq_radius, min=0.0)


def _gaussian(sq_dist: torch.Tensor, sq_radius: torch.Tensor, *, options: GaussianKernelOptions) -> torch.Tensor:
    """K_ij = exp(-d_ij^2 / (2 s^2)) with s = sigma r_ij, set to 0 beyond cutoff * s."""
    sq_width = options.sigma ** 2 * sq_radius
    return torch.exp(-sq_dist / (2.0 * sq_width)) * (sq_dist <= options.cutoff ** 2 * sq_width)


def _connect(kernel: torch.Tensor, sq_dist: torch.Tensor) -> None:
    """Join every piece of the graph to the largest, in place: one piece at a time (largest first), each by its
    closest pair of cells to the growing main piece, with weight 1. The pieces are labelled with SciPy on the CPU:
    faster than on the GPU at these sizes, the copy of the edge list included."""
    n_cells = len(kernel)
    edges = (kernel > 0).nonzero().cpu().numpy()
    n_pieces, label = connected_components(
        coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_cells, n_cells)), directed=False)
    if n_pieces == 1:
        return
    sizes = np.bincount(label)
    main = torch.as_tensor(label == sizes.argmax(), device=kernel.device)
    label = torch.as_tensor(label, device=kernel.device)
    for piece in sorted(set(range(n_pieces)) - {int(sizes.argmax())}, key=lambda c: -sizes[c]):
        inside = (label == piece).nonzero().squeeze(1)
        outside = main.nonzero().squeeze(1)
        nearest = sq_dist[inside][:, outside].argmin()
        a, b = inside[nearest // len(outside)], outside[nearest % len(outside)]
        kernel[a, b] = kernel[b, a] = 1.0
        main[inside] = True
