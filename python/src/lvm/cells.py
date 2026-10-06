from __future__ import annotations

import math

import numpy as np
import torch

from lvm.config import Config
from lvm.config.config import CumlKMeansOptions, LloydKMeansOptions
from lvm.compute import TORCH_DATA_DTYPE, TORCH_SOLVE_DTYPE, rows_per_block


def fit_cells(sample: torch.Tensor, *, config: Config, device: str, seed: int) -> torch.Tensor:
    if len(sample) <= config.cells.n_cells:
        return sample.to(TORCH_SOLVE_DTYPE)
    # Centred, so float32 distances stay accurate far from the origin.
    shift = sample.mean(dim=0, dtype=TORCH_SOLVE_DTYPE).to(TORCH_DATA_DTYPE)
    centred = sample - shift
    strategy = config.cells.kmeans.strategy
    if strategy == "auto":
        strategy = "cuml" if device == "cuda" else "lloyd"
    match strategy:
        case "cuml":
            centroids = _cuml(centred, config.cells.n_cells, options=config.cells.kmeans.cuml, seed=seed)
        case "lloyd":
            centroids = _lloyd(centred, config.cells.n_cells, options=config.cells.kmeans.lloyd, seed=seed)
    return torch.as_tensor(centroids, device=device).to(TORCH_SOLVE_DTYPE) + shift.to(TORCH_SOLVE_DTYPE)


def _cuml(sample: torch.Tensor, n_cells: int, *, options: CumlKMeansOptions, seed: int):
    import cupy as cp
    from cuml.cluster import KMeans

    km = KMeans(n_clusters=n_cells, init=options.init, max_iter=options.max_iter, tol=options.tol, n_init=1,
                random_state=seed, output_type="cupy")
    return km.fit(cp.asarray(sample)).cluster_centers_      # cuML reads the tensor's GPU memory directly


def _lloyd(sample: torch.Tensor, n_cells: int, *, options: LloydKMeansOptions, seed: int) -> torch.Tensor:
    """Lloyd's algorithm (Lloyd, 1982, "Least squares quantization in PCM"): each round moves every centroid to the
    mean of the points nearest to it. Starts from random sample points; an empty cell keeps its centroid."""
    rows = np.random.default_rng(seed).choice(len(sample), n_cells, replace=False)
    centroids = sample[torch.as_tensor(rows, device=sample.device)].to(TORCH_SOLVE_DTYPE)
    for _ in range(options.iterations):
        counts, sums = _cell_sums(sample, centroids)
        filled = counts > 0
        centroids[filled] = sums[filled] / counts[filled, None]
    return centroids


def refine_cells(chunks, sample: torch.Tensor, centroids: torch.Tensor, *, config: Config,
                 device: str) -> torch.Tensor:
    """Mini-batch k-means (Sculley, 2010, "Web-scale k-means clustering"): after each chunk, every centroid is the
    running mean of all points assigned to it so far, the sample's included. With a prefix sample, the sample is the
    start of the stream; it is skipped in the first pass so it counts once."""
    refine, prefix = config.cells.refine, config.sample.strategy == "prefix"
    if refine.passes == 0:
        return centroids
    centroids = centroids.clone()                                  # updated in place
    counts = torch.zeros(len(centroids), dtype=TORCH_SOLVE_DTYPE, device=device)
    counts += torch.bincount(assign_cells(sample, centroids), minlength=len(centroids))
    for p in range(refine.passes):
        to_skip, seen = (len(sample) if prefix and p == 0 else 0), 0
        for chunk in chunks:
            if to_skip:
                drop = min(to_skip, len(chunk))
                chunk, to_skip = chunk[drop:], to_skip - drop
            if refine.max_points is not None:
                chunk = chunk[:refine.max_points - seen]
            if len(chunk) == 0:
                if refine.max_points is not None and seen >= refine.max_points:
                    break
                continue
            n_points, sums = _cell_sums(torch.as_tensor(chunk, device=device), centroids)
            counts += n_points
            centroids += (sums - n_points[:, None] * centroids) / counts.clamp(min=1.0)[:, None]
            seen += len(chunk)
    return centroids


def _cell_sums(points: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    label = assign_cells(points, centroids)
    counts = torch.bincount(label, minlength=len(centroids)).to(TORCH_SOLVE_DTYPE)
    sums = torch.zeros(centroids.shape, dtype=TORCH_DATA_DTYPE, device=centroids.device).index_add_(0, label, points)
    return counts, sums.to(TORCH_SOLVE_DTYPE)


def assign_cells(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    block = rows_per_block((len(centroids) + points.shape[1]) * points.element_size())
    return torch.cat([_nearest(points[start:start + block], centroids) for start in range(0, len(points), block)])


def _nearest(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return sq_distances(points, centroids).argmin(dim=1)


def sq_distances(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """(m, n_cells) |x - c|^2 = |x|^2 + |c|^2 - 2 x.c in float32, with x and c shifted by the same vector (the
    centroids' mean) to limit rounding."""
    shift = centroids.mean(dim=0).to(TORCH_DATA_DTYPE)
    centred = (centroids - shift).to(TORCH_DATA_DTYPE)
    shifted = points - shift
    sq = (shifted * shifted).sum(dim=1)[:, None] + (centred * centred).sum(dim=1)[None, :] - 2.0 * (shifted @ centred.T)
    return sq.clamp(min=0.0)


def cell_masses(chunks, centroids: torch.Tensor, *, config: Config, device: str) -> torch.Tensor:
    """Each cell's share of the data. A cell with c points has a relative mass error of about 1/sqrt(c), so for data
    in random order counting stops once every cell has 1/rel_error^2 points."""
    masses, random_order = config.cells.masses, config.sample.strategy == "prefix"
    counts = torch.zeros(len(centroids), dtype=torch.int64, device=device)
    min_count, seen = math.ceil(1.0 / masses.rel_error ** 2), 0
    for chunk in chunks:
        cells = assign_cells(torch.as_tensor(chunk, device=device), centroids)
        counts += torch.bincount(cells, minlength=len(centroids))
        seen += len(chunk)
        if random_order and int(counts.min()) >= min_count:
            break
        if masses.max_points is not None and seen >= masses.max_points:
            break
    return counts.to(TORCH_SOLVE_DTYPE) / counts.sum()
