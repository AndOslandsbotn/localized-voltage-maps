from __future__ import annotations

import math

import numpy as np
import torch

from lvm.config import Config
from lvm.config.config import CumlKMeansOptions, LloydKMeansOptions, RefineConfig
from lvm.compute import TORCH_DATA_DTYPE, TORCH_SOLVE_DTYPE, rows_per_block


def fit_cells(sample: torch.Tensor, *, config: Config, device: str, seed: int) -> torch.Tensor:
    if len(sample) <= config.cells.n_cells:
        return sample.to(TORCH_SOLVE_DTYPE)
    # Centred, so float32 distances stay accurate far from the origin.
    shift = sample.mean(dim=0, dtype=TORCH_SOLVE_DTYPE).to(TORCH_DATA_DTYPE)
    centred = sample - shift
    if config.cells.placement == "farthest":
        centroids = _farthest(centred, config.cells.n_cells, seed=seed)
        return centroids.to(TORCH_SOLVE_DTYPE) + shift.to(TORCH_SOLVE_DTYPE)
    strategy = config.cells.kmeans.strategy
    if strategy == "auto":
        strategy = "cuml" if device == "cuda" else "lloyd"
    match strategy:
        case "cuml":
            centroids = _cuml(centred, config.cells.n_cells, options=config.cells.kmeans.cuml, seed=seed)
        case "lloyd":
            centroids = _lloyd(centred, config.cells.n_cells, options=config.cells.kmeans.lloyd, seed=seed)
    return torch.as_tensor(centroids, device=device).to(TORCH_SOLVE_DTYPE) + shift.to(TORCH_SOLVE_DTYPE)


def _farthest(sample: torch.Tensor, n_cells: int, *, seed: int) -> torch.Tensor:
    """Farthest-point sampling (Gonzalez, 1985): from a random sample point, repeatedly add the sample point farthest
    from the centroids chosen so far."""
    rows = [int(np.random.default_rng(seed).integers(len(sample)))]
    nearest = sq_distances(sample, sample[rows[0]][None, :])[:, 0]
    for _ in range(n_cells - 1):
        rows.append(int(torch.argmax(nearest)))
        nearest = torch.minimum(nearest, sq_distances(sample, sample[rows[-1]][None, :])[:, 0])
    return sample[torch.as_tensor(rows, device=sample.device)]


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
                 device: str) -> tuple[torch.Tensor, int]:
    """The refined centroids and the number of passes over the data that were made."""
    refine = config.cells.refine
    if refine.passes == 0:
        return centroids, 0
    match refine.strategy:
        case "minibatch":
            return _minibatch(chunks, sample, centroids, config=config, device=device), refine.passes
        case "lloyd":
            return _streamed_lloyd(chunks, centroids, options=refine, device=device)


def _minibatch(chunks, sample: torch.Tensor, centroids: torch.Tensor, *, config: Config, device: str) -> torch.Tensor:
    """Mini-batch k-means (Sculley, 2010, "Web-scale k-means clustering"): after each chunk, every centroid is the
    running mean of all points assigned to it so far, the sample's included. With a prefix sample, the sample is the
    start of the stream; it is skipped in the first pass so it counts once."""
    refine, prefix = config.cells.refine, config.sample.strategy == "prefix"
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


def _streamed_lloyd(chunks, centroids: torch.Tensor, *, options: RefineConfig,
                    device: str) -> tuple[torch.Tensor, int]:
    """Lloyd's algorithm (Lloyd, 1982) over the stream: each pass assigns every point to its nearest centroid, then
    moves every centroid to the mean of its points; an empty cell keeps its centroid. Stops once no centroid moved
    more than ``tolerance`` times its cell's RMS radius, sqrt(mean |x - c|^2) over its points, or after ``passes``."""
    centroids = centroids.to(TORCH_SOLVE_DTYPE, copy=True)
    for done in range(1, options.passes + 1):
        counts = torch.zeros(len(centroids), dtype=TORCH_SOLVE_DTYPE, device=device)
        sums = torch.zeros(centroids.shape, dtype=TORCH_SOLVE_DTYPE, device=device)
        sq_norms = torch.zeros(len(centroids), dtype=TORCH_SOLVE_DTYPE, device=device)
        seen = 0
        for chunk in chunks:
            if options.max_points is not None:
                chunk = chunk[:options.max_points - seen]
                if len(chunk) == 0:
                    break
            points = torch.as_tensor(chunk, device=device)
            n_points, chunk_sums, chunk_sq_norms = _cell_moments(points, centroids)
            counts += n_points
            sums += chunk_sums
            sq_norms += chunk_sq_norms
            seen += len(chunk)
        filled = counts > 0
        mean_sq = (sq_norms - 2.0 * (centroids * sums).sum(dim=1) + counts * (centroids * centroids).sum(dim=1))
        radius = (mean_sq.clamp(min=0.0) / counts.clamp(min=1.0)).sqrt()
        moved = centroids.clone()
        moved[filled] = sums[filled] / counts[filled, None]
        shift = (moved - centroids).norm(dim=1)
        centroids = moved
        if bool((shift <= options.tolerance * radius).all()):
            return centroids, done
    return centroids, options.passes


def _cell_moments(points: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per cell: the number of points, their sum, and the sum of their squared norms."""
    counts, sums = _cell_sums(points, centroids)
    label = assign_cells(points, centroids)
    exact = points.to(TORCH_SOLVE_DTYPE)
    sq_norms = torch.zeros(len(centroids), dtype=TORCH_SOLVE_DTYPE, device=centroids.device)
    return counts, sums, sq_norms.index_add_(0, label, (exact * exact).sum(dim=1))


def split_cells(sample: torch.Tensor, parents: torch.Tensor, *, config: Config, device: str,
                seed: int) -> torch.Tensor:
    """Child cells inside parent cells: each parent gets children in proportion to its points, placed by k-means on
    that parent's points. A small sample gets at most n_points / min_points_per_cell cells (at least 3)."""
    cells = config.cells
    n_cells = min(cells.n_cells, max(3, len(sample) // cells.min_points_per_cell))
    if len(sample) <= n_cells:
        return sample.to(TORCH_SOLVE_DTYPE)
    shift = sample.mean(dim=0, dtype=TORCH_SOLVE_DTYPE).to(TORCH_DATA_DTYPE)
    centred = sample - shift
    owner = assign_cells(sample, parents.to(device))
    children = _allocate(torch.bincount(owner, minlength=len(parents)).cpu().numpy(), n_cells)
    centroids = []
    for parent in np.flatnonzero(children):
        points = centred[owner == int(parent)]
        if children[parent] == len(points):
            centroids.append(points.to(TORCH_SOLVE_DTYPE))
        else:
            centroids.append(_lloyd(points, int(children[parent]), options=cells.kmeans.lloyd, seed=seed + int(parent)))
    return torch.cat(centroids) + shift.to(TORCH_SOLVE_DTYPE)


def _allocate(counts: np.ndarray, budget: int) -> np.ndarray:
    """Largest-remainder apportionment of ``budget`` cells in proportion to ``counts``: at most counts[j] each, and
    at least one for every parent with points when the budget allows."""
    has = counts > 0
    share = budget * counts / counts.sum()
    low = has.astype(int) if budget >= has.sum() else np.zeros_like(counts)
    allocation = np.clip(np.floor(share).astype(int), low, counts)
    while allocation.sum() < budget:
        room = np.where(allocation < counts, share - allocation, -np.inf)
        allocation[np.argmax(room)] += 1
    while allocation.sum() > budget:
        spare = np.where(allocation > low, share - allocation, np.inf)
        allocation[np.argmin(spare)] -= 1
    return allocation


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
    in random order counting stops once every cell has 1/rel_error^2 points; with no rel_error every point counts."""
    masses, random_order = config.cells.masses, config.sample.strategy == "prefix"
    counts = torch.zeros(len(centroids), dtype=torch.int64, device=device)
    min_count = None if masses.rel_error is None else math.ceil(1.0 / masses.rel_error ** 2)
    seen = 0
    for chunk in chunks:
        cells = assign_cells(torch.as_tensor(chunk, device=device), centroids)
        counts += torch.bincount(cells, minlength=len(centroids))
        seen += len(chunk)
        if random_order and min_count is not None and int(counts.min()) >= min_count:
            break
        if masses.max_points is not None and seen >= masses.max_points:
            break
    return counts.to(TORCH_SOLVE_DTYPE) / counts.sum()
