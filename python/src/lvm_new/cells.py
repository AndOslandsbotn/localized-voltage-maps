from __future__ import annotations

import math

import numpy as np
import torch

from lvm_new.config import CellsConfig, MassesConfig, RefineConfig
from lvm_new.config.config import CumlKMeansOptions, LloydKMeansOptions
from lvm_new.compute import TORCH_DATA_DTYPE, TORCH_SOLVE_DTYPE, rows_per_block


def fit_cells(sample: torch.Tensor, *, config: CellsConfig, device: str, seed: int) -> torch.Tensor:
    """(n_cells, n_features) centroids on ``device``: k-means on the sample. A sample of at most n_cells points: one
    cell per point."""
    if len(sample) <= config.n_cells:
        return sample.to(TORCH_SOLVE_DTYPE)
    # k-means on the centred sample: cuML's float32 distances round badly far from the origin, as in _nearest.
    shift = sample.mean(dim=0, dtype=TORCH_SOLVE_DTYPE).to(TORCH_DATA_DTYPE)
    centred = sample - shift
    strategy = config.kmeans.strategy
    if strategy == "auto":
        strategy = "cuml" if device == "cuda" else "lloyd"
    match strategy:
        case "cuml":
            centroids = _cuml(centred, config.n_cells, options=config.kmeans.cuml, seed=seed)
        case "lloyd":
            centroids = _lloyd(centred, config.n_cells, options=config.kmeans.lloyd, seed=seed)
    return torch.as_tensor(centroids, device=device).to(TORCH_SOLVE_DTYPE) + shift.to(TORCH_SOLVE_DTYPE)


def _cuml(sample: torch.Tensor, n_cells: int, *, options: CumlKMeansOptions, seed: int):
    """Lloyd (1982), "Least squares quantization in PCM", on the GPU with RAPIDS cuML; random initialisation, or
    k-means|| (Bahmani et al., 2012, "Scalable k-means++")."""
    import cupy as cp
    from cuml.cluster import KMeans

    km = KMeans(n_clusters=n_cells, init=options.init, max_iter=options.max_iter, tol=options.tol, n_init=1,
                random_state=seed, output_type="cupy")
    return km.fit(cp.asarray(sample)).cluster_centers_      # the sample stays on the GPU: cuML reads torch's memory


def _lloyd(sample: torch.Tensor, n_cells: int, *, options: LloydKMeansOptions, seed: int) -> torch.Tensor:
    """Lloyd (1982), "Least squares quantization in PCM", in torch on the sample's device: random sample points as
    the first centroids, then rounds of moving each centroid to the mean of the points nearest to it. A cell left
    empty keeps its centroid. Faster on the CPU than FAISS and scikit-learn; cuML is faster on the GPU."""
    rows = np.random.default_rng(seed).choice(len(sample), n_cells, replace=False)
    centroids = sample[torch.as_tensor(rows, device=sample.device)].to(TORCH_SOLVE_DTYPE)
    for _ in range(options.iterations):
        counts, sums = _cell_sums(sample, centroids)
        filled = counts > 0
        centroids[filled] = sums[filled] / counts[filled, None]
    return centroids


def refine_cells(chunks, sample: torch.Tensor, centroids: torch.Tensor, *, config: RefineConfig, device: str,
                 skip: int) -> torch.Tensor:
    """The centroids fitted to the whole stream: mini-batch k-means, Sculley (2010), "Web-scale k-means clustering"
    One chunk per update; each centroid becomes the running mean of every point assigned to it so far (step
    1 / count), starting with the sample's points. Holds one chunk and the centroids, whatever the size of the data.
    ``config.passes`` 0: no refinement. ``skip``: points at the start of the first pass that are the sample itself (a
    prefix sample), so they count once."""
    if config.passes == 0:
        return centroids
    centroids = centroids.clone()                                  # updated in place
    counts = torch.zeros(len(centroids), dtype=TORCH_SOLVE_DTYPE, device=device)
    counts += torch.bincount(_labels(sample, centroids), minlength=len(centroids))
    for p in range(config.passes):
        to_skip, seen = (skip if p == 0 else 0), 0
        for chunk in chunks:
            if to_skip:
                drop = min(to_skip, len(chunk))
                chunk, to_skip = chunk[drop:], to_skip - drop
            if config.max_points is not None:
                chunk = chunk[:config.max_points - seen]
            if len(chunk) == 0:
                if config.max_points is not None and seen >= config.max_points:
                    break
                continue
            n_points, sums = _cell_sums(torch.as_tensor(chunk, device=device), centroids)
            counts += n_points
            centroids += (sums - n_points[:, None] * centroids) / counts.clamp(min=1.0)[:, None]
            seen += len(chunk)
    return centroids


def _cell_sums(points: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(counts, sums) in SOLVE_DTYPE: the number of points nearest to each centroid and the sum of those points."""
    label = _labels(points, centroids)
    counts = torch.bincount(label, minlength=len(centroids)).to(TORCH_SOLVE_DTYPE)
    sums = torch.zeros(centroids.shape, dtype=TORCH_DATA_DTYPE, device=centroids.device).index_add_(0, label, points)
    return counts, sums.to(TORCH_SOLVE_DTYPE)


def _labels(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Each point's nearest centroid, a block of points at a time: a block's distance table and shifted points stay
    within the block budget."""
    block = rows_per_block((len(centroids) + points.shape[1]) * points.element_size())
    return torch.cat([_nearest(points[start:start + block], centroids) for start in range(0, len(points), block)])


def _nearest(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Each point's nearest centroid, from |c|^2 - 2 x.c in float32. Far from the origin that rounds badly enough to
    pick the wrong centroid, so points and centroids are first shifted by one float32 vector (the centroids' mean),
    which leaves every distance unchanged."""
    shift = centroids.mean(dim=0).to(TORCH_DATA_DTYPE)
    centred = (centroids - shift).to(TORCH_DATA_DTYPE)
    return ((centred * centred).sum(dim=1)[None, :] - 2.0 * ((points - shift) @ centred.T)).argmin(dim=1)


def cell_masses(chunks, centroids: torch.Tensor, *, config: MassesConfig, device: str,
                random_order: bool) -> torch.Tensor:
    """(n_cells,) share of the data in each cell on ``device``, counted in a streamed pass; sums to 1.

    A cell with c points has a relative mass error of about 1/sqrt(c) (counting statistics), so for data in random
    order the pass stops once every cell has 1/rel_error^2 points; otherwise it reads all the data.
    """
    counts = torch.zeros(len(centroids), dtype=torch.int64, device=device)
    min_count, seen = math.ceil(1.0 / config.rel_error ** 2), 0
    for chunk in chunks:
        counts += torch.bincount(_labels(torch.as_tensor(chunk, device=device), centroids), minlength=len(centroids))
        seen += len(chunk)
        if random_order and int(counts.min()) >= min_count:
            break
        if config.max_points is not None and seen >= config.max_points:
            break
    return counts.to(TORCH_SOLVE_DTYPE) / counts.sum()
