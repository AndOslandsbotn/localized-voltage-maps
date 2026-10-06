from __future__ import annotations

import numpy as np
import torch

from lvm_new.config import DimensionConfig
from lvm_new.compute import TORCH_DATA_DTYPE, TORCH_SOLVE_DTYPE, rows_per_block


def estimate_dimension(sample: torch.Tensor, *, config: DimensionConfig, device: str, seed: int) -> float:
    """The region's intrinsic dimension d̂ (unrounded), by the strategy in ``config``."""
    match config.strategy:
        case "mle":
            return _mle(_subsample(sample, config.sample_size, seed), config.mle.k, device=device)
        case "fixed":
            return config.fixed.d


def _mle(points: torch.Tensor, k: int, *, device: str) -> float:
    """Levina & Bickel (2005), "Maximum likelihood estimation of intrinsic dimension", NIPS 17.

    Each point's estimate is (k - 1) / sum_j log(T_k / T_j) over its neighbour distances T_1 <= ... <= T_k,
    combined by the harmonic mean (MacKay & Ghahramani, 2005): scikit-dimension's MLE with its defaults, on the device.
    """
    distances = _nearest_neighbours(points, k, device=device)
    distances = distances[distances[:, 0] > 0]          # a duplicate point (distance 0) would give log(inf)
    local = (k - 1) / torch.log(distances[:, -1:] / distances).sum(dim=1)
    return float(1.0 / (1.0 / local).mean())


def _subsample(sample: torch.Tensor, size: int, seed: int) -> torch.Tensor:
    if len(sample) <= size:
        return sample
    rows = np.random.default_rng(seed).choice(len(sample), size, replace=False)
    return sample[torch.as_tensor(rows, device=sample.device)]


def _nearest_neighbours(points: torch.Tensor, k: int, *, device: str) -> torch.Tensor:
    """(n, k) distances from each point to its k nearest other points, nearest first, in SOLVE_DTYPE on the device."""
    if device == "cpu":
        from sklearn.neighbors import NearestNeighbors

        distances, _ = NearestNeighbors(n_neighbors=k).fit(points.numpy()).kneighbors()  # no query: itself left out
        return torch.as_tensor(distances).to(TORCH_SOLVE_DTYPE)

    n = len(points)
    exact = points.to(TORCH_SOLVE_DTYPE)
    centred = (exact - exact.mean(dim=0)).to(TORCH_DATA_DTYPE)
    sq_norm = (centred * centred).sum(dim=1)
    indices = torch.empty((n, k + 1), dtype=torch.long, device=device)
    block = rows_per_block(n * centred.element_size())                 # one row of the distance table
    for start in range(0, n, block):
        rows = slice(start, start + block)
        sq_dist = sq_norm[rows, None] + sq_norm[None, :] - 2.0 * (centred[rows] @ centred.T)
        indices[rows] = torch.topk(sq_dist, k + 1, dim=1, largest=False).indices

    own = indices == torch.arange(n, device=device)[:, None]
    indices = torch.gather(indices, 1, torch.argsort(own.to(torch.int8), dim=1, stable=True))[:, :k]

    distances = torch.empty(indices.shape, dtype=TORCH_SOLVE_DTYPE, device=device)
    block = rows_per_block(k * exact.shape[1] * exact.element_size())  # one row's neighbour differences
    for start in range(0, n, block):
        rows = slice(start, start + block)
        distances[rows] = torch.linalg.norm(exact[indices[rows]] - exact[rows, None, :], dim=2)
    return torch.sort(distances, dim=1).values
