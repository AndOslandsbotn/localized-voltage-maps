import numpy as np
import torch

from experiments.hsfv.common.directions import own_edges, stitches
from lvm.config import Config
from lvm.compute import TORCH_SOLVE_DTYPE
from lvm.embedding import fit_embedding
from lvm.region import RegionModel
from lvm.voltage import _largest_passing, _reached, voltage_maps


def neighbour_pairs(regions: dict[int, RegionModel]) -> list[tuple[int, int]]:
    """Pairs of regions joined by at least one edge between their cells (the radial rule, each cell its own
    radius)."""
    ids = sorted(regions)
    across = stitches([regions[r].centroids for r in ids], [regions[r].radius for r in ids])
    owner = np.concatenate([np.full(len(regions[r].centroids), r) for r in ids])
    return sorted({tuple(sorted((int(owner[i]), int(owner[j])))) for i, j in across})


def window(a: RegionModel, b: RegionModel, share_a: float, config: Config) -> RegionModel:
    """A window over two neighbouring regions: their cells together (masses weighted by each region's share
    ``share_a`` and 1 - share_a of the window's points), their own graphs plus the edges between them, their
    landmarks; voltage maps solved on the window's graph at the largest reach where the landmarks still reach the
    data (reached by as many landmarks as a point in one region is), and the library's landmark MDS."""
    centroids = torch.cat([a.centroids, b.centroids])
    radius = torch.cat([a.radius, b.radius])
    masses = torch.cat([share_a * a.masses, (1 - share_a) * b.masses]).to(TORCH_SOLVE_DTYPE)
    offset = len(a.centroids)
    edges = np.concatenate([own_edges(a.centroids, a.radius, config), offset + own_edges(b.centroids, b.radius, config),
                            stitches([a.centroids, b.centroids], [a.radius, b.radius])])
    kernel = torch.zeros((len(centroids), len(centroids)), dtype=TORCH_SOLVE_DTYPE, device=centroids.device)
    rows, columns = torch.as_tensor(edges[:, 0], device=centroids.device), torch.as_tensor(edges[:, 1], device=centroids.device)
    kernel[rows, columns] = kernel[columns, rows] = 1.0
    landmarks = torch.cat([a.landmarks, offset + b.landmarks])
    reach, tau = config.landmarks.reach, config.voltage.threshold
    k = min(len(a.landmarks), len(b.landmarks)) if reach.k is None else min(reach.k, len(landmarks))
    rho_g = _largest_passing(lambda r: _reached(kernel, masses, r, landmarks, tau, reach.share), k, options=reach,
                             guess=min(a.rho_g, b.rho_g))
    maps = voltage_maps(kernel, masses, rho_g, landmarks)
    embedding = fit_embedding(maps, landmarks, config=config)
    return RegionModel(a.id + b.id, a.n_features, 0.5 * (a.dimension + b.dimension), centroids, masses, radius,
                       rho_g, landmarks, maps, embedding)
