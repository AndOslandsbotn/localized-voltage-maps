from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from lvm.cells import assign_cells
from lvm.compute import rows_per_block
from lvm.config import Config

_EIGENVALUE_FLOOR = 1e-6    # relative to the largest: axes below it are set to 0


@dataclass(frozen=True)
class LandmarkMds:
    distances: torch.Tensor     # (L, L) landmark-to-landmark distances, symmetric and complete
    mean_sq: torch.Tensor       # (L,) each landmark's mean squared distance to the landmarks
    projection: torch.Tensor    # (n_components, L) the landmarks' MDS eigenvectors divided by sqrt(eigenvalue)
    floor: float                # voltages below it give no distance

    def triangulate(self, voltages: torch.Tensor) -> torch.Tensor:
        """Triangulation (de Silva & Tenenbaum, 2004): an item with squared distances d^2 to the landmarks is placed
        at y = -1/2 P (d^2 - mean_sq), exact when the distances are Euclidean. ``voltages`` is (m, L)."""
        d = self.item_distances(voltages)
        return -0.5 * (d * d - self.mean_sq.to(d.dtype)) @ self.projection.to(d.dtype).T

    def item_distances(self, voltages: torch.Tensor) -> torch.Tensor:
        """(m, L) distances d = -log v to the landmarks. Below the floor, through the landmarks the item reaches:
        d(x, l) = min_j d(x, j) + D(j, l); -log(floor) if it reaches none."""
        d = _voltage_distances(voltages, self.floor)
        via = (d[:, :, None] + self.distances.to(d.dtype)[None, :, :]).min(dim=1).values
        d = torch.where(torch.isfinite(d), d, via)
        return torch.where(torch.isfinite(d), d, -math.log(self.floor))


def fit_embedding(maps: torch.Tensor, landmarks: torch.Tensor, *, config: Config) -> LandmarkMds:
    floor = config.embedding.distance_floor
    distances = _voltage_distances(maps[:, landmarks], floor)       # row a: landmark a's distance to each landmark
    distances = _reconcile(distances)
    distances = _chain(distances)
    distances = _fill_unreachable(distances, floor)
    mean_sq, projection = _classical_mds(distances, config.embedding.n_components)
    return LandmarkMds(distances, mean_sq, projection, floor)


def _voltage_distances(voltages: torch.Tensor, floor: float) -> torch.Tensor:
    """d = -log v where the voltage is at least the floor; below it the distance is unknown (inf)."""
    known = voltages >= floor
    distances = -torch.log(voltages.clamp(min=floor))
    return torch.where(known, distances, torch.inf)


def _reconcile(distances: torch.Tensor) -> torch.Tensor:
    """Two landmarks generally disagree about their mutual distance, so we force them to agree. Each reads it from its
    own map: distances[a, b] = -log v_a(b), a's voltage at b's cell, and distances[b, a] = -log v_b(a), b's voltage
    at a's cell. Both are replaced by their mean, or by the one that is known if the other is below the floor."""
    both = torch.isfinite(distances) & torch.isfinite(distances.T)
    symmetric = torch.where(both, (distances + distances.T) / 2, torch.minimum(distances, distances.T))
    return symmetric.fill_diagonal_(0.0)


def _chain(distances: torch.Tensor) -> torch.Tensor:
    """Unknown distances by the shortest path through known ones (Floyd-Warshall)."""
    for via in range(len(distances)):
        distances = torch.minimum(distances, distances[:, via, None] + distances[None, via, :])
    return distances


def _fill_unreachable(distances: torch.Tensor, floor: float) -> torch.Tensor:
    """Landmarks in parts of the graph no path joins: as far as anything can be."""
    known = distances[torch.isfinite(distances)]
    return torch.where(torch.isfinite(distances), distances, max(-math.log(floor), float(known.max())))


def _classical_mds(distances: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Classical MDS: B = -1/2 H D^2 H with H = I - 1 1^T / L is the Gram matrix of the landmarks' centred positions.
    With B = V Lambda V^T, the landmarks sit at V_k Lambda_k^(1/2); the projection P = Lambda_k^(-1/2) V_k^T is its
    pseudo-inverse. Returns (the mean squared distance of each landmark, P). An axis with an eigenvalue below
    _EIGENVALUE_FLOOR of the largest (landmarks nearly in a lower dimension) is set to 0."""
    sq = distances * distances
    centring = torch.eye(len(sq), dtype=sq.dtype, device=sq.device) - 1.0 / len(sq)
    eigenvalues, eigenvectors = torch.linalg.eigh(-0.5 * centring @ sq @ centring)
    eigenvalues, eigenvectors = eigenvalues.flip(0)[:n_components], eigenvectors.flip(1)[:, :n_components]
    eigenvectors = eigenvectors * torch.sign(eigenvectors.gather(0, eigenvectors.abs().argmax(0, keepdim=True)))
    usable = eigenvalues > _EIGENVALUE_FLOOR * eigenvalues[0].clamp(min=0.0)
    projection = torch.where(usable, eigenvectors / eigenvalues.clamp(min=1e-300).sqrt(), 0.0).T
    return sq.mean(dim=0), projection


@dataclass(frozen=True)
class LocalChart:
    anchors: torch.Tensor       # (n_cells, k) each cell's mean coordinate
    bases: torch.Tensor         # (n_features, n_cells k): cell c's principal directions in columns c k ... c k + k - 1
    centres: torch.Tensor       # (n_cells, k) each cell's mean, projected on its directions
    rotations: torch.Tensor     # (n_cells, k, k)
    scales: torch.Tensor        # (n_cells,)
    valid: torch.Tensor         # (n_cells,) cells with a chart
    origin: str                 # anchor | point

    def place(self, points: torch.Tensor, cells: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """y = o + s_c u R_c, with the point's local coordinates u = (x - m_c) B_c in its cell c and o the cell's
        anchor (origin "anchor") or the point's own coordinates ("point"). Points of cells without a chart keep their
        coordinates."""
        local = _local_coordinates(points, cells, self.bases, self.centres)
        offsets = self.scales[cells, None] * torch.einsum("mk,mkj->mj", local, self.rotations[cells])
        origins = self.anchors[cells] if self.origin == "anchor" else coordinates
        return torch.where(self.valid[cells, None], origins + offsets, coordinates)


def fit_chart(points: torch.Tensor, centroids: torch.Tensor, coordinates: torch.Tensor, *,
              config: Config) -> LocalChart:
    chart, k = config.embedding.chart, coordinates.shape[1]
    cells = assign_cells(points, centroids)
    counts = torch.bincount(cells, minlength=len(centroids))
    anchors = _cell_means(coordinates, cells, counts)
    means, directions = _principal_directions(points, cells, counts, k, chart.max_points)
    bases = directions.permute(1, 0, 2).reshape(points.shape[1], -1)
    centres = torch.einsum("nd,ndk->nk", means, directions)
    local = _local_coordinates(points, cells, bases, centres)
    rotations = _procrustes(local, coordinates - anchors[cells], cells, len(centroids))
    scales = _scales(local, cells, counts, anchors, chart.fill)
    valid = (counts > k) & torch.isfinite(scales) & (scales > 0)
    dtype = points.dtype                                                               # the data path's precision
    scales = scales.nan_to_num(0.0, 0.0, 0.0).to(dtype)
    return LocalChart(anchors.to(dtype), bases, centres, rotations.to(dtype), scales, valid, chart.origin)


def _cell_means(values: torch.Tensor, cells: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    sums = torch.zeros((len(counts), values.shape[1]), dtype=values.dtype, device=values.device)
    return sums.index_add_(0, cells, values) / counts.clamp(min=1)[:, None].to(values.dtype)


def _principal_directions(points: torch.Tensor, cells: torch.Tensor, counts: torch.Tensor, k: int,
                          max_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each cell's mean and k principal directions, from at most ``max_points`` of its points. With P the cell's
    centred points (rows), the top eigenvectors e of the small Gram matrix P P^T give the directions P^T e / |P^T e|,
    the same as the top eigenvectors of P^T P."""
    n_cells, n_features = len(counts), points.shape[1]
    means = _cell_means(points, cells, counts)
    order = torch.argsort(cells, stable=True)
    rank = torch.arange(len(cells), device=cells.device) - (torch.cumsum(counts, 0) - counts)[cells[order]]
    rows, slots = order[rank < max_points], rank[rank < max_points]
    directions = torch.zeros((n_cells, n_features, k), dtype=points.dtype, device=points.device)
    block = rows_per_block(2 * max_points * n_features * points.element_size())        # cells at a time
    for start in range(0, n_cells, block):
        stop = min(start + block, n_cells)
        in_block = (cells[rows] >= start) & (cells[rows] < stop)
        block_rows, block_cells = rows[in_block], cells[rows[in_block]]
        centred = torch.zeros((stop - start, max_points, n_features), dtype=points.dtype, device=points.device)
        centred[block_cells - start, slots[in_block]] = points[block_rows] - means[block_cells]
        _, vectors = torch.linalg.eigh(centred @ centred.transpose(1, 2))
        top = centred.transpose(1, 2) @ vectors[:, :, -k:].flip(-1)                    # largest first
        directions[start:stop] = top / top.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return means, directions


def _local_coordinates(points: torch.Tensor, cells: torch.Tensor, bases: torch.Tensor,
                       centres: torch.Tensor) -> torch.Tensor:
    """u = (x - m_c) B_c for each point's own cell c: one product with every cell's directions, then its cell's."""
    k = centres.shape[1]
    block = rows_per_block(bases.shape[1] * points.element_size())
    local = []
    for start in range(0, len(points), block):
        block_points, block_cells = points[start:start + block], cells[start:start + block]
        projected = (block_points @ bases).view(len(block_points), -1, k)
        rows = torch.arange(len(block_points), device=points.device)
        local.append(projected[rows, block_cells] - centres[block_cells])
    return torch.cat(local)


def _procrustes(local: torch.Tensor, offsets: torch.Tensor, cells: torch.Tensor, n_cells: int) -> torch.Tensor:
    """Per cell, the rotation or reflection R minimising sum |u R - o|^2 over its points (orthogonal Procrustes):
    with M = sum u^T o = A S B^T, R = A B^T."""
    k = local.shape[1]
    products = local.double()[:, :, None] * offsets.double()[:, None, :]
    M = torch.zeros((n_cells, k, k), dtype=torch.float64, device=local.device).index_add_(0, cells, products)
    A, _, Bt = torch.linalg.svd(M)
    return A @ Bt


def _scales(local: torch.Tensor, cells: torch.Tensor, counts: torch.Tensor, anchors: torch.Tensor,
            fill: float) -> torch.Tensor:
    """s_c = fill x (the distance from the cell's anchor to the nearest other anchor) / (the median |u| of its points),
    so the cell's median point lands at fill x that distance. inf or nan for a cell this can't be computed for."""
    radius = local.norm(dim=1).double()
    by_radius = torch.argsort(radius)
    order = by_radius[torch.argsort(cells[by_radius], stable=True)]                    # by cell, then by radius
    first = torch.cumsum(counts, 0) - counts
    median = torch.full((len(counts),), torch.nan, dtype=torch.float64, device=local.device)
    occupied = counts > 0
    median[occupied] = radius[order[(first + counts // 2)[occupied]]]
    gaps = torch.cdist(anchors.double(), anchors.double())
    gaps.fill_diagonal_(torch.inf)
    gaps[:, ~occupied] = torch.inf
    return fill * gaps.min(dim=1).values / median
