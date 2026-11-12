from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist

from lvm.cells import assign_cells
from lvm.compute import TORCH_DATA_DTYPE, TORCH_SOLVE_DTYPE
from lvm.embedding import LocalChart, _local_coordinates
from lvm.graph import _connect, kernel_weights
from lvm.region import RegionModel, load_region


def nested(Z1: np.ndarray, Z2: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Direction A: each region's level-2 map placed in the level-1 frame by the similarity transform (rotation or
    reflection, scale, shift) that best fits its points' level-2 coordinates onto their level-1 coordinates
    (orthogonal Procrustes with scaling)."""
    Z = np.empty_like(Z1)
    for r in np.unique(region):
        inside = region == r
        local, target = Z2[inside] - Z2[inside].mean(0), Z1[inside] - Z1[inside].mean(0)
        U, S, Vt = np.linalg.svd(local.T @ target)
        Z[inside] = S.sum() / np.square(local).sum() * local @ (U @ Vt) + Z1[inside].mean(0)
    return Z


def own_edges(centroids: torch.Tensor, radius: torch.Tensor, config) -> np.ndarray:
    """The edges (i < j) of the graph ``build_graph`` makes from these cells and radii: its kernel's edges, its
    pieces joined, distances computed as it does."""
    centred = (centroids - centroids.mean(dim=0)).to(TORCH_DATA_DTYPE)
    sq_dist = torch.cdist(centred, centred).square().to(TORCH_SOLVE_DTYPE)
    sq_dist = (sq_dist + sq_dist.T) / 2
    sq_dist.fill_diagonal_(0.0)
    sq_radius = radius.to(TORCH_SOLVE_DTYPE).square() * (1 + 1e-9)
    kernel = kernel_weights(sq_dist, torch.maximum(sq_radius[:, None], sq_radius[None, :]), config.graph.kernel)
    kernel.fill_diagonal_(0.0)
    if config.graph.connect:
        _connect(kernel, sq_dist)
    return torch.triu(kernel, 1).nonzero().cpu().numpy()


def stitches(centroids: list[torch.Tensor], radii: list[torch.Tensor]) -> np.ndarray:
    """Edges (i < j, among all regions' cells together) between cells of different regions by the radial kernel's
    rule, d_ij <= max(r_i, r_j), with each cell's radius from its own region."""
    together, radius = torch.cat(centroids).double(), torch.cat(radii).double()
    label = torch.cat([torch.full((len(c),), r, device=together.device) for r, c in enumerate(centroids)])
    linked = torch.cdist(together, together) <= torch.maximum(radius[:, None], radius[None, :]) * (1 + 1e-5)
    return torch.triu(linked & (label[:, None] != label[None, :]), 1).nonzero().cpu().numpy()


def paths(centroids: torch.Tensor, edges: np.ndarray) -> np.ndarray:
    """Shortest paths between all cells along the edges, each as long as the distance between its cells' centres."""
    centres = centroids.double().cpu().numpy()
    lengths = np.linalg.norm(centres[edges[:, 0]] - centres[edges[:, 1]], axis=1)
    graph = csr_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(len(centres), len(centres)))
    pieces = connected_components(graph, directed=False)[0]
    if pieces != 1:
        raise ValueError(f"The cell graph has {pieces} pieces; expected 1")
    return dijkstra(graph, directed=False)


def chart_coordinates(chart: LocalChart, points: torch.Tensor, cells: torch.Tensor) -> np.ndarray:
    """Each point's coordinates in its cell's chart: u = (x - m_c) B_c, along the cell's PCA directions."""
    return _local_coordinates(points.to(chart.bases.dtype), cells, chart.bases, chart.centres).cpu().numpy()


def graph_distances(local: np.ndarray, cell: np.ndarray, centres: np.ndarray,
                    paths: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Direction B: distances between points along the cell graph, with each cell's chart for the part inside a
    cell: from the point to its cell's centre within the cell's chart, the shortest path between the two cells,
    from the other cell's centre to the other point; within the chart between points of the same cell. ``local``
    are the points' chart coordinates, ``centres`` those of each cell's centre in its own chart."""
    offset = np.linalg.norm(local - centres[cell], axis=1)

    def distances(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        along = offset[rows][:, None] + paths[np.ix_(cell[rows], cell[columns])] + offset[columns][None]
        same = cell[rows][:, None] == cell[columns][None]
        return np.where(same, cdist(local[rows], local[columns]), along)

    return distances


def level1_graph(root: RegionModel, X: np.ndarray, config, device: str) -> tuple[Callable, dict]:
    """Direction B on level 1's own graph (the control): the distance function, and the graph's size."""
    points = torch.as_tensor(X, device=device)
    cell = assign_cells(points, root.centroids)
    every = torch.arange(len(root.centroids), device=device)
    edges = own_edges(root.centroids, root.radius, config)
    distances = graph_distances(chart_coordinates(root.chart, points, cell), cell.cpu().numpy(),
                                chart_coordinates(root.chart, root.centroids, every), paths(root.centroids, edges))
    return distances, {"cells": len(root.centroids), "edges": len(edges)}


def level2_graph(folder: Path, X: np.ndarray, region: np.ndarray, config, device: str) -> tuple[Callable, dict]:
    """Direction B on level 2: every region's own graph, stitched to its neighbours by the same rule; each point in
    its region's cell and chart. The distance function, and the graph's size."""
    centroids, radii, centres, own = [], [], [], []
    cell, local, start = np.empty(len(X), dtype=np.int64), None, 0
    for r in np.unique(region):
        model = load_region(folder / "level2" / f"region_{r}.npz", device)
        inside = np.flatnonzero(region == r)
        points = torch.as_tensor(X[inside], device=device)
        mine = assign_cells(points, model.centroids)
        coordinates = chart_coordinates(model.chart, points, mine)
        if local is None:
            local = np.empty((len(X), coordinates.shape[1]), dtype=coordinates.dtype)
        cell[inside], local[inside] = start + mine.cpu().numpy(), coordinates
        every = torch.arange(len(model.centroids), device=device)
        centres.append(chart_coordinates(model.chart, model.centroids, every))
        own.append(start + own_edges(model.centroids, model.radius, config))
        centroids.append(model.centroids)
        radii.append(model.radius)
        start += len(model.centroids)
    across = stitches(centroids, radii)
    distances = graph_distances(local, cell, np.concatenate(centres),
                                paths(torch.cat(centroids), np.concatenate([*own, across])))
    return distances, {"cells": start, "own_edges": sum(len(e) for e in own), "stitches": len(across)}
