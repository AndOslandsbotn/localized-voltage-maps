from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.spatial.distance import cdist, squareform

from experiments.common.geodesics import rank_correlation
from experiments.common.record import fitted
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells
from lvm.config import load_config
from lvm.data import as_chunks
from lvm.region import RegionModel, fit_region, fit_region_chart, save_region

MAPS = ("level1", "level1_chart", "level2", "level2_chart")
MIN_FOR_DIMENSION = 100
KS = tuple(int(k) for k in np.geomspace(5, 3000, 15).round())
BANDS = ("close", "middle", "far")
METRICS = ("rank", *(f"rank_{b}" for b in BANDS), *(f"recall@{k}" for k in KS))


def fit_two_levels(X: np.ndarray, *, config: dict, seed: int, models: Path,
                   points_per_cell: float | None = None) -> dict:
    """Paper 1's two levels: LVM on all points, and the same LVM again in each level-1 region (the landmark basins
    of the cells), its cells split from the region's level-1 cells; each level placed with and without the chart.
    Every map of a level-2 region is in that region's own frame. Both levels' models are saved under ``models``.
    With ``points_per_cell``, each level-2 region has as many cells as gives it that many points per cell; else
    the configured number."""
    level1, Z1, Z1_chart = fit_level1(X, config=config, seed=seed, models=models)
    level2 = fit_level2(X, level1, config=config, seed=seed, models=models, points_per_cell=points_per_cell)
    return {"level1": Z1, "level1_chart": Z1_chart, **level2, "level1_fitted": fitted(level1.root_)}


def fit_level2(X: np.ndarray, level1: LocalizedVoltageMaps, *, config: dict, seed: int, models: Path,
               points_per_cell: float | None = None, membership: np.ndarray | None = None) -> dict:
    """Paper 1's level 2: the same LVM again in each level-1 region, its cells split from the region's level-1
    cells, placed with and without the chart, each region in its own frame; the models are saved under
    ``models``/level2. A region is a set of level-1 cells: ``membership`` (regions x level-1 cells, True where a cell
    belongs to a region), by default each cell to its strongest landmark only (the landmark basins). A region is
    fitted on the points of all its cells; a point's own region is its strongest landmark's. With an overlapping
    ``membership``, every region also returns the rows and coordinates of all its points. With
    ``points_per_cell``, each region has as many cells as gives it that many points per cell; else the configured
    number."""
    (models / "level2").mkdir(parents=True, exist_ok=True)
    root, device = level1.root_, level1.device_
    strongest = root.maps.argmax(dim=0).cpu().numpy()
    cell = assign_cells(torch.as_tensor(X, device=device), root.centroids).cpu().numpy()
    region = strongest[cell]
    overlapping = membership is not None
    if not overlapping:
        membership = np.arange(len(root.maps))[:, None] == strongest[None, :]
    k = level1.config_.embedding.n_components
    Z2, Z2_chart = np.zeros((len(X), k), dtype=np.float32), np.zeros((len(X), k), dtype=np.float32)
    regions, covered = {}, {}
    for r in range(len(membership)):
        rows = np.flatnonzero(membership[r][cell])
        if len(rows) == 0:
            continue
        members = np.flatnonzero(region == r)
        dimension = ({} if len(rows) >= MIN_FOR_DIMENSION
                     else {"dimension": {"strategy": "fixed", "fixed": {"d": root.dimension}}})
        cells = {} if points_per_cell is None else {"cells": {"n_cells": round(len(rows) / points_per_cell)}}
        settings = load_config(config, {**dimension, **cells, "embedding": {"local_chart": "none"}})
        chunks = as_chunks(X[rows], settings.data.chunk_size)
        parents = root.centroids[torch.as_tensor(membership[r], device=device)]
        level2 = fit_region(chunks, settings, device=device, region_id=(r,), seed=seed, parents=parents)
        if len(members):
            Z2[members] = place(level2, X[members], settings, device)
        level2 = fit_region_chart(level2, chunks, config=settings, device=device)
        if len(members):
            Z2_chart[members] = place(level2, X[members], settings, device)
        if overlapping:
            covered[r] = {"ids": rows, "level2_chart": place(level2, X[rows], settings, device)}
        save_region(level2, models / "level2" / f"region_{r}.npz")
        regions[r] = fitted(level2)
    result = {"level2": Z2, "level2_chart": Z2_chart, "region": region, "level2_fitted": regions}
    return {**result, "covered": covered} if overlapping else result


def fit_level1(X: np.ndarray, *, config: dict, seed: int,
               models: Path) -> tuple[LocalizedVoltageMaps, np.ndarray, np.ndarray]:
    """Paper 1's LVM on all points, placed without and with the chart; the model is saved under ``models``."""
    level1 = LocalizedVoltageMaps(local_chart="none", random_state=seed, config=config).fit(X)
    Z = level1.transform(X)
    level1.fit_charts(X)
    Z_chart = level1.transform(X)
    level1.save(models / "level1")
    return level1, Z, Z_chart


def place(region: RegionModel, X: np.ndarray, config, device: str) -> np.ndarray:
    return region.transform_chunk(torch.as_tensor(X, device=device), config=config).cpu().numpy()


def score_regions(X: np.ndarray, maps: dict[str, np.ndarray | Callable[[np.ndarray, np.ndarray], np.ndarray]],
                  region: np.ndarray,
                  truth: Callable[[np.ndarray, np.ndarray], np.ndarray], *, sample: int, seed: int,
                  min_points: int = 0) -> list[dict]:
    """Each map (coordinates, or a function giving the distances from some points to others by their rows)
    against the true distances inside each region, on at most ``sample`` of the region's points, at
    growing scales: recall of the k true nearest neighbours among all the region's points for each k in KS (NaN
    where k is not smaller than the region), and rank correlation of the pairwise distances, over all pairs and over
    the close, middle and far third of them. Regions with fewer than ``min_points`` points are left out."""
    rng, rows = np.random.default_rng(seed), []
    for r in np.unique(region):
        members = np.flatnonzero(region == r)
        if len(members) < min_points:
            continue
        ks = [k for k in KS if k < len(members)]
        queries = rng.choice(len(members), min(sample, len(members)), replace=False)
        true = truth(X[members][queries], X[members])
        exact = squareform(true[:, queries], checks=False)
        band = np.searchsorted(np.quantile(exact, [1 / 3, 2 / 3]), exact)
        true_rank = np.full(true.shape, max(ks), dtype=np.int32)
        np.put_along_axis(true_rank, nearest(true, queries, max(ks)), np.arange(max(ks), dtype=np.int32)[None], axis=1)
        for name, Z in maps.items():
            embedded = Z(members[queries], members) if callable(Z) else cdist(Z[members][queries], Z[members])
            pairs = squareform(embedded[:, queries], checks=False)
            found_rank = np.take_along_axis(true_rank, nearest(embedded, queries, max(ks)), axis=1)
            row = {"region": int(r), "points": len(members), "map": name, "rank": rank_correlation(pairs, exact)}
            row |= {f"rank_{b}": rank_correlation(pairs[band == i], exact[band == i]) for i, b in enumerate(BANDS)}
            row |= {f"recall@{k}": recall(found_rank, k) if k in ks else np.nan for k in KS}
            rows.append(row)
    return rows


def nearest(distances: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Each query's k nearest points, nearest first, itself excluded."""
    distances = distances.copy()
    distances[np.arange(len(queries)), queries] = np.inf
    candidates = np.argpartition(distances, k, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(distances, candidates, axis=1), axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def recall(found_rank: np.ndarray, k: int) -> float:
    """Share of each query's k nearest in the map that are among its k true nearest, from their true ranks."""
    return float(np.mean(found_rank[:, :k] < k))


def weighted(rows: list[dict], name: str, metric: str) -> float:
    """The metric's average over the regions that have it, each weighted by its number of points."""
    chosen = [row for row in rows if row["map"] == name and np.isfinite(row[metric])]
    return float(np.average([row[metric] for row in chosen], weights=[row["points"] for row in chosen]))
