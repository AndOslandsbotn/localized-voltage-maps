from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from lvm.cells import assign_cells, cell_masses, fit_cells, refine_cells, split_cells
from lvm.compute import rows_per_block
from lvm.config import Config
from lvm.data import sample_region
from lvm.dimension import estimate_dimension
from lvm.embedding import LandmarkMds, LocalChart, fit_chart, fit_embedding
from lvm.graph import build_graph
from lvm.voltage import choose_landmarks, point_voltages, voltage_maps


@dataclass(frozen=True)
class RegionModel:
    id: tuple[int, ...]           # path in the hierarchy; () for the root
    n_features: int
    dimension: float              # estimated intrinsic dimension (d̂)
    centroids: torch.Tensor       # (n_cells, n_features)
    masses: torch.Tensor          # (n_cells,) share of the region's data in each cell, sums to 1
    radius: torch.Tensor          # (n_cells,) each cell's radius (graph.py)
    rho_g: float                  # ground scaling, the landmarks' reach: the larger, the more local the maps
    landmarks: torch.Tensor       # (n_landmarks,) the cell each landmark sits at
    maps: torch.Tensor            # (n_landmarks, n_cells) the landmarks' voltage maps, unthresholded
    embedding: LandmarkMds        # places items from their voltages to the landmarks
    chart: LocalChart | None = None   # places points within their cell; None: no chart
    children: tuple["RegionModel", ...] = ()
    refine_passes: int = 0        # passes over the data the cells' refinement made

    def transform_chunk(self, points: torch.Tensor, *, config: Config) -> torch.Tensor:
        bytes_per_point = 4 * len(self.centroids) * points.element_size()
        block = rows_per_block(bytes_per_point)
        coordinates = []
        for start in range(0, len(points), block):
            block_points = points[start:start + block]
            coordinates.append(self._transform_block(block_points, config))
        return torch.cat(coordinates)

    def _transform_block(self, points: torch.Tensor, config: Config) -> torch.Tensor:
        voltages = point_voltages(points, self.centroids, self.masses, self.radius, 
                                  self.maps, self.rho_g, config=config
                                  )
        coordinates = self.embedding.triangulate(voltages)
        if self.chart is None:
            return coordinates
        return self.chart.place(points, assign_cells(points, self.centroids), coordinates)


def fit_region(chunks, config: Config, *, device: str, region_id: tuple[int, ...], seed: int,
               timer: Callable[[str], AbstractContextManager] | None = None,
               parents: torch.Tensor | None = None) -> RegionModel:
    """``timer(step)``, if given, is entered around each step, e.g. to time it; by default nothing is done. With
    ``parents`` (the parent region's centroids), the cells are split from them instead of fitted from scratch."""
    timer = timer or _no_timer
    with timer("sample"):
        sample = sample_region(chunks, config=config, device=device, seed=seed)
    with timer("dimension"):
        dimension = estimate_dimension(sample, config=config, device=device, seed=seed)

    with timer("cells"):
        if parents is None:
            centroids = fit_cells(sample, config=config, device=device, seed=seed)
        else:
            centroids = split_cells(sample, parents, config=config, device=device, seed=seed)
    with timer("refine"):
        centroids, refine_passes = refine_cells(chunks, sample, centroids, config=config, device=device)
    with timer("masses"):
        masses = cell_masses(chunks, centroids, config=config, device=device)
        keep = masses > 0
        centroids, masses = centroids[keep], masses[keep]
    with timer("graph"):
        kernel, radius = build_graph(centroids, sample, config=config, device=device, seed=seed)

    with timer("landmarks"):
        landmarks, rho_g = choose_landmarks(kernel, masses, dimension, config=config)
    with timer("maps"):
        maps = voltage_maps(kernel, masses, rho_g, landmarks)

    with timer("embedding"):
        embedding = fit_embedding(maps, landmarks, config=config)
    region = RegionModel(region_id, sample.shape[1], dimension, centroids,
                         masses, radius, rho_g, landmarks, maps, embedding, refine_passes=refine_passes)
    if config.embedding.local_chart != "none":
        with timer("chart"):
            coordinates = region.transform_chunk(sample, config=config)
            region = replace(region, chart=fit_chart(sample, centroids, coordinates, config=config))
    return region


def _no_timer(step: str) -> AbstractContextManager:
    return nullcontext()


def fit_region_chart(region: RegionModel, chunks, *, config: Config, device: str) -> RegionModel:
    """The region with a chart fitted from the data in one pass: at most max_points points per cell."""
    max_points, n_cells = config.embedding.chart.max_points, len(region.centroids)
    taken = torch.zeros(n_cells, dtype=torch.long, device=device)
    kept = []
    for chunk in chunks:
        points = torch.as_tensor(chunk, device=device)
        cells = assign_cells(points, region.centroids)
        order = torch.argsort(cells, stable=True)
        counts = torch.bincount(cells, minlength=n_cells)
        rank = torch.arange(len(cells), device=device) - (torch.cumsum(counts, 0) - counts)[cells[order]]
        keep = order[rank + taken[cells[order]] < max_points]
        kept.append(points[keep])
        taken += torch.bincount(cells[keep], minlength=n_cells)
        if bool((taken >= max_points).all()):
            break
    points = torch.cat(kept)
    plain = replace(region, chart=None)
    coordinates = plain.transform_chunk(points, config=config)
    return replace(region, chart=fit_chart(points, region.centroids, coordinates, config=config))


def save_region(region: RegionModel, path: Path) -> None:
    """The region's arrays in one .npz, without its children (each region has its own file)."""
    np.savez(path, **_arrays(region))


def load_region(path: Path, device: str) -> RegionModel:
    with np.load(path, allow_pickle=False) as arrays:
        return _from_arrays(RegionModel, dict(arrays), "", device)


_NESTED = {"embedding": LandmarkMds, "chart": LocalChart}


def _arrays(item, prefix: str = "") -> dict[str, np.ndarray]:
    arrays = {}
    for field in fields(item):
        value = getattr(item, field.name)
        if field.name == "children" or value is None:
            continue
        if is_dataclass(value):
            arrays.update(_arrays(value, f"{prefix}{field.name}."))
        elif isinstance(value, torch.Tensor):
            arrays[prefix + field.name] = value.cpu().numpy()
        else:
            arrays[prefix + field.name] = np.asarray(value)
    return arrays


def _from_arrays(cls, arrays: dict[str, np.ndarray], prefix: str, device: str):
    values = {}
    for field in fields(cls):
        key = prefix + field.name
        if field.name in _NESTED:
            if any(name.startswith(key + ".") for name in arrays):
                values[field.name] = _from_arrays(_NESTED[field.name], arrays, key + ".", device)
        elif field.name == "id":
            values["id"] = tuple(int(i) for i in arrays[key])
        elif key in arrays:
            array = arrays[key]
            values[field.name] = array.item() if array.ndim == 0 else torch.as_tensor(array, device=device)
    return cls(**values)

