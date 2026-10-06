from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import torch

from lvm_new.cells import assign_cells, cell_masses, fit_cells, refine_cells
from lvm_new.compute import rows_per_block
from lvm_new.config import Config
from lvm_new.data import sample_region
from lvm_new.dimension import estimate_dimension
from lvm_new.embedding import LandmarkMds, LocalChart, fit_chart, fit_embedding
from lvm_new.graph import build_graph
from lvm_new.voltage import choose_landmarks, point_voltages, voltage_maps


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


def fit_region(chunks, config: Config, *, device: str, region_id: tuple[int, ...], seed: int) -> RegionModel:
    sample = sample_region(chunks, config=config, device=device, seed=seed)
    dimension = estimate_dimension(sample, config=config, device=device, seed=seed)

    centroids = fit_cells(sample, config=config, device=device, seed=seed)
    centroids = refine_cells(chunks, sample, centroids, config=config, device=device)
    masses = cell_masses(chunks, centroids, config=config, device=device)

    keep = masses > 0
    centroids, masses = centroids[keep], masses[keep]
    kernel, radius = build_graph(centroids, sample, config=config, device=device, seed=seed)

    landmarks, rho_g = choose_landmarks(kernel, masses, dimension, config=config)
    maps = voltage_maps(kernel, masses, rho_g, landmarks)

    embedding = fit_embedding(maps, landmarks, config=config)
    region = RegionModel(region_id, sample.shape[1], dimension, centroids, 
                         masses, radius, rho_g, landmarks, maps, embedding
                         )
    if config.embedding.local_chart != "none":
        coordinates = region.transform_chunk(sample, config=config)
        region = replace(region, chart=fit_chart(sample, centroids, coordinates, config=config))
    return region


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

