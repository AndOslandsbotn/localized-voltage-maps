from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from lvm_new.cells import cell_masses, fit_cells, refine_cells
from lvm_new.config import Config
from lvm_new.data import sample_region
from lvm_new.dimension import estimate_dimension
from lvm_new.embedding import fit_chart, fit_embedding
from lvm_new.graph import build_graph
from lvm_new.voltage import choose_landmarks, voltage_maps


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
    embedding: Any                # Landmark MDS placement (embedding.py)
    chart: Any = None             # local chart, or None (embedding.py)
    children: tuple["RegionModel", ...] = ()

    def transform_chunk(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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

    embedding = fit_embedding(maps, config=config)
    region = RegionModel(region_id, sample.shape[1], dimension, centroids, 
                         masses, radius, rho_g, landmarks, maps, embedding
                         )
    if config.embedding.local_chart != "none":
        region = replace(region, chart=fit_chart(region, sample, config=config, device=device))
    return region
