from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from lvm_new.cells import cell_masses, fit_cells
from lvm_new.config import Config
from lvm_new.dimension import estimate_dimension
from lvm_new.embedding import fit_chart, fit_embedding
from lvm_new.graph import build_graph
from lvm_new.sample import sample_region
from lvm_new.scaling import choose_scaling
from lvm_new.voltage import landmark_maps


@dataclass(frozen=True)
class RegionModel:
    """The fitted model of one region: its cells, graph, ground scaling, landmarks, voltage maps and embedding."""

    id: tuple[int, ...]           # path in the hierarchy; () for the root
    n_features: int
    dimension: float              # estimated intrinsic dimension (d̂)
    centroids: np.ndarray         # (n_cells, n_features)
    masses: np.ndarray            # (n_cells,) share of the region's data in each cell, sums to 1
    graph: Any                    # kernel between cells and their radii (graph.py)
    rho_g: float                  # ground scaling
    landmarks: np.ndarray         # (n_landmarks,) the cell each landmark sits at
    maps: Any                     # the landmarks' voltage maps (voltage.py)
    embedding: Any                # Landmark MDS placement (embedding.py)
    chart: Any = None             # local chart, or None (embedding.py)
    children: tuple["RegionModel", ...] = ()

    def transform_chunk(self, X: np.ndarray) -> np.ndarray:
        """Coordinates of one chunk of points: voltages at the points, distances, triangulation, then the chart."""
        raise NotImplementedError


def fit_region(chunks, config: Config, *, device: str, region_id: tuple[int, ...], seed: int) -> RegionModel:
    """Fit one region to its data: the method's nine steps, in order."""
    sample = sample_region(chunks, config=config.sample, seed=seed)
    dimension = estimate_dimension(sample, config=config.dimension, device=device, seed=seed)
    centroids = fit_cells(sample, chunks, config, device=device, seed=seed)
    masses = cell_masses(chunks, centroids, config, device=device)
    graph = build_graph(centroids, config)
    rho_g, landmarks = choose_scaling(graph, masses, dimension, config, device=device)
    maps = landmark_maps(graph, masses, rho_g, landmarks, config, device=device)
    embedding = fit_embedding(maps, config)
    region = RegionModel(region_id, sample.shape[1], dimension, 
                         centroids, masses, graph, rho_g, landmarks, maps, embedding)
    if config.embedding.local_chart != "none":
        region = replace(region, chart=fit_chart(region, sample, config, device=device))
    return region
