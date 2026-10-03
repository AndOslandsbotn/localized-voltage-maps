"""Single-level Localized Voltage Maps: from a data stream to an embedding.

One level of the hierarchy, run on the whole dataset (the root region):

    sample -> intrinsic dimension -> k-means cells -> streamed masses -> radius
    -> kernel -> rho_g -> candidate voltage maps -> threshold -> landmarks -> embedding

``fit_level`` returns a ``LevelModel`` whose ``transform`` embeds any data
points, streamed in chunks, so the dataset never has to fit in memory. Every
choice is made by the config (see ``config.yaml`` and ``lvm.strategies``).
The recursive, hierarchical driver will run this same sequence per region.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Iterator

import numpy as np
import torch

from lvm.cells import fit_cells, sq_distances
from lvm.config import Config, KernelConfig
from lvm.dimension import Dimension, choose_dimension
from lvm.embedding import LandmarkMdsEmbedding, LocalPca, LocalScale, LogMdsEmbedding, fit_embedding, fit_local_scale
from lvm.graph import adaptive_knn_kernel, choose_kernel, choose_radius
from lvm.landmarks import LandmarkSelection, choose_landmarks, landmark_count
from lvm.regions import CellMasses, Region, estimate_masses, min_count_for, sample_regions
from lvm.scaling import RhoChoice, choose_rho_g
from lvm.stream import ChunkSource
from lvm.voltage import choose_sources, extend_voltages, solve_grounded_voltage_maps, threshold_voltages


@dataclass(frozen=True)
class LevelModel:
    config: Config
    centroids: np.ndarray          # (n, d) cell centroids
    masses: CellMasses             # streamed cell masses; .p sums to 1
    r: float                       # kernel radius
    K: np.ndarray                  # (n, n) kernel between cells
    rho: RhoChoice                 # chosen ground scaling
    landmarks: LandmarkSelection   # .indices are rows of the candidate maps
    landmark_cells: np.ndarray     # (L,) cell each landmark sits at
    V: np.ndarray                  # (L, n) landmarks' voltage maps over the cells, thresholded at tau
    embedding: LogMdsEmbedding | LandmarkMdsEmbedding
    dimension: Dimension | None = None   # estimated intrinsic dimension of the data
    device: str = "cpu"            # where the per-point steps (voltages, transform) run
    timings: dict[str, float] = field(default_factory=dict)   # seconds per stage
    local_scale: LocalScale | LocalPca | None = None   # embedding.local_scale: a second scale per cell
    V_dist: np.ndarray | None = None   # the same maps thresholded at embedding.distance_floor (None: = V)

    @cached_property
    def _on_device(self) -> tuple:
        """Centroids, masses and landmark maps as float32 on ``device`` (copied once)."""
        arrays = (self.centroids, self.masses.p, self.V, self.V if self.V_dist is None else self.V_dist)
        if self.device == "cpu":
            return tuple(np.asarray(a, dtype=np.float32) for a in arrays)
        return tuple(torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in arrays)

    @property
    def _point_kernel(self) -> KernelConfig:
        """Kernel from data points to cells: ``extension.kernel``, or the graph's when that is "graph"."""
        kernel = self.config.graph.kernel
        name = self.config.extension.kernel
        return kernel if name in ("graph", "knn") else replace(kernel, strategy=name)

    @property
    def distance_floor(self) -> float:
        """epsilon: voltages are read as distances down to this (``embedding.distance_floor``, default tau)."""
        floor = self.config.embedding.distance_floor
        return self.config.voltage.threshold if floor is None else floor

    def _voltages(self, X):
        """Landmark voltages (L, m) at points X, thresholded at tau, as an array on ``device``."""
        return self._voltages_and_cells(X)[0]

    def _voltages_and_cells(self, X, *, for_distances: bool = False):
        """Landmark voltages (L, m) at points X, each point's nearest cell, and X, on ``device``.

        Thresholded at tau (support), or with ``for_distances`` at the distance
        floor epsilon, from the maps thresholded the same way. Every step runs
        where the data is: one distance computation serves both the kernel and
        the nearest-cell fallback, then the extension and threshold, all
        without leaving the device.
        """
        cfg = self.config
        C, p, V_tau, V_eps = self._on_device
        V, threshold = (V_eps, self.distance_floor) if for_distances else (V_tau, cfg.voltage.threshold)
        if self.device == "cpu":
            X = np.asarray(X, dtype=np.float32)
            d2 = sq_distances(X, C, dtype=np.float32)
        else:
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
            d2 = sq_distances(X, C)
        if cfg.extension.kernel == "knn":
            Kx = adaptive_knn_kernel(d2, k=cfg.extension.knn.k, sharpness=cfg.extension.knn.sharpness)
        else:
            Kx = choose_kernel(d2, r=self.r, config=self._point_kernel).K
        nearest = d2.argmin(1)
        VX = extend_voltages(Kx, V, p, config=cfg.extension, rho_g=self.rho.rho_g, nearest=nearest)
        return threshold_voltages(VX, threshold), nearest, X

    def voltages(self, X: np.ndarray) -> np.ndarray:
        """Landmark voltages at data points X (m, d) -> (L, m), thresholded."""
        return _to_numpy(self._voltages(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embedding coordinates for data points X (m, d) -> (m, n_components)."""
        Z, cell, Xd = self._embed_on_device(X)
        return _to_numpy(Z if self.local_scale is None else self.local_scale.apply(Z, cell, Xd))

    def _embed_on_device(self, X: np.ndarray):
        """Coordinates before the local scale, each point's nearest cell, and X, all on ``device``."""
        VX, nearest, Xd = self._voltages_and_cells(X, for_distances=True)
        return self.embedding.transform(VX), nearest, Xd

    def transform_source(self, source: ChunkSource) -> Iterator[np.ndarray]:
        """Embed a whole stream, one chunk at a time."""
        for chunk in source():
            yield self.transform(chunk)


def fit_level(source: ChunkSource, config: Config, *, device: str | None = None) -> LevelModel:
    """Fit one level of LVM to the data in ``source``.

    ``device`` defaults to ``config.compute.device``.
    """
    cfg = config
    device = device or cfg.compute.device
    tau = cfg.voltage.threshold
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[name] = now - clock
        clock = now

    root = Region()
    sample = sample_regions(source, root, cfg.cells.sample_size, shuffled=cfg.data.shuffled, seed=cfg.compute.seed)[()]
    lap("sample")

    dimension = choose_dimension(sample.points, config=cfg.dimension, seed=cfg.compute.seed, device=device)
    n_landmarks = landmark_count(dimension.d, config=cfg.landmarks)
    lap("dimension")

    root.centroids = fit_cells(
        sample.points, config=cfg.cells.kmeans, n_cells=cfg.cells.n_cells, seed=cfg.compute.seed
    ).centroids
    lap("cells")

    masses = estimate_masses(
        source, root,
        min_count=min_count_for(cfg.cells.masses.rel_error),
        shuffled=cfg.data.shuffled,
        max_points=cfg.cells.masses.max_points,
        device=device,
    )[()]
    p = masses.p
    lap("masses")

    r = choose_radius(root.centroids, config=cfg.graph.radius, points=sample.points).r
    K = choose_kernel(sq_distances(root.centroids, root.centroids), r=r, config=cfg.graph.kernel, exclude_self=True).K
    lap("graph")

    sources = choose_sources(p, config=cfg.sources)
    rho = choose_rho_g(K, p, config=cfg.scaling, tau=tau, n_landmarks=n_landmarks, landmarks=cfg.landmarks,
                       sources=sources, device=device)
    lap("scaling")

    V_raw = solve_grounded_voltage_maps(K, p, rho.rho_g, sources.sets, device=device)
    V_all = threshold_voltages(V_raw, tau)
    floor = tau if cfg.embedding.distance_floor is None else cfg.embedding.distance_floor
    if not 0.0 < floor <= tau:
        raise ValueError(f"embedding.distance_floor must be in (0, voltage.threshold], got {floor}")
    lap("voltages")

    # reach chooses the landmarks together with rho_g; the other strategies leave it to this step.
    landmarks = rho.landmarks
    if landmarks is None:
        landmarks = choose_landmarks(V_all, p, config=cfg.landmarks, n_landmarks=n_landmarks, cells=sources.cells,
                                     tau=tau, device=device)
    V = V_all[landmarks.indices]
    landmark_cells = sources.cells[landmarks.indices]
    lap("landmarks")

    # Distances are read from the maps down to the distance floor (tau keeps the support and reach roles).
    V_dist = V if floor == tau else threshold_voltages(V_raw[landmarks.indices], floor)
    embedding = fit_embedding(V_dist, p, config=cfg.embedding, tau=floor, landmark_cells=landmark_cells)
    lap("embedding")

    model = LevelModel(
        config=cfg,
        centroids=root.centroids,
        masses=masses,
        r=r,
        K=K,
        rho=rho,
        landmarks=landmarks,
        landmark_cells=landmark_cells,
        V=V,
        V_dist=None if floor == tau else V_dist,
        embedding=embedding,
        dimension=dimension,
        device=device,
        timings=timings,
    )
    if cfg.embedding.local_scale.strategy != "none":
        # The cells' points are needed to measure each cell's spread: embed the sample once.
        chunk = cfg.data.chunk_size
        parts = [model._embed_on_device(sample.points[s:s + chunk]) for s in range(0, len(sample.points), chunk)]
        torch_device = "cpu" if device == "cpu" else device
        Z, cell, Xd = (torch.cat([torch.as_tensor(part[i], device=torch_device) for part in parts]) for i in range(3))
        local = fit_local_scale(Z, cell.to(torch.long), root.centroids.shape[0], config=cfg.embedding.local_scale, X=Xd)
        model = replace(model, local_scale=local)
        lap("local_scale")
    return model


def _to_numpy(x) -> np.ndarray:
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
