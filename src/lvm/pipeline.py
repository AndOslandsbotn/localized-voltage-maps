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
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator

import numpy as np
import torch

from lvm.cells import fit_cells, sq_distances
from lvm.config import Config
from lvm.dimension import Dimension, choose_dimension
from lvm.embedding import LandmarkMdsEmbedding, LogMdsEmbedding, fit_embedding
from lvm.graph import choose_kernel, choose_radius
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
    V: np.ndarray                  # (L, n) landmarks' thresholded voltage maps over the cells
    embedding: LogMdsEmbedding | LandmarkMdsEmbedding
    dimension: Dimension | None = None   # estimated intrinsic dimension of the data
    device: str = "cpu"            # where the per-point steps (voltages, transform) run
    timings: dict[str, float] = field(default_factory=dict)   # seconds per stage

    @cached_property
    def _on_device(self) -> tuple:
        """Centroids, masses and landmark maps as float32 on ``device`` (copied once)."""
        arrays = (self.centroids, self.masses.p, self.V)
        if self.device == "cpu":
            return tuple(np.asarray(a, dtype=np.float32) for a in arrays)
        return tuple(torch.as_tensor(a, dtype=torch.float32, device=self.device) for a in arrays)

    def _voltages(self, X):
        """Thresholded landmark voltages (L, m) at points X, as an array on ``device``.

        Every step runs where the data is: one distance computation serves
        both the kernel and the nearest-cell fallback, then the extension and
        threshold, all without leaving the device.
        """
        cfg = self.config
        C, p, V = self._on_device
        if self.device == "cpu":
            X = np.asarray(X, dtype=np.float32)
            d2 = sq_distances(X, C, dtype=np.float32)
        else:
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
            d2 = sq_distances(X, C)
        Kx = choose_kernel(d2, r=self.r, config=cfg.graph.kernel).K
        VX = extend_voltages(Kx, V, p, config=cfg.extension, rho_g=self.rho.rho_g, nearest=d2.argmin(1))
        return threshold_voltages(VX, cfg.voltage.threshold)

    def voltages(self, X: np.ndarray) -> np.ndarray:
        """Landmark voltages at data points X (m, d) -> (L, m), thresholded."""
        return _to_numpy(self._voltages(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embedding coordinates for data points X (m, d) -> (m, n_components)."""
        return _to_numpy(self.embedding.transform(self._voltages(X)))

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

    dimension = choose_dimension(sample.points, config=cfg.dimension, seed=cfg.compute.seed)
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

    r = choose_radius(root.centroids, config=cfg.graph.radius).r
    K = choose_kernel(sq_distances(root.centroids, root.centroids), r=r, config=cfg.graph.kernel, exclude_self=True).K
    lap("graph")

    sources = choose_sources(p, config=cfg.sources)
    rho = choose_rho_g(K, p, config=cfg.scaling, tau=tau, n_landmarks=n_landmarks, landmarks=cfg.landmarks,
                       sources=sources, device=device)
    lap("scaling")

    V_all = threshold_voltages(solve_grounded_voltage_maps(K, p, rho.rho_g, sources.sets, device=device), tau)
    lap("voltages")

    # reach chooses the landmarks together with rho_g; the other strategies leave it to this step.
    landmarks = rho.landmarks
    if landmarks is None:
        landmarks = choose_landmarks(V_all, p, config=cfg.landmarks, n_landmarks=n_landmarks, cells=sources.cells,
                                     tau=tau, device=device)
    V = V_all[landmarks.indices]
    landmark_cells = sources.cells[landmarks.indices]
    lap("landmarks")

    embedding = fit_embedding(V, p, config=cfg.embedding, tau=tau, landmark_cells=landmark_cells)
    lap("embedding")

    return LevelModel(
        config=cfg,
        centroids=root.centroids,
        masses=masses,
        r=r,
        K=K,
        rho=rho,
        landmarks=landmarks,
        landmark_cells=landmark_cells,
        V=V,
        embedding=embedding,
        dimension=dimension,
        device=device,
        timings=timings,
    )


def _to_numpy(x) -> np.ndarray:
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
