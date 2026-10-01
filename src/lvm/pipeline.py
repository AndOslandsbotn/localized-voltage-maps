"""Single-level Localized Voltage Maps: from a data stream to an embedding.

One level of the hierarchy, run on the whole dataset (the root region):

    sample -> k-means cells -> streamed masses -> radius -> kernel -> rho_g
    -> candidate voltage maps -> threshold -> landmarks -> embedding

``fit_level`` returns a ``LevelModel`` whose ``transform`` embeds any data
points, streamed in chunks, so the dataset never has to fit in memory. Every
choice is made by the config (see ``config.yaml`` and ``lvm.strategies``).
The recursive, hierarchical driver will run this same sequence per region.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from lvm.cells import fit_cells, sq_distances
from lvm.config import Config
from lvm.embedding import LogMdsEmbedding, fit_embedding
from lvm.graph import choose_kernel, choose_radius
from lvm.landmarks import LandmarkSelection, choose_landmarks
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
    embedding: LogMdsEmbedding
    timings: dict[str, float] = field(default_factory=dict)   # seconds per stage

    def voltages(self, X: np.ndarray) -> np.ndarray:
        """Landmark voltages at data points X (m, d) -> (L, m), thresholded."""
        cfg = self.config
        # One distance computation (float32 BLAS) serves both the kernel and
        # the nearest-cell fallback.
        d2 = sq_distances(X, self.centroids, dtype=np.float32)
        Kx = choose_kernel(d2, r=self.r, config=cfg.graph.kernel).K
        VX = extend_voltages(
            Kx, self.V, self.masses.p,
            config=cfg.extension, rho_g=self.rho.rho_g, nearest=d2.argmin(axis=1),
        )
        return threshold_voltages(VX, cfg.voltage.threshold)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embedding coordinates for data points X (m, d) -> (m, n_components)."""
        return self.embedding.transform(self.voltages(np.asarray(X, dtype=np.float64)))

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

    km = cfg.cells.kmeans
    root.centroids = fit_cells(
        sample.points, cfg.cells.n_cells,
        max_iter=km.max_iter, tol=km.tol, n_local_trials=km.n_local_trials,
        init_sample_size=km.init_sample_size, seed=cfg.compute.seed, device=device,
    )
    lap("cells")

    masses = estimate_masses(
        source, root,
        min_count=min_count_for(cfg.cells.masses.rel_error),
        shuffled=cfg.data.shuffled,
        max_points=cfg.cells.masses.max_points,
    )[()]
    p = masses.p
    lap("masses")

    r = choose_radius(root.centroids, config=cfg.graph.radius).r
    K = choose_kernel(sq_distances(root.centroids, root.centroids), r=r, config=cfg.graph.kernel, exclude_self=True).K
    lap("graph")

    rho = choose_rho_g(K, p, config=cfg.scaling, tau=tau, n_landmarks=cfg.landmarks.n_landmarks, device=device)
    lap("scaling")

    sources = choose_sources(p, config=cfg.sources)
    V_all = threshold_voltages(solve_grounded_voltage_maps(K, p, rho.rho_g, sources.sets, device=device), tau)
    lap("voltages")

    landmarks = choose_landmarks(V_all, p, config=cfg.landmarks, device=device)
    V = V_all[landmarks.indices]
    lap("landmarks")

    embedding = fit_embedding(V, p, config=cfg.embedding, tau=tau)
    lap("embedding")

    return LevelModel(
        config=cfg,
        centroids=root.centroids,
        masses=masses,
        r=r,
        K=K,
        rho=rho,
        landmarks=landmarks,
        landmark_cells=sources.cells[landmarks.indices],
        V=V,
        embedding=embedding,
        timings=timings,
    )
