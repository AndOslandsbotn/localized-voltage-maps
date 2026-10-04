from __future__ import annotations

import numpy as np

from lvm_new.config import Config


def fit_cells(sample: np.ndarray, chunks, config: Config, *, device: str, seed: int) -> np.ndarray:
    """(n_cells, n_features) centroids: k-means on the sample, then optionally refined over the whole stream."""
    raise NotImplementedError


def cell_masses(chunks, centroids: np.ndarray, config: Config, *, device: str) -> np.ndarray:
    """(n_cells,) share of the data in each cell, counted in one streamed pass; sums to 1."""
    raise NotImplementedError
