from __future__ import annotations

import numpy as np

from lvm_new.config import Config


def build_graph(centroids: np.ndarray, config: Config):
    """The graph between cells: each cell's radius, the kernel between cells, every piece connected."""
    raise NotImplementedError
