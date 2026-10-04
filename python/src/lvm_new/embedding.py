from __future__ import annotations

import numpy as np

from lvm_new.config import Config


def fit_embedding(maps, config: Config):
    """Landmark MDS: the landmarks placed from their voltage distances, ready to triangulate points."""
    raise NotImplementedError


def fit_chart(region, sample: np.ndarray, config: Config, *, device: str):
    """The local chart: each cell's main directions, from the sample placed by the region's embedding."""
    raise NotImplementedError
