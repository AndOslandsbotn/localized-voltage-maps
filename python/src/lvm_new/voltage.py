from __future__ import annotations

import numpy as np

from lvm_new.config import Config


def landmark_maps(graph, masses: np.ndarray, rho_g: float, landmarks: np.ndarray, config: Config, *, device: str):
    """The landmarks' voltage maps over the cells, thresholded for support and for distances."""
    raise NotImplementedError
