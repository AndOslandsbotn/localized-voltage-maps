from __future__ import annotations

import numpy as np

from lvm_new.config import Config


def choose_scaling(graph, masses: np.ndarray, dimension: float, config: Config, *, device: str) -> tuple[float, np.ndarray]:
    """The ground scaling rho_g and the landmarks (cells), chosen together."""
    raise NotImplementedError
