"""Shared test helpers: build kernels through the public, config-driven API."""

import numpy as np

from lvm_old.config import load_config
from lvm_old.cells import sq_distances
from lvm_old.graph import choose_kernel, choose_radius


def kernel(centroids: np.ndarray, r: float) -> np.ndarray:
    """Default-strategy kernel between ``centroids`` at radius ``r``."""
    return choose_kernel(sq_distances(centroids, centroids), r=r, config=load_config().graph.kernel, exclude_self=True).K


def region_kernel(centroids: np.ndarray, radius_strategy: str | None = None) -> np.ndarray:
    """Kernel with the radius from ``radius_strategy`` (default: the config default)."""
    overrides = {"graph": {"radius": {"strategy": radius_strategy}}} if radius_strategy else None
    config = load_config(overrides=overrides).graph
    return kernel(centroids, choose_radius(centroids, config=config.radius).r)
