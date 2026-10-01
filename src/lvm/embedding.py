"""Low-dimensional coordinates from landmark voltage maps (config ``embedding``).

Step (6) of Fig. 7 in the paper. Theorem 12 / Corollary 13 bound a voltage
map between two exponentials in the distance to its landmark, so -log(v) grows
roughly linearly with that distance. The features -log v therefore behave like
distances to the landmarks, and a linear method recovers the geometry from
them. A zeroed voltage (below the threshold tau) means "farther than the
map's support", so features are clipped at -log(tau) instead of going to
infinity.

Configurable choice point ``embedding``; see ``lvm.strategies``. A fitted
embedding maps any voltage vectors -- the cells' or, after
``voltage.extend_voltages``, individual data points' -- to coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from array_api_compat import array_namespace, device

from lvm.config import EmbeddingConfig, LogMdsConfig
from lvm.strategies import resolve


@dataclass(frozen=True)
class LogMdsEmbedding:
    tau: float
    mean: np.ndarray          # (L,) mass-weighted mean feature vector of the cells
    components: np.ndarray    # (n_components, L) principal axes
    cell_coords: np.ndarray   # (n_cells, n_components) coordinates of the cells

    def transform(self, V):
        """Coordinates for voltage vectors ``V`` (L landmarks x m items) -> (m, n_components).

        ``V`` may be a NumPy array or a torch tensor; the result is the same
        kind, on the same device.
        """
        F = _log_features(V, self.tau)
        xp = array_namespace(F)
        mean = xp.asarray(self.mean, dtype=F.dtype, device=device(F))
        components = xp.asarray(self.components, dtype=F.dtype, device=device(F))
        return (F - mean) @ components.T


def fit_embedding(V: np.ndarray, p: np.ndarray, *, config: EmbeddingConfig, tau: float) -> LogMdsEmbedding:
    """Fit the embedding named in ``config.strategy`` to the cells' landmark maps.

    ``V`` is (L landmarks, n cells), typically thresholded; ``p`` the cell
    masses, which weight the fit so each cell counts as much as the data in it.
    """
    strategy, options = resolve(_STRATEGIES, config)
    return strategy(np.asarray(V, dtype=np.float64), np.asarray(p, dtype=np.float64), options=options, tau=tau)


def _log_features(V, tau: float):
    """(m, L) features -log(max(v, tau)) for voltage vectors V (L x m), NumPy or torch."""
    if not hasattr(V, "shape"):
        V = np.asarray(V, dtype=np.float64)
    xp = array_namespace(V)
    return -xp.log(xp.clip(V, min=tau)).T


def _log_mds(V: np.ndarray, p: np.ndarray, *, options: LogMdsConfig, tau: float) -> LogMdsEmbedding:
    """Classical MDS on the Euclidean distances between cells' features.

    Implemented as a weighted eigendecomposition (``np.linalg.eigh``) because
    scikit-learn's PCA takes no sample weights.
    With Euclidean distances classical MDS equals PCA of the features, so the
    fit is a linear projection that applies unchanged to new points. Cells
    are weighted by mass; zero-mass cells don't influence the axes but still
    get coordinates. Axis signs are fixed (largest loading positive) so the
    result is deterministic.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1) for log features, got {tau}")
    F = _log_features(V, tau)                        # (n, L)
    w = p / p.sum()
    mean = w @ F
    C = (F - mean).T @ ((F - mean) * w[:, None])     # (L, L) weighted covariance
    eigvals, eigvecs = np.linalg.eigh(C)
    k = min(options.n_components, F.shape[1])
    components = eigvecs[:, ::-1][:, :k].T           # largest variance first
    signs = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
    components = components * signs[:, None]
    return LogMdsEmbedding(tau=tau, mean=mean, components=components, cell_coords=(F - mean) @ components.T)


_STRATEGIES: dict[str, Callable[..., LogMdsEmbedding]] = {
    "log_mds": _log_mds,
}
