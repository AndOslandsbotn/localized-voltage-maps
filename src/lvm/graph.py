import numpy as np
from scipy.spatial.distance import cdist


def build_graph(centroids: np.ndarray, r: float) -> np.ndarray:
    c = np.asarray(centroids)
    dist = cdist(c, c, metric="euclidean")
    return dist <= r


def apply_ground_resistance(W: np.ndarray, rho: float = 0.3) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    col_sums = W.sum(axis=0)
    D = np.diag(col_sums + rho)
    return D - W
