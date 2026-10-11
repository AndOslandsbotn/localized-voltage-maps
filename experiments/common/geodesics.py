from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.neighbors import kneighbors_graph


def graph_geodesics(X: np.ndarray, sample: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """Pairwise distances between the sampled points along the data: shortest paths in the k-nearest-neighbour
    graph of all points (condensed, as scipy's pdist); inf between points the graph doesn't connect."""
    graph = kneighbors_graph(X, n_neighbors, mode="distance", n_jobs=-1)
    distances = dijkstra(graph, directed=False, indices=sample)[:, sample]
    return squareform((distances + distances.T) / 2, checks=False)


def rank_correlation(distances: np.ndarray, reference: np.ndarray) -> float:
    """Spearman rank correlation between two sets of pairwise distances, over the pairs both know: 1 if every pair
    that is farther apart than another in the reference is also farther apart in ``distances``."""
    known = np.isfinite(distances) & np.isfinite(reference)
    return float(spearmanr(distances[known], reference[known]).statistic)
