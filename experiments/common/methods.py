from __future__ import annotations

import numpy as np
import torch

from lvm import LocalizedVoltageMaps
from lvm.compute import TORCH_SOLVE_DTYPE
from lvm.embedding import classical_mds, triangulate_distances

# Each method maps data X (n, d), float32, to coordinates (n, n_components) as a NumPy array on the host. The baselines'
# settings are those tuned on MNIST's tuning split (highest mean trustworthiness).


def lvm(X: np.ndarray, *, seed: int, n_components: int = 2, config: dict | None = None) -> np.ndarray:
    model = LocalizedVoltageMaps(n_components=n_components, local_chart="none", random_state=seed, config=config)
    return model.fit(X).transform(X)


def lvm_chart(X: np.ndarray, *, seed: int, n_components: int = 2, config: dict | None = None) -> np.ndarray:
    model = LocalizedVoltageMaps(n_components=n_components, local_chart="last", random_state=seed, config=config)
    return model.fit(X).transform(X)


def umap_gpu(X: np.ndarray, *, seed: int, n_components: int = 2, n_neighbors: int = 30, min_dist: float = 0.0,
             build_algo: str = "brute_force_knn") -> np.ndarray:
    from cuml.manifold import UMAP

    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, build_algo=build_algo)
    return np.asarray(umap.fit_transform(X))


def tsne_gpu(X: np.ndarray, *, seed: int, perplexity: float = 15.0, late_exaggeration: float = 2.0) -> np.ndarray:
    from cuml.manifold import TSNE

    n = len(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_neighbors=min(int(3 * perplexity), n - 1),
                early_exaggeration=12.0, late_exaggeration=late_exaggeration, exaggeration_iter=250, max_iter=750,
                learning_rate_method="none", learning_rate=n / 3.0, init="pca", method="fft", random_state=seed)
    return np.asarray(tsne.fit_transform(X))


def tsne_cpu(X: np.ndarray, *, seed: int, n_components: int = 2, perplexity: float = 15.0,
             late_exaggeration: float = 2.0) -> np.ndarray:
    """openTSNE with tsne_gpu's schedule; for more than 2 components (cuML's t-SNE is 2-D only), with Barnes-Hut
    gradients, since the FFT method works in 1-2 dimensions only."""
    from openTSNE import TSNE

    tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=12, early_exaggeration_iter=250,
                n_iter=500, exaggeration=late_exaggeration, initialization="pca",
                negative_gradient_method="fft" if n_components <= 2 else "bh", n_jobs=-1, random_state=seed)
    return np.asarray(tsne.fit(X))


def laplacian_eigenmaps_gpu(X: np.ndarray, *, seed: int, n_components: int = 2, n_neighbors: int = 5) -> np.ndarray:
    from cuml.manifold import SpectralEmbedding

    embedding = SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=seed)
    return np.asarray(embedding.fit_transform(X))


def landmark_isomap_gpu(X: np.ndarray, *, seed: int, n_components: int = 2, n_neighbors: int = 10,
                        n_landmarks: int = 60) -> np.ndarray:
    """Landmark Isomap (de Silva & Tenenbaum, 2004): shortest-path distances in the k-nearest-neighbour graph from
    random landmarks, then Landmark MDS with LVM's own classical MDS and triangulation, so it differs from LVM only
    in its distances. Points no landmark reaches are placed as far as the farthest reached point."""
    landmarks = np.random.default_rng(seed).choice(len(X), min(n_landmarks, len(X)), replace=False)
    distances = _graph_distances_gpu(X, n_neighbors, landmarks)                      # (L, n)
    distances = np.where(np.isfinite(distances), distances, distances[np.isfinite(distances)].max())
    distances = torch.as_tensor(distances, dtype=torch.float64, device="cuda")
    mean_sq, projection = classical_mds(distances[:, landmarks], n_components)
    return triangulate_distances(distances.T, mean_sq, projection).cpu().numpy()


def fit_lvm(X: np.ndarray, *, seed: int, n_components: int = 2, config: dict | None = None) -> LocalizedVoltageMaps:
    return LocalizedVoltageMaps(n_components=n_components, local_chart="none", random_state=seed,
                                config=config).fit(X)


def fit_lvm_chart(X: np.ndarray, *, seed: int, n_components: int = 2,
                  config: dict | None = None) -> LocalizedVoltageMaps:
    return LocalizedVoltageMaps(n_components=n_components, local_chart="last", random_state=seed,
                                config=config).fit(X)


def uniform_masses(chunks, centroids: torch.Tensor, *, config, device: str) -> torch.Tensor:
    return torch.full((len(centroids),), 1.0 / len(centroids), dtype=TORCH_SOLVE_DTYPE, device=device)


def fit_umap_gpu(X: np.ndarray, *, seed: int, n_neighbors: int = 30, min_dist: float = 0.0):
    from cuml.manifold import UMAP

    return UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, build_algo="brute_force_knn").fit(X)


def fit_landmark_isomap_gpu(X: np.ndarray, *, seed: int, n_neighbors: int = 10,
                            n_landmarks: int = 60) -> "LandmarkIsomapGpu":
    return LandmarkIsomapGpu(X, seed=seed, n_neighbors=n_neighbors, n_landmarks=n_landmarks)


class LandmarkIsomapGpu:
    """Landmark Isomap fitted on X, placing new points (de Silva & Tenenbaum, 2004): a new point's distance to a
    landmark is the shortest, over its k nearest fitted points, of its distance to that point plus that point's graph
    distance to the landmark; then LVM's triangulation."""

    def __init__(self, X: np.ndarray, *, seed: int, n_neighbors: int, n_landmarks: int):
        from cuml.neighbors import NearestNeighbors

        landmarks = np.random.default_rng(seed).choice(len(X), min(n_landmarks, len(X)), replace=False)
        distances = _graph_distances_gpu(X, n_neighbors, landmarks)
        distances = np.where(np.isfinite(distances), distances, distances[np.isfinite(distances)].max())
        distances = torch.as_tensor(distances, dtype=torch.float64, device="cuda")
        self.to_landmarks = distances.T.contiguous()
        self.mean_sq, self.projection = classical_mds(distances[:, landmarks], n_components=2)
        self.neighbours = NearestNeighbors(n_neighbors=n_neighbors).fit(_on_gpu(X))

    def transform(self, points: np.ndarray) -> np.ndarray:
        distance, nearest = self.neighbours.kneighbors(_on_gpu(points))
        distance = torch.as_tensor(distance, device="cuda").double()
        nearest = torch.as_tensor(nearest, device="cuda").long()
        via = (distance[:, :, None] + self.to_landmarks[nearest]).min(dim=1).values
        return triangulate_distances(via, self.mean_sq, self.projection).cpu().numpy()


def _graph_distances_gpu(X: np.ndarray, n_neighbors: int, sources: np.ndarray) -> np.ndarray:
    """(len(sources), n) shortest-path distances from each source in the k-nearest-neighbour graph of X (cuML
    neighbours, cuGraph single-source shortest paths); inf where a point is not reached."""
    import cudf
    import cugraph
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    n = len(X)
    points = _on_gpu(X)
    distance, nearest = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(points).kneighbors(points)
    distance, nearest = distance[:, 1:], nearest[:, 1:]                                 # without each point itself
    edges = cudf.DataFrame({"source": cp.repeat(cp.arange(n, dtype=cp.int32), n_neighbors),
                            "target": nearest.ravel().astype(cp.int32), "weight": distance.ravel()})
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(edges, source="source", destination="target", edge_attr="weight")
    result = cp.full((len(sources), n), cp.inf, dtype=cp.float32)
    for i, source in enumerate(sources):
        paths = cugraph.sssp(graph, source=int(source))
        distance = paths["distance"].values
        reached = distance < np.finfo(np.float32).max
        result[i, paths["vertex"].values[reached]] = distance[reached]
    return cp.asnumpy(result).astype(np.float64)


def _on_gpu(X: np.ndarray):
    import cupy as cp

    return cp.asarray(torch.as_tensor(X, device="cuda"))                 # torch copies without cupy's pinned buffer
