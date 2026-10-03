"""Every benchmarked method, in one place: LVM, UMAP, Laplacian Eigenmaps, Landmark Isomap and t-SNE,
each on GPU and CPU.

A method maps (X, params, seed) to (Z, fit seconds, info). ``params`` are the
method's hyperparameters, normally read from ``tuning/<family>/best.yaml``:

* lvm:  nested lvm config overrides, e.g. {"cells": {"n_cells": 300}, "landmarks": {"n_landmarks": 20}}
* umap: {"n_neighbors": ..., "min_dist": ...}
* le:   {"n_neighbors": ...}
* lisomap: {"n_neighbors": ..., "n_landmarks": ...}
* tsne: {"perplexity": ..., "late_exaggeration": ...}

GPU and CPU versions of a family share hyperparameters (tuned once, on GPU).
Each pair runs the same algorithm on one kind of hardware:

* LVM:  GPU = cuML k-means + torch on CUDA;  CPU = FAISS k-means + torch on CPU
* UMAP: GPU = RAPIDS cuML;                   CPU = umap-learn (the reference implementation)
* LE:   GPU = cuML SpectralEmbedding;        CPU = scikit-learn SpectralEmbedding with the
        AMG eigensolver (pyamg); ARPACK's default shift-invert mode needs a sparse LU
        factorisation whose fill-in is ~10x slower and exhausts memory at 70k.
* Landmark Isomap (de Silva & Tenenbaum, 2004): kNN graph, shortest-path distances from
        random landmarks, then the same Landmark MDS step LVM uses (lvm.embedding.place_landmarks).
        It differs from LVM only in its distances (shortest paths vs grounded voltages),
        so it is the ablation for what voltage distances add.
        GPU = cuML kNN + cuGraph shortest paths;  CPU = scikit-learn kNN + SciPy Dijkstra.
* t-SNE: GPU = cuML TSNE (FFT);  CPU = openTSNE (FFT-accelerated, FIt-SNE).
        Same schedule on both: PCA init, early exaggeration 12 for 250 iterations, then 500
        iterations at the tuned late exaggeration, 3 x perplexity neighbours. Learning rate:
        openTSNE's "auto" (n / exaggeration); cuML n/3 with its adaptive mode off, which would
        otherwise override the neighbour count and so the perplexity
        (diagnostics/tsne_optimizer: n/3 gives cuML's lowest KL divergence).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

TUNING = Path(__file__).resolve().parents[1] / "tuning"


@dataclass(frozen=True)
class Method:
    family: str          # lvm | lvm_pca | umap | le | lisomap | tsne -- shares hyperparameters and a colour
    device: str          # gpu | cpu
    label: str
    embed: Callable[[np.ndarray, dict, int], tuple[np.ndarray, float, dict]]
    tuned_as: str | None = None                          # family whose tuned settings it uses (default: its own)
    overrides: dict = field(default_factory=dict)        # fixed settings on top of those


def _merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        out[key] = _merge(out.get(key, {}), value) if isinstance(value, dict) else value
    return out


def _lvm(gpu: bool):
    def embed(X: np.ndarray, params: dict, seed: int):
        from lvm.config import load_config
        from lvm.pipeline import fit_level
        from lvm.stream import array_source

        overrides = _merge(
            {"compute": {"device": "cuda" if gpu else "cpu", "seed": seed},
             "cells": {"kmeans": {"strategy": "cuml" if gpu else "faiss"}}},
            params,
        )
        config = load_config(overrides=overrides)
        source = array_source(X, config.data.chunk_size)
        t0 = time.perf_counter()
        model = fit_level(source, config)
        fit_s = time.perf_counter() - t0
        Z = np.concatenate(list(model.transform_source(source)))
        return Z, fit_s, {"stages": model.timings, "rho_g": model.rho.rho_g, "reach": model.rho.reach,
                          "rho_rounds": model.rho.n_rounds,
                          "n_landmarks": int(model.V.shape[0]),
                          "dimension": model.dimension.d if model.dimension else None}
    return embed


def _umap(gpu: bool):
    def embed(X: np.ndarray, params: dict, seed: int):
        kw = {"n_components": 2, "n_neighbors": params.get("n_neighbors", 15), "min_dist": params.get("min_dist", 0.1)}
        t0 = time.perf_counter()
        if gpu:
            from cuml.manifold import UMAP

            Z = np.asarray(UMAP(**kw).fit_transform(X.astype(np.float32)), dtype=np.float64)
        else:
            import umap

            # No random_state on either side: fixing it makes umap-learn single-threaded.
            Z = umap.UMAP(**kw).fit_transform(X)
        return Z, time.perf_counter() - t0, {}
    return embed


def _le(gpu: bool):
    def embed(X: np.ndarray, params: dict, seed: int):
        n_neighbors = params.get("n_neighbors", 10)
        t0 = time.perf_counter()
        if gpu:
            from cuml.manifold import SpectralEmbedding

            Z = SpectralEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=seed).fit_transform(X.astype(np.float32))
            Z = np.asarray(Z, dtype=np.float64)
        else:
            from sklearn.manifold import SpectralEmbedding

            Z = SpectralEmbedding(n_components=2, affinity="nearest_neighbors", n_neighbors=n_neighbors,
                                  eigen_solver="amg", random_state=seed, n_jobs=-1).fit_transform(X)
        return Z, time.perf_counter() - t0, {}
    return embed


def _tsne(gpu: bool):
    def embed(X: np.ndarray, params: dict, seed: int):
        perplexity = float(params.get("perplexity", 30.0))
        late = float(params.get("late_exaggeration", 1.0))
        n = X.shape[0]
        t0 = time.perf_counter()
        if gpu:
            from cuml.manifold import TSNE

            Z = TSNE(n_components=2, perplexity=perplexity, n_neighbors=min(int(3 * perplexity), n - 1),
                     early_exaggeration=12.0, late_exaggeration=late, exaggeration_iter=250, max_iter=750,
                     learning_rate_method="none", learning_rate=n / 3.0, init="pca", method="fft",
                     random_state=seed).fit_transform(X.astype(np.float32))
            Z = np.asarray(Z, dtype=np.float64)
        else:
            from openTSNE import TSNE

            Z = np.asarray(TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12,
                                early_exaggeration_iter=250, n_iter=500, exaggeration=late, initialization="pca",
                                negative_gradient_method="fft", n_jobs=-1, random_state=seed).fit(X))
        return Z, time.perf_counter() - t0, {}
    return embed


def _lisomap(gpu: bool):
    def embed(X: np.ndarray, params: dict, seed: int):
        from lvm.embedding import place_landmarks

        n_neighbors = params.get("n_neighbors", 10)
        n_landmarks = min(params.get("n_landmarks", 30), X.shape[0])
        landmarks = np.random.default_rng(seed).choice(X.shape[0], n_landmarks, replace=False)
        t0 = time.perf_counter()
        if gpu:
            D = _lisomap_distances_gpu(X, n_neighbors, landmarks)
        else:
            from scipy.sparse.csgraph import dijkstra
            from sklearn.neighbors import kneighbors_graph

            G = kneighbors_graph(X, n_neighbors, mode="distance", n_jobs=-1)
            D = dijkstra(G, directed=False, indices=landmarks)           # (L, n)
        # Points in a part of the kNN graph no landmark reaches: as far as the farthest known.
        finite = np.isfinite(D)
        n_unreached = int((~finite).any(axis=0).sum())
        D = np.where(finite, D, D[finite].max())
        placement = place_landmarks(D[:, landmarks], n_components=2)
        Z = np.asarray(placement.triangulate(D.T))
        return Z, time.perf_counter() - t0, {"usable_components": placement.usable, "unreached_points": n_unreached}
    return embed


def _lisomap_distances_gpu(X: np.ndarray, n_neighbors: int, landmarks: np.ndarray) -> np.ndarray:
    """(L, n) shortest-path distances from each landmark on the GPU (cuML kNN + cuGraph SSSP)."""
    import cudf
    import cugraph
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    n = X.shape[0]
    dist, idx = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X.astype(np.float32)).kneighbors(X.astype(np.float32))
    dist, idx = cp.asarray(dist)[:, 1:], cp.asarray(idx)[:, 1:]          # drop each point itself
    edges = cudf.DataFrame({"src": cp.repeat(cp.arange(n, dtype=cp.int32), n_neighbors),
                            "dst": idx.ravel().astype(cp.int32), "w": dist.ravel().astype(cp.float32)})
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="w")
    D = np.full((len(landmarks), n), np.inf)
    for i, landmark in enumerate(landmarks):
        out = cugraph.sssp(graph, source=int(landmark))
        reached = out["distance"].to_numpy() < np.finfo(np.float32).max
        D[i, out["vertex"].to_numpy()[reached]] = out["distance"].to_numpy()[reached]
    return D


LOCAL_PCA = {"embedding": {"local_scale": {"strategy": "pca", "pca": {"fill": 0.35}}}}

METHODS: dict[str, Method] = {
    "lvm_gpu": Method("lvm", "gpu", "LVM (GPU)", _lvm(True)),
    "umap_gpu": Method("umap", "gpu", "UMAP (GPU, cuML)", _umap(True)),
    "le_gpu": Method("le", "gpu", "Laplacian Eigenmaps (GPU, cuML)", _le(True)),
    "lvm_cpu": Method("lvm", "cpu", "LVM (CPU)", _lvm(False)),
    "umap_cpu": Method("umap", "cpu", "UMAP (CPU, umap-learn)", _umap(False)),
    "le_cpu": Method("le", "cpu", "Laplacian Eigenmaps (CPU, sklearn + AMG)", _le(False)),
    "lisomap_gpu": Method("lisomap", "gpu", "Landmark Isomap (GPU, cuML + cuGraph)", _lisomap(True)),
    "lisomap_cpu": Method("lisomap", "cpu", "Landmark Isomap (CPU, sklearn + SciPy)", _lisomap(False)),
    "tsne_gpu": Method("tsne", "gpu", "t-SNE (GPU, cuML)", _tsne(True)),
    "tsne_cpu": Method("tsne", "cpu", "t-SNE (CPU, openTSNE)", _tsne(False)),
    # LVM plus a local chart per cell: its points placed by the cell's own 2-D PCA (fill 0.35, chosen in
    # diagnostics/kernels). The single-level stand-in for zooming into each region; uses LVM's tuning.
    "lvm_pca_gpu": Method("lvm_pca", "gpu", "LVM + local PCA (GPU)", _lvm(True), tuned_as="lvm",
                          overrides=LOCAL_PCA),
    "lvm_pca_cpu": Method("lvm_pca", "cpu", "LVM + local PCA (CPU)", _lvm(False), tuned_as="lvm",
                          overrides=LOCAL_PCA),
}


def method_params(name: str) -> dict:
    """The settings a method runs with: its family's tuned settings plus its fixed overrides."""
    m = METHODS[name]
    return _merge(best_params(m.tuned_as or m.family), m.overrides)


def best_params(family: str) -> dict:
    """The tuned hyperparameters for a method family (``tuning/<family>/best.yaml``)."""
    path = TUNING / family / "best.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run tuning/{family}/tune.py first")
    return yaml.safe_load(path.read_text())["params"]
