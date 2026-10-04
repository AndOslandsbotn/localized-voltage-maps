from __future__ import annotations

import numpy as np
import torch

from lvm_new.config import CellsConfig, Config, RefineConfig
from lvm_new.config.config import CumlKMeansOptions, FaissKMeansOptions, SklearnKMeansOptions, StreamRefineOptions
from lvm_new.precision import DATA_DTYPE, SOLVE_DTYPE

_TORCH_DATA, _TORCH_SOLVE = (getattr(torch, np.dtype(t).name) for t in (DATA_DTYPE, SOLVE_DTYPE))


def fit_cells(sample: np.ndarray, *, config: CellsConfig, device: str, seed: int) -> np.ndarray:
    """(n_cells, n_features) centroids: k-means on the sample. A sample of at most n_cells points: one cell per point."""
    if len(sample) <= config.n_cells:
        return np.asarray(sample, dtype=SOLVE_DTYPE)
    name = config.kmeans.strategy
    if name == "auto":
        name = "cuml" if device == "cuda" else "faiss"
    centroids = _KMEANS[name](sample, options=getattr(config.kmeans, name), n_cells=config.n_cells, seed=seed)
    return np.asarray(centroids, dtype=SOLVE_DTYPE)


def _cuml(sample: np.ndarray, *, options: CumlKMeansOptions, n_cells: int, seed: int) -> np.ndarray:
    """Lloyd (1982), "Least squares quantization in PCM", on the GPU with RAPIDS cuML; random initialisation, or
    k-means|| (Bahmani et al., 2012, "Scalable k-means++")."""
    from cuml.cluster import KMeans

    km = KMeans(n_clusters=n_cells, init=options.init, max_iter=options.max_iter, tol=options.tol, n_init=1,
                random_state=seed)
    return km.fit(np.asarray(sample, dtype=DATA_DTYPE)).cluster_centers_


def _faiss(sample: np.ndarray, *, options: FaissKMeansOptions, n_cells: int, seed: int) -> np.ndarray:
    """k-means on the CPU, multi-threaded, with FAISS (Johnson, Douze & Jégou, 2019, "Billion-scale similarity search
    with GPUs")."""
    import faiss

    # max_points_per_centroid: train on the whole sample, don't let FAISS subsample it.
    km = faiss.Kmeans(sample.shape[1], n_cells, niter=options.niter, seed=seed, max_points_per_centroid=len(sample),
                      verbose=False)
    km.train(np.ascontiguousarray(sample, dtype=DATA_DTYPE))
    return km.centroids


def _sklearn(sample: np.ndarray, *, options: SklearnKMeansOptions, n_cells: int, seed: int) -> np.ndarray:
    """scikit-learn's k-means, the reference implementation: k-means++ initialisation (Arthur & Vassilvitskii, 2007,
    "k-means++: the advantages of careful seeding"), then Lloyd's iterations."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_cells, n_init=1, max_iter=options.max_iter, tol=options.tol, random_state=seed)
    return km.fit(np.asarray(sample, dtype=DATA_DTYPE)).cluster_centers_


def refine_cells(chunks, sample: np.ndarray, centroids: np.ndarray, *, config: RefineConfig, device: str,
                 skip: int) -> np.ndarray:
    """The centroids fitted to the whole stream, by the strategy in ``config``. ``skip``: points at the start of the
    first pass that are the sample itself (a prefix sample), so they count once."""
    return _REFINE[config.strategy](chunks, sample, centroids, options=config.options, device=device, skip=skip)


def _none(chunks, sample, centroids, *, options: None, device: str, skip: int) -> np.ndarray:
    return centroids


def _stream(chunks, sample: np.ndarray, centroids: np.ndarray, *, options: StreamRefineOptions, device: str,
            skip: int) -> np.ndarray:
    """Mini-batch k-means, Sculley (2010), "Web-scale k-means clustering", WWW: one chunk per update, each centroid
    the running mean of every point assigned to it so far (step 1 / count), starting with the sample's points.
    Holds one chunk and the centroids, whatever the size of the data."""
    C = torch.tensor(np.asarray(centroids, dtype=SOLVE_DTYPE), device=device)          # a copy: updated in place
    counts = torch.zeros(len(C), dtype=_TORCH_SOLVE, device=device)
    for start in range(0, len(sample), 10_000):
        counts += torch.bincount(_nearest(sample[start:start + 10_000], C, device), minlength=len(C))
    for p in range(options.passes):
        to_skip, seen = (skip if p == 0 else 0), 0
        for chunk in chunks:
            if to_skip:
                drop = min(to_skip, len(chunk))
                chunk, to_skip = chunk[drop:], to_skip - drop
            if options.max_points is not None:
                chunk = chunk[:options.max_points - seen]
            if len(chunk) == 0:
                if options.max_points is not None and seen >= options.max_points:
                    break
                continue
            X = torch.as_tensor(np.asarray(chunk, dtype=DATA_DTYPE), device=device)
            label = _nearest(X, C, device)
            n_c = torch.bincount(label, minlength=len(C)).to(_TORCH_SOLVE)
            sums = torch.zeros(C.shape, dtype=_TORCH_DATA, device=device).index_add_(0, label, X).to(_TORCH_SOLVE)
            counts += n_c
            C += (sums - n_c[:, None] * C) / counts.clamp(min=1.0)[:, None]
            seen += len(chunk)
    return C.cpu().numpy()


def _nearest(X, C: torch.Tensor, device: str) -> torch.Tensor:
    """Index of each point's nearest centroid (|x|^2 is the same for every centroid, so it is left out)."""
    X = torch.as_tensor(np.asarray(X, dtype=DATA_DTYPE), device=device) if isinstance(X, np.ndarray) else X
    C = C.to(_TORCH_DATA)
    return ((C * C).sum(dim=1)[None, :] - 2.0 * (X @ C.T)).argmin(dim=1)


def cell_masses(chunks, centroids: np.ndarray, config: Config, *, device: str) -> np.ndarray:
    """(n_cells,) share of the data in each cell, counted in one streamed pass; sums to 1."""
    raise NotImplementedError


_KMEANS = {
    "cuml": _cuml,
    "faiss": _faiss,
    "sklearn": _sklearn,
}

_REFINE = {
    "none": _none,
    "stream": _stream,
}
