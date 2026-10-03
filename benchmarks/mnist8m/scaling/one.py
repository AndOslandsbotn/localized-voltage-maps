"""One MNIST8M run: a method at n points, data streamed from disk where the method allows it; prints a JSON row.

* lvm_gpu / lvm_pca_gpu (/ lvm_cpu): fit from the disk stream, then stream every point through transform.
* umap_gpu, lisomap_gpu: fit on the first --fit-n points (in memory), then transform the rest chunk by chunk,
  the usual way to apply a method to data that does not fit. Landmark Isomap's transform is the standard
  out-of-sample rule: d(x, l) = min over x's k nearest fitted points x_j of |x - x_j| + D(l, x_j), then the same
  triangulation.
* tsne_gpu, le_gpu: no transform (cuML), so fitted on all n points, as far as memory allows.

Time is from the stream to coordinates for all n points (after a warm-up on 2000 points); memory is the process's
peak resident memory and the GPU's peak (NVML). Quality: on 5000 random points of the n, as on MNIST.

    python benchmarks/mnist8m/scaling/one.py --method lvm_gpu --n 1000000
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import _mnist8m_arrays, mnist8m_rows, stream  # noqa: E402
from common.methods import METHODS, _merge, method_params  # noqa: E402
from common.metrics import quality  # noqa: E402
from common.runner import _free_gpu_caches, _GpuSampler  # noqa: E402

CHUNK = 10_000
_PROGRESS = {"last": 0.0, "start": time.perf_counter(), "run": ""}


def progress(stage: str, done: int | None = None, total: int | None = None, force: bool = False) -> None:
    """Write the current stage (and % done) to progress_run.json for run.py's overview, at most every 10 s."""
    now = time.perf_counter()
    if not force and now - _PROGRESS["last"] < 10.0:
        return
    _PROGRESS["last"] = now
    (HERE / "progress_run.json").write_text(json.dumps({"run": _PROGRESS["run"], "stage": stage, "done": done,
                                                         "total": total, "elapsed_s": now - _PROGRESS["start"]}))


def run_lvm(method: str, n: int, seed: int):
    from lvm.config import load_config
    from lvm.pipeline import fit_level
    from lvm.stream import array_source

    gpu = METHODS[method].device == "gpu"
    cfg = load_config(overrides=_merge(method_params(method), {
        "compute": {"device": "cuda" if gpu else "cpu", "seed": seed},
        "cells": {"kmeans": {"strategy": "cuml" if gpu else "faiss"}}, "data": {"chunk_size": CHUNK}}))
    fit_level(array_source(mnist8m_rows(0, 2000).astype(np.float64) / 255.0, CHUNK), cfg)   # warm-up
    source = stream("mnist8m", n=n, chunk_size=CHUNK)
    progress("fitting", force=True)
    t0 = time.perf_counter()
    model = fit_level(source, cfg)
    fit_s = time.perf_counter() - t0
    parts, done = [], 0
    for Zc in model.transform_source(source):
        parts.append(Zc)
        done += len(Zc)
        progress("transform (streamed)", done, n)
    Z = np.concatenate(parts)
    total_s = time.perf_counter() - t0
    return Z, fit_s, total_s, {"stages": model.timings, "n_cells": model.centroids.shape[0],
                               "n_landmarks": int(model.V.shape[0])}


def _float32(n: int) -> np.ndarray:
    """The first n points as float32 in [0, 1]: one array, filled chunk by chunk with plain file reads."""
    data = np.empty((n, 784), dtype=np.float32)
    for start in range(0, n, 200_000):
        stop = min(start + 200_000, n)
        data[start:stop] = mnist8m_rows(start, stop)
    data /= 255.0
    return data


def run_in_memory(method: str, n: int, seed: int):
    """A method that has no transform: its own wrapper (MNIST-tuned settings) on all n points held in memory."""
    m, params = METHODS[method], method_params(method)
    m.embed(_float32(2000), params, seed)                                            # warm-up
    progress(f"fitting on all {n:,} points (no progress inside the library)", force=True)
    t0 = time.perf_counter()
    Z, fit_s, info = m.embed(_float32(n), params, seed)
    return Z, fit_s, time.perf_counter() - t0, {**(info or {}), "fit_n": n}


def run_lisomap(n: int, fit_n: int, seed: int):
    """Landmark Isomap fitted on fit_n points (kNN graph + shortest paths, GPU), the rest placed out of sample."""
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    from common.methods import _lisomap_distances_gpu
    from lvm.embedding import place_landmarks

    params = method_params("lisomap_gpu")
    k, L = params["n_neighbors"], params["n_landmarks"]
    METHODS["lisomap_gpu"].embed(_float32(2000), params, seed)                       # warm-up
    fit_n = min(fit_n, n)
    progress(f"fitting on {fit_n:,} points (graph + shortest paths)", force=True)
    t0 = time.perf_counter()
    Xfit = _float32(fit_n)
    landmarks = np.random.default_rng(seed).choice(fit_n, L, replace=False)
    D = _lisomap_distances_gpu(Xfit, k, landmarks)                                   # (L, fit_n)
    finite = np.isfinite(D)
    D = np.where(finite, D, D[finite].max())
    placement = place_landmarks(D[:, landmarks], n_components=2)
    parts = [np.asarray(placement.triangulate(D.T))]
    fit_s = time.perf_counter() - t0
    if fit_n < n:
        nn = NearestNeighbors(n_neighbors=k).fit(Xfit)
        Dt = cp.asarray(D.T, dtype=cp.float32)                                       # (fit_n, L)
        for start in range(fit_n, n, 100_000):
            chunk = mnist8m_rows(start, min(start + 100_000, n)).astype(np.float32)
            chunk /= 255.0
            dist, idx = nn.kneighbors(chunk)
            dist, idx = cp.asarray(dist), cp.asarray(idx)
            d = cp.min(dist[:, :, None] + Dt[idx], axis=1)                            # (m, L) best route
            parts.append(cp.asnumpy(placement.triangulate(d)).astype(np.float64))
            progress("transform", min(start + 100_000, n), n)
    return np.concatenate(parts), fit_s, time.perf_counter() - t0, {"fit_n": fit_n}


def run_umap(n: int, fit_n: int, seed: int):
    from cuml.manifold import UMAP

    params = method_params("umap_gpu")
    kw = {"n_components": 2, "n_neighbors": params["n_neighbors"], "min_dist": params["min_dist"]}
    UMAP(**kw).fit_transform(_float32(2000))                                           # warm-up
    fit_n = min(fit_n, n)
    progress(f"fitting on {fit_n:,} points (no progress inside the library)", force=True)
    t0 = time.perf_counter()
    model = UMAP(**kw)
    data = _float32(fit_n)
    Zfit = np.asarray(model.fit_transform(data), dtype=np.float64)
    del data
    fit_s = time.perf_counter() - t0
    parts = [Zfit]
    for start in range(fit_n, n, 100_000):
        chunk = mnist8m_rows(start, min(start + 100_000, n)).astype(np.float32)
        chunk /= 255.0
        parts.append(np.asarray(model.transform(chunk), dtype=np.float64))
        progress("transform", min(start + 100_000, n), n)
    total_s = time.perf_counter() - t0
    return np.concatenate(parts), fit_s, total_s, {"fit_n": fit_n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["lvm_gpu", "lvm_cpu", "lvm_pca_gpu", "umap_gpu", "lisomap_gpu", "tsne_gpu", "le_gpu"])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--fit-n", type=int, default=None, help="umap: points to fit on (default: all n)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _PROGRESS["run"] = f"{args.method} n={args.n:,}" + (f" (fit on {args.fit_n:,})" if args.fit_n else "")
    progress("starting", force=True)
    gpu = METHODS[args.method].device == "gpu"
    if gpu:
        _free_gpu_caches()
        sampler = _GpuSampler()
        sampler.start()
    if args.method == "umap_gpu":
        Z, fit_s, total_s, info = run_umap(args.n, args.fit_n or args.n, args.seed)
    elif args.method == "lisomap_gpu":
        Z, fit_s, total_s, info = run_lisomap(args.n, args.fit_n or args.n, args.seed)
    elif args.method in ("tsne_gpu", "le_gpu"):
        Z, fit_s, total_s, info = run_in_memory(args.method, args.n, args.seed)
    else:
        Z, fit_s, total_s, info = run_lvm(args.method, args.n, args.seed)
    gpu_peak = sampler.stop() if gpu else None
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    progress("scoring (5000 random points)", force=True)
    rows = np.sort(np.random.default_rng(args.seed + 1).choice(args.n, 5000, replace=False))
    X, y = _mnist8m_arrays()
    scores = quality(X[rows].astype(np.float64) / 255.0, Z[rows], y[rows].astype(np.int64), seed=args.seed)
    tag = f"{args.method}_n{args.n}" + (f"_fit{info['fit_n']}" if info.get("fit_n", args.n) < args.n else "")
    np.save(HERE / f"Z_{tag}.npy", Z[::max(1, args.n // 200_000)].astype(np.float32))      # a subsample, for pictures
    np.save(HERE / f"y_{tag}.npy", y[:args.n][::max(1, args.n // 200_000)])
    row = {"method": args.method, "n": args.n, "seed": args.seed, "fit_s": fit_s, "total_s": total_s,
           "peak_rss_mb": rss_peak, "gpu_peak_mb": gpu_peak, **scores, "info": info}
    print("RESULT " + json.dumps(row, default=str), flush=True)


if __name__ == "__main__":
    main()
