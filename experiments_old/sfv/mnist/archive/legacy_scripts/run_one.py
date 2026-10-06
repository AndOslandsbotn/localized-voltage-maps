"""Run one method on the first n (shuffled) MNIST images and print a JSON result line.

Run as its own process by ``run_all.py`` so that peak memory belongs to this
method alone:

    python benchmarks/run_one.py --method umap --n 20000

Before timing, each method is run once on a small subset so that one-off
costs (numba compilation for UMAP, CUDA start-up for LVM) are not counted.
Quality is measured afterwards on at most ``--eval-size`` points and is not
part of the timings.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from data import load_mnist  # noqa: E402
from metrics import quality  # noqa: E402

METHODS = ("lvm", "lvm_cpu", "le", "umap", "umap_gpu", "pca")


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def embed(method: str, X: np.ndarray, seed: int) -> tuple[np.ndarray, float, dict]:
    """(coordinates for every row of X, fit seconds, extra info)."""
    if method in ("lvm", "lvm_cpu"):
        from lvm_old.config import load_config
        from lvm_old.pipeline import fit_level
        from lvm_old.stream import array_source

        gpu = method == "lvm"
        # Everything on one kind of hardware: the CPU run also uses a CPU k-means.
        config = load_config(overrides={
            "compute": {"device": "cuda" if gpu else "cpu", "seed": seed},
            "cells": {"kmeans": {"strategy": "cuml" if gpu else "faiss"}},
        })
        source = array_source(X, config.data.chunk_size)
        t0 = time.perf_counter()
        model = fit_level(source, config)
        fit_s = time.perf_counter() - t0
        Z = np.concatenate(list(model.transform_source(source)))
        return Z, fit_s, {"stages": model.timings, "rho_g": model.rho.rho_g}

    t0 = time.perf_counter()
    if method == "le":
        from sklearn.manifold import SpectralEmbedding

        Z = SpectralEmbedding(n_components=2, affinity="nearest_neighbors", n_neighbors=10, random_state=seed, n_jobs=-1).fit_transform(X)
    elif method == "umap":
        import umap

        # No random_state: fixing it makes UMAP single-threaded.
        Z = umap.UMAP(n_components=2, n_neighbors=15).fit_transform(X)
    elif method == "umap_gpu":
        # RAPIDS cuML's GPU implementation of UMAP.
        from cuml.manifold import UMAP as CumlUMAP

        Z = CumlUMAP(n_components=2, n_neighbors=15).fit_transform(X.astype(np.float32))
        Z = np.asarray(Z, dtype=np.float64)
    elif method == "pca":
        from sklearn.decomposition import PCA

        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
    else:
        raise ValueError(f"unknown method {method!r}")
    return Z, time.perf_counter() - t0, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-size", type=int, default=5000)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    X, y = load_mnist(args.seed, n=args.n)
    if args.method == "umap_gpu":
        import rmm.statistics

        rmm.statistics.enable_statistics()  # before cuML allocates anything

    if not args.no_warmup:
        embed(args.method, X[:2000].copy(), args.seed)
    if args.method == "lvm":
        import torch

        torch.cuda.reset_peak_memory_stats()
    if args.method == "umap_gpu":
        rmm.statistics.push_statistics()  # fresh counters, so the warm-up isn't counted
    rss_before = peak_rss_mb()

    t0 = time.perf_counter()
    Z, fit_s, info = embed(args.method, X, args.seed)
    total_s = time.perf_counter() - t0
    rss_peak = peak_rss_mb()

    gpu_peak_mb = None
    if args.method == "lvm":
        import torch

        gpu_peak_mb = torch.cuda.max_memory_allocated() / 2**20
    if args.method == "umap_gpu":
        gpu_peak_mb = rmm.statistics.pop_statistics().peak_bytes / 2**20

    result = {
        "method": args.method,
        "n": args.n,
        "fit_s": fit_s,
        "total_s": total_s,
        "peak_rss_mb": rss_peak,
        # Peak above what loading the data and the warm-up had already reached;
        # 0 means the method never exceeded that earlier peak.
        "extra_rss_mb": max(0.0, rss_peak - rss_before),
        "gpu_peak_mb": gpu_peak_mb,
        **quality(X, Z, y, eval_size=args.eval_size, seed=args.seed),
        **({"info": info} if info else {}),
    }
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
