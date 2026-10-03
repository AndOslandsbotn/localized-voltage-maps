"""One kernel variant of the tuned LVM on the tuning split (20k, GPU); prints a JSON line, saves Z.

    python benchmarks/mnist/diagnostics/kernels/one.py --variant B --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.methods import _merge, best_params  # noqa: E402
from common.metrics import quality  # noqa: E402
from lvm.cells import sq_distances  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402

N = 20000
VARIANTS = {   # graph kernel, point kernel
    "radial": {"graph": {"kernel": {"strategy": "radial"}}, "extension": {"kernel": "graph"}},
    "A": {"graph": {"kernel": {"strategy": "radial"}}, "extension": {"kernel": "tapered"}},
    "B": {"graph": {"kernel": {"strategy": "tapered"}}, "extension": {"kernel": "graph"}},
    "C": {"graph": {"kernel": {"strategy": "gaussian"}}, "extension": {"kernel": "graph"}},
    "D": {"graph": {"kernel": {"strategy": "radial"}}, "extension": {"kernel": "knn"}},
    "E": {"graph": {"kernel": {"strategy": "radial"}, "radius": {"strategy": "point_knn"}},
          "extension": {"kernel": "graph"}},
    **{f"D_k{k}_s{s:g}": {"graph": {"kernel": {"strategy": "radial"}},
                          "extension": {"kernel": "knn", "knn": {"k": k, "sharpness": s}}}
       for k in (3, 5, 10) for s in (1.0, 4.0, 16.0)},
    **{f"L_s{s:g}_f{f:g}": {"graph": {"kernel": {"strategy": "radial"}},
                            "extension": {"kernel": "knn", "knn": {"k": 3, "sharpness": s}},
                            "embedding": {"local_scale": {"strategy": "cell", "cell": {"fill": f}}}}
       for s in (4.0, 16.0) for f in (0.25, 0.5)},
    **{f"P_f{f:g}": {"graph": {"kernel": {"strategy": "radial"}},
                     "extension": {"kernel": "knn", "knn": {"k": 3, "sharpness": 16.0}},
                     "embedding": {"local_scale": {"strategy": "pca", "pca": {"fill": f}}}}
       for f in (0.35, 0.5)},
}


def stacking(Z: np.ndarray) -> dict:
    span = Z.max(0) - Z.min(0)
    squares = np.floor((Z - Z.min(0)) / (span * 1e-3)).astype(np.int64)
    _, counts = np.unique(squares, axis=0, return_counts=True)
    return {"occupied_squares": int(len(counts)), "share_in_150_fullest": float(np.sort(counts)[::-1][:150].sum() / len(Z))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    X, y = load("tune", n=N, seed=args.seed)
    cfg = load_config(overrides=_merge(_merge(best_params("lvm"), VARIANTS[args.variant]),
                                       {"compute": {"device": "cuda", "seed": args.seed},
                                        "cells": {"kmeans": {"strategy": "cuml"}}}))
    source = array_source(X, cfg.data.chunk_size)
    fit_level(array_source(X[:2000], cfg.data.chunk_size), cfg)          # warm-up
    t = time.perf_counter()
    model = fit_level(source, cfg)
    Z = np.concatenate(list(model.transform_source(source)))
    seconds = time.perf_counter() - t
    no_cell = float(np.mean(np.concatenate([sq_distances(X[s:s + 5000], model.centroids).min(axis=1)
                                            for s in range(0, N, 5000)]) > model.r**2))
    if args.seed == 0:
        np.savez(HERE / f"Z_{args.variant}.npz", Z=Z, y=y)
    print("RESULT " + json.dumps({"variant": args.variant, "seed": args.seed, "seconds": seconds,
                                  "rho_g": model.rho.rho_g, "no_cell_within_r": no_cell, **stacking(Z),
                                  "r": model.r, "cell_degree": float((model.K > 0).sum(axis=1).mean()),
                                  **quality(X, Z, y, seed=args.seed)}), flush=True)


if __name__ == "__main__":
    main()
