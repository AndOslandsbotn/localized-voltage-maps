"""One run for run.py: tuned LVM with a k-means sample of `--ppc` points per cell; prints a JSON line.

The dimension is fixed at the MNIST MLE value (d = 12.3 -> 14 landmarks), so
the landmark count doesn't move with the sample and only k-means changes.
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

N, D_FIXED = 20000, 12.3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppc", type=float, required=True, help="k-means sample points per cell")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    X, y = load("tune", n=N, seed=args.seed)
    params = best_params("lvm")
    n_cells = params["cells"]["n_cells"]
    sample_size = min(N, round(args.ppc * n_cells))
    cfg = load_config(overrides=_merge(params, {
        "compute": {"device": "cuda", "seed": args.seed},
        "cells": {"sample_size": sample_size, "kmeans": {"strategy": "cuml"}},
        "dimension": {"strategy": "fixed", "fixed": {"d": D_FIXED}},
    }))
    source = array_source(X, cfg.data.chunk_size)
    fit_level(array_source(X[:2000], cfg.data.chunk_size), cfg)          # warm-up
    t = time.perf_counter()
    model = fit_level(source, cfg)
    fit_s = time.perf_counter() - t
    Z = np.concatenate(list(model.transform_source(source)))
    # k-means objective on every point: mean squared distance to the nearest centroid.
    objective = float(np.concatenate([sq_distances(X[s:s + 5000], model.centroids).min(axis=1)
                                      for s in range(0, N, 5000)]).mean())
    print("RESULT " + json.dumps({
        "ppc": args.ppc, "seed": args.seed, "sample_size": sample_size, "objective": objective,
        "fit_s": fit_s, "cells_s": model.timings["cells"], "n_landmarks": int(model.V.shape[0]),
        **quality(X, Z, y, seed=args.seed),
    }), flush=True)


if __name__ == "__main__":
    main()
