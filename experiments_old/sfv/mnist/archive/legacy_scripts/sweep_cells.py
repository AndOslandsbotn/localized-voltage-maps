"""Speed/quality trade-off of single-level LVM against the number of cells (graph nodes).

Fits LVM on MNIST for each cells.n_cells value and seed, embeds every point,
and records stage timings and quality. Results go to
benchmarks/results/sweep_cells.csv.

    python benchmarks/sweep_cells.py [--cells 100 250 500 1000 2000] [--seeds 0 1] [--n 70000]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import load_mnist  # noqa: E402
from metrics import quality  # noqa: E402

from lvm_old.config import load_config  # noqa: E402
from lvm_old.pipeline import fit_level  # noqa: E402
from lvm_old.stream import array_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[100, 250, 500, 1000, 2000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--n", type=int, default=70000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = HERE / "results" / "sweep_cells.csv"
    rows = []
    # Warm-up so cuML/CUDA start-up isn't charged to the first configuration.
    X, _ = load_mnist(0, n=5000)
    fit_level(array_source(X, 5000), load_config(overrides={"cells": {"n_cells": 100}}), device=args.device)

    for seed in args.seeds:
        X, y = load_mnist(seed, n=args.n)
        for n_cells in args.cells:
            config = load_config(overrides={"compute": {"seed": seed, "device": args.device}, "cells": {"n_cells": n_cells}})
            source = array_source(X, config.data.chunk_size)
            t0 = time.perf_counter()
            model = fit_level(source, config)
            fit_s = time.perf_counter() - t0
            Z = np.concatenate(list(model.transform_source(source)))
            total_s = time.perf_counter() - t0
            q = quality(X, Z, y, eval_size=5000, seed=seed)
            row = {"n_cells": n_cells, "seed": seed, "fit_s": fit_s, "total_s": total_s, **q,
                   "stages": json.dumps({k: round(v, 3) for k, v in model.timings.items()})}
            rows.append(row)
            print(f"cells={n_cells:5d} seed={seed}  total {total_s:5.2f}s  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in model.timings.items())
                  + f"  trust {q['trustworthiness']:.3f}  knn {q['knn_accuracy']:.3f}", flush=True)
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"results: {out}")


if __name__ == "__main__":
    main()
