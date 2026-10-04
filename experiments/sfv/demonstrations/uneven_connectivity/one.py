"""Uneven connectivity, one case: LVM at a pinned settings file on MNIST (evaluation split, 50k) or the half sphere
(20k), seed 0.

Measures how unevenly the cells are spaced (each cell's distance to its 10th nearest cell: 90th / 10th
percentile over cells), the edges per cell (10th / 50th / 90th percentile), the usual quality and, on MNIST, per
digit: median edges per cell and room = (RMS spread of its points in the picture / over all points) divided by the
same in pixel space (1 = its pixel-space share).

    python experiments/sfv/demonstrations/uneven_connectivity/one.py --dataset mnist --variant before|after
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.metrics import quality, stacking  # noqa: E402
from lvm.cells import assign_cells  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "half_sphere"], required=True)
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    args = parser.parse_args()
    if args.dataset == "mnist":
        X, y = load("eval", n=50_000, seed=0)
    else:
        X, y = load("eval", n=20_000, seed=0, dataset="half_sphere")
    X = np.asarray(X, dtype=np.float64)
    cfg = load_config(HERE / f"settings_{args.variant}.yaml")
    m = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    Z = np.concatenate([m.transform(X[s:s + 10_000]) for s in range(0, len(X), 10_000)])
    from sklearn.neighbors import NearestNeighbors

    tenth = NearestNeighbors(n_neighbors=11).fit(m.centroids).kneighbors(m.centroids)[0][:, 10]
    degree = (np.asarray(m.K) > 0).sum(axis=1)
    out = {"dataset": args.dataset, "variant": args.variant, "radius": cfg.graph.radius.strategy,
           "spacing_spread_p90_over_p10": float(np.percentile(tenth, 90) / np.percentile(tenth, 10)),
           "edges_p10_p50_p90": [float(np.percentile(degree, q)) for q in (10, 50, 90)],
           "n_landmarks": int(m.V.shape[0]), "rho_g": m.rho.rho_g, **quality(X, Z, y, seed=0), **stacking(Z)}
    if y is not None:
        cell = assign_cells(X, m.centroids, device="cuda")
        digit = np.stack([np.bincount(cell[y == d], minlength=len(m.centroids)) for d in range(10)]).argmax(axis=0)
        rms = lambda P: float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean()))
        out["edges_per_digit"] = {d: float(np.median(degree[digit == d])) for d in range(10) if np.any(digit == d)}
        out["room_per_digit"] = {d: (rms(Z[y == d]) / rms(Z)) / (rms(X[y == d]) / rms(X)) for d in range(10)}
    colour = y if y is not None else np.arctan2(X[:, 1], X[:, 0])
    (HERE / "embeddings").mkdir(exist_ok=True)
    np.savez(HERE / "embeddings" / f"{args.dataset}_{args.variant}.npz", Z=Z, colour=colour)
    (HERE / f"{args.dataset}_{args.variant}.json").write_text(json.dumps(out, indent=1))
    print(args.dataset, args.variant, {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()
                                       if not isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    main()
