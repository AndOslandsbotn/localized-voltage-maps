"""A preview of the hierarchy: zoom into the region of some landmarks and fit LVM again there (MNIST evaluation split,
50k, seed 0, default settings with the local PCA chart).

--level0: fit LVM on all points; each cell goes to its highest-voltage landmark (argmax partition: one region per
  landmark); each landmark is named by the majority digit of its own cell (labels only name landmarks, they never
  define the regions). Saves the coordinates, each point's region, the landmarks' digits and positions.
--zoom 7,9: the points in the regions of the landmarks named 7 or 9, fitted again on their own (level 1: new cells,
  graph, landmarks, chart). Scores, on the same points, the level-0 coordinates (what zooming the camera shows) and
  the level-1 ones. Saves the level-1 landmarks' positions too.

A landmark's position is where LVM places its cell's centroid, so it sits among its cell's points.

    python experiments/sfv/mnist/zoom/one.py --level0
    python experiments/sfv/mnist/zoom/one.py --zoom 7,9
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
from common.methods import _merge, method_params  # noqa: E402
from common.metrics import quality, stacking  # noqa: E402

CHUNK = 10_000


def fit(X: np.ndarray):
    from lvm.config import load_config
    from lvm.pipeline import fit_level
    from lvm.stream import array_source

    cfg = load_config(overrides=_merge(method_params("lvm_pca_gpu"), {"compute": {"device": "cuda", "seed": 0}}))
    model = fit_level(array_source(X, CHUNK), cfg)
    return model, np.concatenate([model.transform(X[s:s + CHUNK]) for s in range(0, len(X), CHUNK)])


def landmark_positions(model) -> np.ndarray:
    """Where each landmark sits in the picture: its cell's centroid, embedded like any point."""
    return model.transform(model.centroids[model.landmark_cells])


def level0(X: np.ndarray, y: np.ndarray) -> None:
    from lvm.cells import assign_cells

    model, Z = fit(X)
    cell = assign_cells(X, model.centroids, device="cuda")
    V = model.V if model.V_dist is None else model.V_dist          # (L, n_cells), read down to the distance floor
    region_of_cell = np.asarray(V).argmax(axis=0)                   # argmax partition: cell -> landmark
    region = region_of_cell[cell]
    digits = [int(np.bincount(y[cell == c], minlength=10).argmax()) if np.any(cell == c) else -1
              for c in model.landmark_cells]
    (HERE / "embeddings").mkdir(exist_ok=True)
    np.savez(HERE / "embeddings" / "level0.npz", Z=Z, region=region, landmark_digits=np.array(digits),
             landmarks=landmark_positions(model))
    out = {"n": len(X), "n_landmarks": len(digits), "landmark_digits": digits,
           "region_sizes": np.bincount(region, minlength=len(digits)).tolist(), **quality(X, Z, y, seed=0)}
    (HERE / "level0.json").write_text(json.dumps(out, indent=1))
    print("level0", {k: v for k, v in out.items() if k != "region_sizes"}, flush=True)


def zoom(X: np.ndarray, y: np.ndarray, digits: list[int]) -> None:
    e = np.load(HERE / "embeddings" / "level0.npz")
    chosen = [j for j, d in enumerate(e["landmark_digits"]) if d in digits]
    name = "zoom_" + "_".join(map(str, digits))
    if not chosen:
        (HERE / f"{name}.json").write_text(json.dumps({"digits": digits, "error": "no landmark at these digits"}))
        print(name, "no landmark at these digits", flush=True)
        return
    rows = np.flatnonzero(np.isin(e["region"], chosen))
    Xr, yr, Z0 = X[rows], y[rows], e["Z"][rows]
    model1, Z1 = fit(Xr)
    composition = {int(d): float(np.mean(yr == d)) for d in np.unique(yr)}
    out = {"digits": digits, "landmarks": chosen, "landmarks_per_digit": {d: int(np.sum(e["landmark_digits"] == d))
                                                                          for d in digits},
           "n": len(rows), "composition": composition,
           "level0": {**quality(Xr, Z0, yr, seed=0), **stacking(Z0)},
           "level1": {**quality(Xr, Z1, yr, seed=0), **stacking(Z1)}}
    np.savez(HERE / "embeddings" / f"{name}.npz", rows=rows, Z1=Z1, landmarks1=landmark_positions(model1))
    (HERE / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(name, f"n={len(rows)}", {k: {m: round(v, 3) for m, v in out[k].items()} for k in ("level0", "level1")},
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level0", action="store_true")
    parser.add_argument("--zoom", default=None, help="digits whose landmarks' regions to zoom into, e.g. 7,9")
    args = parser.parse_args()
    X, y = load("eval", n=50_000, seed=0)
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)
    if args.level0:
        level0(X, y)
    if args.zoom:
        zoom(X, y, [int(d) for d in args.zoom.split(",")])


if __name__ == "__main__":
    main()
