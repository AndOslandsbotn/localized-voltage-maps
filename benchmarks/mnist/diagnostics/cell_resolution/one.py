"""How much of a point's position inside its cell survives in LVM's representation? MNIST eval split (50k), seed 0.

resolution = (median distance from a point to its cell's centre) / (median distance from a cell's centre to the
nearest other cell's centre), in the same space. In pixel space this is ~1: points sit on a shell about as wide as
the cell spacing. In a representation that keeps where a point is inside its cell it stays ~1; near 0, every
point collapses onto its cell. Measured in pixels, in LVM's landmark distance vectors (-log v, chained) and in
LVM's 2-D coordinates, for the point step's sharpness (extension.knn.sharpness; default 16), with differences
computed directly (no |x|^2 + |y|^2 - 2 x.y cancellation).

    python benchmarks/mnist/diagnostics/cell_resolution/one.py --sharpness 16
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.methods import _merge, method_params  # noqa: E402
from common.metrics import quality  # noqa: E402

CHUNK = 10_000


def resolution(R: np.ndarray, cell: np.ndarray) -> dict:
    """Point-to-own-centre spread over centre-to-nearest-centre spacing, centres = mean of each cell's points."""
    from scipy.spatial import cKDTree

    cells = np.unique(cell)
    centres = np.stack([R[cell == c].mean(axis=0) for c in cells])
    index = np.searchsorted(cells, cell)
    spread = float(np.median(np.linalg.norm(R - centres[index], axis=1)))
    spacing = float(np.median(cKDTree(centres).query(centres, k=2)[0][:, 1]))
    return {"spread": spread, "spacing": spacing, "resolution": spread / spacing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sharpness", type=float, required=True)
    args = parser.parse_args()
    from lvm.config import load_config
    from lvm.embedding import point_landmark_distances
    from lvm.pipeline import _to_numpy, fit_level
    from lvm.stream import array_source

    X, y = load("eval", n=50_000, seed=0)
    X = np.asarray(X, dtype=np.float32)
    cfg = load_config(overrides=_merge(method_params("lvm_gpu"), {
        "compute": {"device": "cuda", "seed": 0}, "extension": {"knn": {"sharpness": args.sharpness}}}))
    model = fit_level(array_source(X, CHUNK), cfg)
    D, Z, cell = [], [], []
    for s in range(0, len(X), CHUNK):
        VX, nearest, _ = model._voltages_and_cells(X[s:s + CHUNK], for_distances=True)
        D.append(_to_numpy(point_landmark_distances(VX, model.embedding.landmark_D, tau=model.distance_floor,
                                                    missing=model.embedding.missing)).astype(np.float64))
        Z.append(model.transform(X[s:s + CHUNK]))
        cell.append(_to_numpy(nearest))
    D, Z, cell = np.concatenate(D), np.concatenate(Z), np.concatenate(cell)
    rows = np.sort(np.random.default_rng(1).choice(len(X), 5000, replace=False))
    scores = quality(X[rows].astype(np.float64), Z[rows], y[rows], seed=0)
    print("RESULT " + json.dumps({"sharpness": args.sharpness, "pixels": resolution(X.astype(np.float64), cell),
                                  "lvm_distances": resolution(D, cell), "lvm_2d": resolution(Z, cell),
                                  "trustworthiness": scores["trustworthiness"], "knn_accuracy": scores["knn_accuracy"]}),
          flush=True)


if __name__ == "__main__":
    main()
