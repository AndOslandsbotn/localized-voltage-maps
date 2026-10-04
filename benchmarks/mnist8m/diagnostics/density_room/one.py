"""Are dense digits squeezed, and does their room follow their landmarks? One LVM fit (one seed) on a fixed 50k set.

Per digit:
* share: its fraction of the points; density: the median distance from its points to their 10 nearest points
  (any digit) in pixel space, relative to the median over all points (< 1: denser than typical);
* landmarks: how many landmarks sit in a cell where it is the majority digit;
* for each representation R (LVM's landmark distance vectors, the information the 2-D layout gets; LVM's 2-D; UMAP's
  2-D for comparison), relative to pixel space and to the other digits (1 = as in pixel space, < 1 = squeezed):
  - local: median 10-NN distance of its points in R / median over all points in R, divided by the same in pixels;
  - room: RMS distance of its points to its mean in R / the same over all points in R, divided by the same in pixels.

    python benchmarks/mnist8m/diagnostics/density_room/one.py --dataset mnist8m --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.methods import METHODS, _merge, method_params  # noqa: E402

N, K, CHUNK = 50_000, 10, 10_000


def data(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """The fixed 50k set: MNIST's evaluation split, or MNIST8M's first 50k points (its B sample)."""
    if dataset == "mnist":
        from common.datasets import load

        X, y = load("eval", n=N, seed=0)
    else:
        from common.datasets import _mnist8m_arrays, mnist8m_rows

        X, y = mnist8m_rows(0, N), _mnist8m_arrays()[1][:N]
    return np.asarray(X, dtype=np.float32) / (255.0 if np.asarray(X).max() > 1 else 1.0), np.asarray(y, dtype=np.int64)


def knn_distance(R: np.ndarray) -> np.ndarray:
    """Distance from each point to its K-th nearest other point.

    Low-dimensional R (LVM's distance vectors, 2-D): scikit-learn in float64. Points of one cell have nearly equal
    landmark distances, and cuML's float32 |x|^2 + |y|^2 - 2 x.y rounds those small distances to 0. Pixels: cuML.
    """
    if R.shape[1] <= 64:
        from sklearn.neighbors import NearestNeighbors

        dist, _ = NearestNeighbors(n_neighbors=K + 1).fit(R).kneighbors(R)
        return dist[:, K]
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    R = np.ascontiguousarray(R, dtype=np.float32)
    dist, _ = NearestNeighbors(n_neighbors=K + 1).fit(R).kneighbors(R)
    return cp.asnumpy(cp.asarray(dist))[:, K].astype(np.float64)


def spreads(R: np.ndarray, y: np.ndarray) -> tuple[dict, float, dict, float]:
    knn = knn_distance(R)
    rms = lambda P: float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean()))
    return ({d: float(np.median(knn[y == d])) for d in range(10)}, float(np.median(knn)),
            {d: rms(R[y == d]) for d in range(10)}, rms(R))


def lvm(X: np.ndarray, seed: int):
    """LVM fitted on X (lvm_gpu's settings): its landmark distance vectors, 2-D coordinates and landmarks' cells."""
    import torch

    from lvm.config import load_config
    from lvm.embedding import point_landmark_distances
    from lvm.pipeline import _to_numpy, fit_level
    from lvm.stream import array_source

    cfg = load_config(overrides=_merge(method_params("lvm_gpu"), {"compute": {"device": "cuda", "seed": seed}}))
    model = fit_level(array_source(X, CHUNK), cfg)
    D, Z, cell = [], [], []
    for s in range(0, len(X), CHUNK):
        VX, nearest, _ = model._voltages_and_cells(X[s:s + CHUNK], for_distances=True)
        D.append(_to_numpy(point_landmark_distances(VX, model.embedding.landmark_D, tau=model.distance_floor,
                                                    missing=model.embedding.missing)))
        Z.append(model.transform(X[s:s + CHUNK]))
        cell.append(_to_numpy(nearest))
    torch.cuda.empty_cache()
    return np.concatenate(D), np.concatenate(Z), np.concatenate(cell), model.landmark_cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "mnist8m"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    X, y = data(args.dataset)
    D, Z, cell, landmark_cells = lvm(X, args.seed)
    U, _, _ = METHODS["umap_gpu"].embed(X, method_params("umap_gpu"), args.seed)
    landmark_digits = [int(np.bincount(y[cell == c], minlength=10).argmax()) if np.any(cell == c) else -1
                       for c in landmark_cells]
    pix_knn, pix_knn_all, pix_rms, pix_rms_all = spreads(X, y)
    per_digit = {d: {"share": float(np.mean(y == d)), "density": pix_knn[d] / pix_knn_all,
                     "landmarks": landmark_digits.count(d)} for d in range(10)}
    for name, R in (("lvm_distances", D), ("lvm_2d", Z), ("umap_2d", np.asarray(U))):
        knn, knn_all, rms, rms_all = spreads(np.asarray(R, dtype=np.float64), y)
        for d in range(10):
            per_digit[d][f"local_{name}"] = (knn[d] / knn_all) / (pix_knn[d] / pix_knn_all)
            per_digit[d][f"room_{name}"] = (rms[d] / rms_all) / (pix_rms[d] / pix_rms_all)
    print("RESULT " + json.dumps({"dataset": args.dataset, "seed": args.seed, "n_landmarks": len(landmark_cells),
                                  "landmark_digits": landmark_digits, "per_digit": per_digit}), flush=True)


if __name__ == "__main__":
    main()
