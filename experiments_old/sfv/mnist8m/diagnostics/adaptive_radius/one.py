"""One radius strategy (graph.radius: knn = one global r, or adaptive_per_cell) on a fixed 50k set, one LVM seed.

LVM with its local PCA chart (lvm_pca_gpu's settings); scores both the charted coordinates and the plain ones
(before the chart). Per digit: median edges per cell, and room = (RMS spread of its points around their mean / that
over all points) in the 2-D picture, divided by the same in pixel space (1 = its pixel-space share, < 1 squeezed,
> 1 blown up). Seed 0 saves the picture.

    python experiments_old/sfv/mnist8m/diagnostics/adaptive_radius/one.py --dataset mnist8m --radius adaptive_per_cell --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.methods import _merge, method_params  # noqa: E402
from common.metrics import quality, stacking  # noqa: E402

CHUNK = 10_000


def data(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "mnist":
        from common.datasets import load

        X, y = load("eval", n=50_000, seed=0)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
    from common.datasets import _mnist8m_arrays, mnist8m_rows

    return mnist8m_rows(0, 50_000).astype(np.float32) / 255.0, np.asarray(_mnist8m_arrays()[1][:50_000], dtype=np.int64)


def room(R: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    rms = lambda P: float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1).mean()))
    r_all, x_all = rms(R), rms(X)
    return {d: (rms(R[y == d]) / r_all) / (rms(X[y == d]) / x_all) for d in range(10)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "mnist8m"], required=True)
    parser.add_argument("--radius", choices=["knn", "adaptive_per_cell"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from lvm_old.cells import assign_cells
    from lvm_old.config import load_config
    from lvm_old.pipeline import _to_numpy, fit_level
    from lvm_old.stream import array_source

    X, y = data(args.dataset)
    cfg = load_config(overrides=_merge(method_params("lvm_pca_gpu"), {
        "compute": {"device": "cuda", "seed": args.seed}, "graph": {"radius": {"strategy": args.radius}}}))
    model = fit_level(array_source(X, CHUNK), cfg)
    Z = np.concatenate([model.transform(X[s:s + CHUNK]) for s in range(0, len(X), CHUNK)])
    Z0 = np.concatenate([_to_numpy(model._embed_on_device(X[s:s + CHUNK])[0]) for s in range(0, len(X), CHUNK)])
    X64 = X.astype(np.float64)
    charted, plain = {**quality(X64, Z, y, seed=args.seed), **stacking(Z)}, quality(X64, Z0, y, seed=args.seed)
    cell = assign_cells(X, model.centroids, device="cuda")
    n = len(model.centroids)
    digit = np.stack([np.bincount(cell[y == d], minlength=n) for d in range(10)]).argmax(axis=0)
    degree = (np.asarray(model.K) > 0).sum(axis=1)
    edges = {d: float(np.median(degree[digit == d])) if np.any(digit == d) else float("nan") for d in range(10)}
    if args.seed == 0:
        (HERE / "embeddings").mkdir(exist_ok=True)
        np.savez(HERE / "embeddings" / f"{args.dataset}_{args.radius}.npz", Z=Z.astype(np.float32), y=y)
    print("RESULT " + json.dumps({"dataset": args.dataset, "radius": args.radius, "seed": args.seed,
                                  "charted": charted, "plain": plain, "edges": edges, "room": room(Z, X64, y),
                                  "n_landmarks": int(model.V.shape[0]), "rho_g": model.rho.rho_g}), flush=True)


if __name__ == "__main__":
    main()
