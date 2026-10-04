"""One LVM run for the landmark ablation: a selection strategy and a landmark count, on one dataset and seed.

Strategies (config landmarks): mi (mutual information), maxmin (farthest point), and combined, half each, in either
order: maxmin+mi (maxmin first, then MI given those) and mi+maxmin. The count is landmarks.count.dimension's
multiplier m: ceil(m (d_hat + 1)) landmarks (1 = the default d_hat + 1). Everything else: lvm_gpu's settings.

Datasets, each a fixed set of points (only LVM's seed varies): mnist_tune (MNIST's tuning split, 20k: the strategy is
chosen here), mnist (evaluation split, 50k), mnist8m (its first 50k points, the sample LVM fits on in the scaling
runs), half_sphere (20k). Seed 0 also saves the embedding and the landmarks' positions for the figures.
--local-pca adds LVM's second scale (the local PCA chart, fill 0.6, as lvm_pca_gpu): the scores are then those of
the charted coordinates, and the embedding is saved as <tag>_pca.npz.

    python experiments/sfv/ablations/landmarks/one.py --dataset mnist --strategy maxmin+mi --multiplier 1 --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.methods import _merge, method_params  # noqa: E402
from common.metrics import quality  # noqa: E402

CHUNK = 10_000
STRATEGIES = {
    "mi": {"strategy": "mutual_information"},
    "maxmin": {"strategy": "maxmin"},
    "maxmin+mi": {"strategy": "combined", "combined": {"first": "maxmin", "second": "mutual_information",
                                                       "fraction": 0.5}},
    "mi+maxmin": {"strategy": "combined", "combined": {"first": "mutual_information", "second": "maxmin",
                                                       "fraction": 0.5}},
}


def data(dataset: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(X float32, labels or None, a colour per point for the figures)."""
    from common.datasets import load

    if dataset == "mnist_tune":
        X, y = load("tune", n=20_000, seed=0)
    elif dataset == "mnist":
        X, y = load("eval", n=50_000, seed=0)
    elif dataset == "mnist8m":
        from common.datasets import _mnist8m_arrays, mnist8m_rows

        X, y = mnist8m_rows(0, 50_000).astype(np.float32) / 255.0, _mnist8m_arrays()[1][:50_000]
    else:
        X, y = load("eval", n=20_000, seed=0, dataset="half_sphere")
        X = np.asarray(X, dtype=np.float32)
        return X, None, np.arctan2(X[:, 1], X[:, 0])                     # coloured by angle around the pole
    X = np.asarray(X, dtype=np.float32)
    return X, np.asarray(y, dtype=np.int64), np.asarray(y, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist_tune", "mnist", "mnist8m", "half_sphere"], required=True)
    parser.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-pca", action="store_true")
    args = parser.parse_args()
    from lvm.config import load_config
    from lvm.pipeline import fit_level
    from lvm.stream import array_source

    X, y, colour = data(args.dataset)
    cfg = load_config(overrides=_merge(method_params("lvm_pca_gpu" if args.local_pca else "lvm_gpu"), {
        "compute": {"device": "cuda", "seed": args.seed},
        "landmarks": {**STRATEGIES[args.strategy], "count": {"strategy": "dimension",
                                                             "dimension": {"multiplier": args.multiplier}}}}))
    t0 = time.perf_counter()
    model = fit_level(array_source(X, CHUNK), cfg)
    Z = np.concatenate([model.transform(X[s:s + CHUNK]) for s in range(0, len(X), CHUNK)])
    seconds = time.perf_counter() - t0
    scores = quality(X.astype(np.float64), Z, y, seed=args.seed)
    info = {"n_landmarks": int(model.V.shape[0]), "d_hat": model.dimension.d, "rho_g": model.rho.rho_g,
            "rho_rounds": model.rho.n_rounds}
    if y is not None:                                   # which digit each landmark's cell mostly holds
        from lvm.cells import assign_cells

        cell = assign_cells(X, model.centroids, device="cuda")
        info["landmark_digits"] = [int(np.bincount(y[cell == c], minlength=10).argmax()) if np.any(cell == c) else -1
                                   for c in model.landmark_cells]
    tag = f"{args.dataset}_{args.strategy}_m{args.multiplier:g}" + ("_pca" if args.local_pca else "")
    if args.seed == 0:
        np.savez_compressed(HERE / "embeddings" / f"{tag}.npz", Z=Z.astype(np.float32), colour=colour.astype(np.float32),
                            landmarks=np.asarray(model.embedding.cell_coords[model.landmark_cells], dtype=np.float32))
    print("RESULT " + json.dumps({"dataset": args.dataset, "strategy": args.strategy, "multiplier": args.multiplier,
                                  "seed": args.seed, "local_pca": args.local_pca, "seconds": seconds, **scores, **info}), flush=True)


if __name__ == "__main__":
    main()
