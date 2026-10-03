"""LVM on the half sphere with a fixed number of landmarks; saves the embedding, landmarks and tears.

    python benchmarks/hemisphere/sweep_one.py --landmarks 9 [--method lvm_gpu]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common.methods import _merge, method_params  # noqa: E402
from hemisphere.one import half_sphere, scores  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402

N, SEED, K = 20000, 0, 10


def torn(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Points with one of their K nearest neighbours on the sphere placed > 10% of the embedding's width away."""
    _, idx = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)
    far = np.linalg.norm(Z[idx[:, 1:]] - Z[:, None, :], axis=2).max(axis=1)
    width = np.linalg.norm(np.quantile(Z, 0.99, axis=0) - np.quantile(Z, 0.01, axis=0))
    return far > 0.1 * width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=int, required=True)
    parser.add_argument("--method", default="lvm_gpu", choices=["lvm_gpu", "lvm_pca_gpu"])
    args = parser.parse_args()
    X = half_sphere(N, SEED)
    cfg = load_config(overrides=_merge(method_params(args.method), {
        "compute": {"device": "cuda", "seed": SEED},
        "landmarks": {"n_landmarks": args.landmarks, "count": {"strategy": "fixed"}}}))
    model = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    Z = model.transform(X)
    t = torn(X, Z)
    name = f"sweep_{args.method}_L{args.landmarks}"
    np.savez(HERE / f"{name}.npz", X=X, Z=Z, torn=t, landmarks=model.centroids[model.landmark_cells])
    out = {"method": args.method, "landmarks": args.landmarks, "torn_share": float(t.mean()),
           "rho_g": model.rho.rho_g, **scores(X, Z, SEED)}
    (HERE / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
