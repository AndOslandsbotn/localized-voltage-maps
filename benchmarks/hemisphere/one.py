"""Embed the half sphere with one method; save the embedding and its scores (for plot.py).

Points uniform on the upper half of the unit sphere in R^3 (z >= 0). It can be
flattened into a disc, so a good 2-D embedding keeps every neighbourhood and
changes colour smoothly. Settings: each method's MNIST-tuned ones, unchanged.

    python benchmarks/hemisphere/one.py --method lvm_gpu [--n 20000] [--seed 0]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.manifold import trustworthiness  # noqa: E402

from common.methods import METHODS, _merge, method_params  # noqa: E402


def half_sphere(n: int, seed: int) -> np.ndarray:
    """n points uniform on {x in R^3 : |x| = 1, x_3 >= 0}, in random order."""
    X = np.random.default_rng(seed).normal(size=(n, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X[:, 2] = np.abs(X[:, 2])
    return X


def scores(X: np.ndarray, Z: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(len(X), 5000, replace=False)
    g = idx[:1500]
    great_circle = np.arccos(np.clip(1 - pdist(X[g]) ** 2 / 2, -1, 1))     # distance along the sphere
    return {
        "trustworthiness": float(trustworthiness(X[idx], Z[idx], n_neighbors=10)),
        "continuity": float(trustworthiness(Z[idx], X[idx], n_neighbors=10)),
        "geodesic_correlation": float(spearmanr(great_circle, pdist(Z[g])).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", default=None, help="JSON merged over the method's settings (diagnostics)")
    parser.add_argument("--name", default=None, help="output name (default: the method)")
    args = parser.parse_args()
    name = args.name or args.method
    X = half_sphere(args.n, args.seed)
    method, params = METHODS[args.method], method_params(args.method)
    if args.override:
        params = _merge(params, json.loads(args.override))
    method.embed(X[:2000].copy(), params, args.seed)                         # warm-up
    Z, fit_s, info = method.embed(X, params, args.seed)
    out = {"method": args.method, "label": method.label, "n": args.n, "seed": args.seed, "params": params,
           "fit_s": fit_s, "info": info, **scores(X, Z, args.seed)}
    np.savez(HERE / f"{name}.npz", X=X, Z=Z)
    (HERE / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))
    print(name, {k: round(v, 3) for k, v in out.items() if isinstance(v, float)},
          {k: info.get(k) for k in ("dimension", "n_landmarks", "rho_g")} if info else "", flush=True)


if __name__ == "__main__":
    main()
