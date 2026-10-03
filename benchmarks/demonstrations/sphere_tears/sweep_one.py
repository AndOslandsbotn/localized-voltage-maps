"""Sphere tears, one run: LVM on the half sphere with a given number of landmarks, at the pinned settings.

Saves the embedding, the landmarks and which points are torn (``torn``), and
the scores. Everything but the landmark count comes from ``settings.yaml``.

    python benchmarks/demonstrations/sphere_tears/sweep_one.py --landmarks 9 [--variant before|fix_A]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common.datasets import half_sphere  # noqa: E402
from common.metrics import quality  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402

N, DATA_SEED, K = 20000, 0, 10


def torn(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Points with one of their K nearest neighbours on the sphere placed > 10% of the embedding's width away."""
    _, idx = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)
    far = np.linalg.norm(Z[idx[:, 1:]] - Z[:, None, :], axis=2).max(axis=1)
    width = np.linalg.norm(np.quantile(Z, 0.99, axis=0) - np.quantile(Z, 0.01, axis=0))
    return far > 0.1 * width


SETTINGS = {"before": "settings.yaml", "fix_A": "settings_fix_A.yaml"}


def fit(landmarks: int, variant: str = "before"):
    X = half_sphere(N, DATA_SEED)
    cfg = load_config(HERE / SETTINGS[variant], overrides={"landmarks": {"n_landmarks": landmarks}})
    return X, fit_level(array_source(X, cfg.data.chunk_size), cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=int, required=True)
    parser.add_argument("--variant", choices=sorted(SETTINGS), default="before")
    args = parser.parse_args()
    X, model = fit(args.landmarks, args.variant)
    Z = model.transform(X)
    t = torn(X, Z)
    name = f"sweep_L{args.landmarks}" if args.variant == "before" else f"sweep_{args.variant}_L{args.landmarks}"
    # How much is still chained: point-landmark pairs and landmark pairs whose voltage is below the distance floor.
    floor = model.distance_floor
    v = model._voltages_and_cells(X, for_distances=True)[0]
    v = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    V_dist = model.V if model.V_dist is None else model.V_dist
    lm_v = V_dist[:, model.landmark_cells]
    np.savez(HERE / f"{name}.npz", X=X, Z=Z, torn=t, landmarks=model.centroids[model.landmark_cells])
    out = {"variant": args.variant, "landmarks": args.landmarks, "torn_share": float(t.mean()),
           "rho_g": model.rho.rho_g, "distance_floor": floor,
           "chained_point_pairs": float(np.mean(v < floor)),
           "chained_landmark_pairs": int(((lm_v < floor) & (lm_v.T < floor)).sum() // 2),
           **quality(X, Z, None, seed=DATA_SEED)}
    (HERE / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
