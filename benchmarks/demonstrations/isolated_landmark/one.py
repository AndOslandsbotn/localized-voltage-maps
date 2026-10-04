"""Isolated landmark, one variant: LVM at a pinned settings file on MNIST8M's first 50k points (seed 0).

    python benchmarks/demonstrations/isolated_landmark/one.py --variant before|after
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from common.datasets import load  # noqa: E402
from common.metrics import quality, stacking  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    args = parser.parse_args()
    X, y = load("eval", n=50000, seed=0, dataset="mnist8m")
    cfg = load_config(HERE / f"settings_{args.variant}.yaml")
    m = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    Z = m.transform(X)
    n_pieces, piece = connected_components(m.K > 0, directed=False)
    sizes = np.bincount(piece)
    out = {"variant": args.variant, "graph_pieces": int(n_pieces),
           "landmarks_in_size1_pieces": int(np.sum(sizes[piece[m.landmark_cells]] == 1)),
           "n_landmarks": int(m.V.shape[0]), "rho_g": m.rho.rho_g, "rho_g_lower_bound": cfg.scaling.reach.rho_g_bounds[0],
           "reach": m.rho.reach, **quality(X, Z, y, seed=0), **stacking(Z)}
    (HERE / "embeddings").mkdir(exist_ok=True)
    np.savez(HERE / "embeddings" / f"{args.variant}.npz", Z=Z, y=y)
    (HERE / f"{args.variant}.json").write_text(json.dumps(out, indent=1))
    print(args.variant, {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
