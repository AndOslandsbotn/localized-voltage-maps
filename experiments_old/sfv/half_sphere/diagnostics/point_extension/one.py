"""Point extension on the half sphere: grounded (zero-mass node with ground) vs average (Def. 10, no ground).

Pinned fix-A settings of demonstrations/sphere_tears, one landmark count and strategy per run.

    python experiments_old/sfv/half_sphere/diagnostics/point_extension/one.py --landmarks 3 --strategy average
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "sfv" / "demonstrations" / "sphere_tears"))

import numpy as np  # noqa: E402

from common.metrics import quality  # noqa: E402
from lvm_old.cells import sq_distances  # noqa: E402
from lvm_old.config import load_config  # noqa: E402
from lvm_old.pipeline import fit_level  # noqa: E402
from lvm_old.stream import array_source  # noqa: E402
from sweep_one import DATA_SEED, HERE as TEARS, N, torn  # noqa: E402
from common.datasets import half_sphere  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=int, required=True)
    parser.add_argument("--strategy", choices=["grounded", "average"], required=True)
    args = parser.parse_args()
    X = half_sphere(N, DATA_SEED)
    cfg = load_config(TEARS / "settings_fix_A.yaml", overrides={
        "landmarks": {"n_landmarks": args.landmarks}, "extension": {"strategy": args.strategy}})
    m = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    Z = m.transform(X)
    Vp = m.voltages(X)
    Vc = m.V[:, sq_distances(X, m.centroids).argmin(1)]
    both = (Vp > 0) & (Vc > 0)
    out = {"landmarks": args.landmarks, "strategy": args.strategy, "torn": float(torn(X, Z).mean()),
           "point_over_cell_voltage": float(np.median(Vp[both] / Vc[both])),
           "point_pairs_below_tau": float(np.mean(Vp < cfg.voltage.threshold)), **quality(X, Z, None, seed=0)}
    out.pop("knn_accuracy")
    (HERE / f"L{args.landmarks}_{args.strategy}.json").write_text(json.dumps(out, indent=1))
    print("RESULT " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}), flush=True)


if __name__ == "__main__":
    main()
