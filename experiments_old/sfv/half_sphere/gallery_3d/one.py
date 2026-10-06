"""One 3-D embedding of the half sphere (eval stream, 20k, seed 0); saves Z and scores for plot.py.

    python experiments_old/sfv/half_sphere/gallery_3d/one.py --name lvm --method lvm_gpu --override '{...}'
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.methods import METHODS, _merge, method_params  # noqa: E402
from common.metrics import quality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--override", default="{}")
    args = parser.parse_args()
    X, _ = load("eval", n=20000, seed=0, dataset="half_sphere")
    m, params = METHODS[args.method], _merge(method_params(args.method), json.loads(args.override))
    m.embed(X[:2000].copy(), params, 0)                              # warm-up
    t0 = time.perf_counter()
    Z, _, info = m.embed(X, params, 0)
    seconds = time.perf_counter() - t0
    scores = quality(X, Z, None, seed=0)
    scores.pop("knn_accuracy")
    (HERE / "embeddings").mkdir(exist_ok=True)
    np.savez(HERE / "embeddings" / f"{args.name}.npz", X=X, Z=Z)
    (HERE / f"{args.name}.json").write_text(json.dumps({"method": args.method, "label": m.label, "params": params,
                                                         "seconds": seconds, "info": info, **scores}, indent=1, default=str))
    print(args.name, {k: round(v, 3) for k, v in scores.items()}, f"{seconds:.1f}s",
          f"landmarks {info.get('n_landmarks')}" if info else "", flush=True)


if __name__ == "__main__":
    main()
