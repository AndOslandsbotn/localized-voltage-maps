"""MNIST without the 0s: do the other digits fill the room the 0s took? One method on the evaluation split (50k,
seed 0, as the gallery), with or without the 0s; saves the embedding and prints a JSON row.

    python experiments/sfv/mnist/diagnostics/without_zeros/one.py --method lvm_pca_gpu --drop-zeros
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.methods import METHODS, method_params  # noqa: E402
from common.metrics import quality, stacking  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--drop-zeros", action="store_true")
    args = parser.parse_args()
    X, y = load("eval", n=50_000, seed=0)
    if args.drop_zeros:
        X, y = X[y != 0], y[y != 0]
    m, params = METHODS[args.method], method_params(args.method)
    m.embed(X[:2000].copy(), params, 0)                                               # warm-up
    Z, fit_s, _ = m.embed(X, params, 0)
    scores = {**quality(X, Z, y, seed=0), **stacking(Z)}
    name = f"{args.method}_{'without' if args.drop_zeros else 'with'}_zeros"
    (HERE / "embeddings").mkdir(exist_ok=True)
    np.savez(HERE / "embeddings" / f"{name}.npz", Z=np.asarray(Z, dtype=np.float32), y=y)
    print("RESULT " + json.dumps({"name": name, "method": args.method, "drop_zeros": args.drop_zeros, "n": len(X),
                                  "fit_s": fit_s, **scores}), flush=True)


if __name__ == "__main__":
    main()
