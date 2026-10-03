"""Embed the evaluation split with one method and save it (for gallery/plot.py).

    python benchmarks/gallery/one.py --name lvm --method lvm_gpu [--params '{"...": ...}']
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from common.data import load  # noqa: E402
from common.methods import METHODS, method_params  # noqa: E402
from common.metrics import quality  # noqa: E402

N, SEED = 50000, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--params", default=None, help="JSON; default: the tuned settings")
    args = parser.parse_args()
    method = METHODS[args.method]
    params = json.loads(args.params) if args.params else method_params(args.method)
    X, y = load("eval", n=N, seed=SEED)
    Z, fit_s, _ = method.embed(X, params, SEED)
    scores = quality(X, Z, y, seed=SEED)
    np.savez(HERE / f"{args.name}.npz", Z=Z, y=y)
    (HERE / f"{args.name}.json").write_text(json.dumps(
        {"method": args.method, "label": method.label, "params": params, "fit_s": fit_s, **scores}, indent=1))
    print(args.name, {k: round(v, 3) for k, v in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
