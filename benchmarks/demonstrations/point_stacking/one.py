"""Point stacking, one variant: LVM at a pinned settings file on the MNIST tuning split (20k, seed 0).

    python benchmarks/demonstrations/point_stacking/one.py --variant before|after|after_local_pca
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.metrics import quality, stacking  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402

VARIANTS = ("before", "after", "after_local_pca")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    X, y = load("tune", n=20000, seed=0)
    cfg = load_config(HERE / f"settings_{args.variant}.yaml")
    fit_level(array_source(X[:2000], cfg.data.chunk_size), cfg)          # warm-up
    t = time.perf_counter()
    model = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    Z = np.concatenate(list(model.transform_source(array_source(X, cfg.data.chunk_size))))
    seconds = time.perf_counter() - t
    out = {"variant": args.variant, "seconds": seconds, **quality(X, Z, y, seed=0), **stacking(Z)}
    (HERE / "embeddings").mkdir(exist_ok=True)
    np.savez(HERE / "embeddings" / f"{args.variant}.npz", Z=Z, y=y)
    (HERE / f"{args.variant}.json").write_text(json.dumps(out, indent=1))
    print(args.variant, {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
