"""Where does single-level LVM lose quality: in the voltage features or in the 2-D projection?

For each seed and number of landmarks L, fits LVM on MNIST and measures
trustworthiness, continuity and 5-NN accuracy on
  * the 2-D embedding (what the benchmark reports), and
  * the full L-dimensional log-voltage features, -log(max(v, tau)),
on the same evaluation points. A large gap means the information is in the
features and the linear 2-D projection discards it.

Runs on the *tuning* split: exploratory analysis never touches the evaluation
split. Writes results.txt next to this script.

    python experiments_old/sfv/mnist/diagnostics/feature_space/run.py [--n 20000] [--landmarks 20 40] [--overlap 2 4 8] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

from common.datasets import load  # noqa: E402
from common.metrics import quality  # noqa: E402

from lvm_old.config import load_config  # noqa: E402
from lvm_old.embedding import _log_features  # noqa: E402
from lvm_old.pipeline import fit_level  # noqa: E402
from lvm_old.stream import array_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--landmarks", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--overlap", type=float, nargs="+", default=[2.0], help="scaling.coverage.overlap values")
    parser.add_argument("--eval-size", type=int, default=5000)
    args = parser.parse_args()

    out = open(HERE / "results.txt", "w")

    def emit(line: str) -> None:
        print(line, flush=True)
        out.write(line + "\n")

    emit(f"{'L':>3} {'ovl':>4} {'seed':>4}  {'space':12s} {'trust':>6} {'contin':>6} {'5-NN':>6}  covered-by")
    for L, overlap in [(L, o) for L in args.landmarks for o in args.overlap]:
        for seed in args.seeds:
            X, y = load("tune", n=args.n, seed=seed)
            config = load_config(overrides={
                "compute": {"seed": seed},
                "landmarks": {"n_landmarks": L},
                "scaling": {"coverage": {"overlap": overlap}},
            })
            model = fit_level(array_source(X, config.data.chunk_size), config)
            idx = np.random.default_rng(seed + 1).choice(X.shape[0], min(args.eval_size, X.shape[0]), replace=False)
            Xe, ye = X[idx], y[idx]
            VX = model.voltages(Xe)
            reach = np.median((VX > 0).sum(axis=0))  # landmarks in reach of the median point
            for space, Z in (("2-D", model.embedding.transform(VX)), (f"{L}-D voltage", _log_features(VX, config.voltage.threshold))):
                q = quality(Xe, Z, ye, eval_size=len(idx), seed=seed)
                emit(f"{L:>3} {overlap:>4g} {seed:>4}  {space:12s} {q['trustworthiness']:6.3f} {q['continuity']:6.3f} "
                      f"{q['knn_accuracy']:6.3f}  {reach:g}")
    out.close()


if __name__ == "__main__":
    main()
