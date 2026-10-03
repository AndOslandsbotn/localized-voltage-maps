"""Point extension on MNIST: grounded (zero-mass node with ground) vs average (Def. 10, no ground).

Tuned LVM and LVM + local PCA, tuning split 20k, GPU, seeds 0-2, each run isolated and memory-guarded.

    python benchmarks/mnist/diagnostics/point_extension/run.py
"""

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

from common.methods import _merge, method_params  # noqa: E402
from common.results import read_rows, write_metadata, write_rows  # noqa: E402
from common.runner import run_isolated  # noqa: E402

N, SEEDS = 20000, (0, 1, 2)
STRATEGIES, METHODS = ("grounded", "average"), ("lvm_gpu", "lvm_pca_gpu")


def main() -> None:
    csv = HERE / "results.csv"
    rows = read_rows(csv)
    done = {(r["method"], r["strategy"], int(r["seed"])) for r in rows}
    write_metadata(HERE, {"split": "tune", "n": N, "seeds": list(SEEDS), "strategies": STRATEGIES})
    for method in METHODS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                if (method, strategy, seed) in done:
                    continue
                params = _merge(method_params(method), {"extension": {"strategy": strategy}})
                rows.append({**run_isolated(method, split="tune", n=N, seed=seed, params=params, threads=16),
                             "strategy": strategy})
                write_rows(csv, rows)
    print(f"{'method':12s} {'extension':9s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6} {'time':>5}")
    for method in METHODS:
        for strategy in STRATEGIES:
            rs = [r for r in read_rows(csv) if r["method"] == method and r["strategy"] == strategy and r["status"] == "ok"]
            m = lambda k: statistics.mean(float(r[k]) for r in rs)
            print(f"{method:12s} {strategy:9s} {m('trustworthiness'):6.3f} {m('continuity'):6.3f} "
                  f"{m('knn_accuracy'):6.3f} {m('distance_correlation'):6.3f} {m('total_s'):5.2f}")


if __name__ == "__main__":
    main()
