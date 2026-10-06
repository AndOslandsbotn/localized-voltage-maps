"""LVM's k-means sample in float32 (no extra copy) instead of float64: same quality on MNIST?

Tuned LVM, tuning split 20k, GPU, seeds 0-2, each run isolated; compared with the float64 runs of
../connect/results.csv (connect = true, the same settings otherwise).

    python experiments_old/sfv/mnist/diagnostics/sample_float32/run.py
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

from common.methods import method_params  # noqa: E402
from common.results import read_rows, write_metadata, write_rows  # noqa: E402
from common.runner import run_isolated  # noqa: E402

N, SEEDS = 20000, (0, 1, 2)


def summary(rows: list) -> str:
    m = lambda k: statistics.mean(float(r[k]) for r in rows)
    return (f"trust {m('trustworthiness'):.3f}  cont {m('continuity'):.3f}  knn {m('knn_accuracy'):.3f}  "
            f"dcorr {m('distance_correlation'):.3f}  peak {m('peak_rss_mb'):.0f} MB  time {m('total_s'):.2f} s")


def main() -> None:
    csv = HERE / "results.csv"
    rows = read_rows(csv)
    write_metadata(HERE, {"split": "tune", "n": N, "seeds": list(SEEDS)})
    for seed in SEEDS:
        if any(int(r["seed"]) == seed for r in rows):
            continue
        rows.append(run_isolated("lvm_gpu", split="tune", n=N, seed=seed, params=method_params("lvm_gpu"), threads=16))
        write_rows(csv, rows)
    before = [r for r in read_rows(HERE.parent / "connect" / "results.csv") if r["connect"] == "true" and r["status"] == "ok"]
    print("float64 sample:", summary(before))
    print("float32 sample:", summary([r for r in read_rows(csv) if r["status"] == "ok"]))


if __name__ == "__main__":
    main()
