"""Overlap sweep for the best Landmark MDS variant (chain, MI, d+1 landmarks, 200 cells).

Overlap was still improving at 4 in run.py, so this checks larger values.
Tuning split, 20k images, seeds 0-2, each run isolated and memory-guarded.
Writes overlap.csv; prints a summary.

    python benchmarks/mnist/diagnostics/landmark_mds/overlap_sweep.py
"""

import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

from common.runner import THREAD_VARS  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

from common.results import read_rows, write_rows  # noqa: E402
from common.runner import run_isolated  # noqa: E402

N, SEEDS, OVERLAPS = 20000, (0, 1, 2), (4.0, 6.0, 8.0, 12.0)


def params(overlap: float) -> dict:
    return {"cells": {"n_cells": 200},
            "landmarks": {"strategy": "mutual_information", "count": {"strategy": "dimension",
                                                                      "dimension": {"multiplier": 1.0}}},
            "scaling": {"coverage": {"overlap": overlap}},
            "embedding": {"strategy": "landmark_mds", "landmark_mds": {"missing": "chain"}}}


def main() -> None:
    csv_path = HERE / "overlap.csv"
    rows = read_rows(csv_path)
    done = {(r["params"], int(r["seed"])) for r in rows}
    for overlap in OVERLAPS:
        for seed in SEEDS:
            p = params(overlap)
            if (json.dumps(p, sort_keys=True), seed) in done:
                continue
            row = {**run_isolated("lvm_gpu", split="tune", n=N, seed=seed, params=p, threads=16), "overlap": overlap}
            rows.append(row)
            write_rows(csv_path, rows)
            status = row["status"] if row["status"] != "ok" else (
                f"trust {row['trustworthiness']:.3f}  cont {row['continuity']:.3f}  knn {row['knn_accuracy']:.3f}  "
                f"dcorr {row['distance_correlation']:.3f}  peak {row['guard_peak_mb']:.0f}MB")
            print(f"overlap {overlap:4g} seed={seed}  {status}", flush=True)

    print(f"\n{'overlap':>7} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6}")
    for overlap in OVERLAPS:
        rs = [r for r in read_rows(csv_path) if r["status"] == "ok" and float(r["overlap"]) == overlap]
        m = lambda k: statistics.mean(float(r[k]) for r in rs)
        print(f"{overlap:7g} {m('trustworthiness'):6.3f} {m('continuity'):6.3f} {m('knn_accuracy'):6.3f} "
              f"{m('distance_correlation'):6.3f}")


if __name__ == "__main__":
    main()
