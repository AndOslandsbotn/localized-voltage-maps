"""First look at LVM with Landmark MDS, on the tuning split.

Variants of single-level LVM with ``embedding: landmark_mds``:
  landmark selection {mutual_information, maxmin} x count {d+1, 2(d+1)}
  x missing distances {clip, chain} x coverage overlap {2, 4},
against the current tuned LVM (log_mds) as the reference. 200 cells (tuned).
Each configuration runs with seeds 0-2 on 20k images of the tuning split.
Writes results.csv; prints a summary table (mean over seeds).

    python experiments/sfv/mnist/diagnostics/landmark_mds/run.py
"""

import itertools
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.runner import THREAD_VARS  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

from common.results import read_rows, write_metadata, write_rows  # noqa: E402
from common.runner import run_isolated  # noqa: E402

N, SEEDS = 20000, (0, 1, 2)
BASE = {"cells": {"n_cells": 200}}
REFERENCE = {**BASE, "landmarks": {"n_landmarks": 10}}   # tuned log_mds LVM
VARIANTS = [
    {**BASE,
     "landmarks": {"strategy": strategy, "count": {"strategy": "dimension", "dimension": {"multiplier": mult}}},
     "scaling": {"coverage": {"overlap": overlap}},
     "embedding": {"strategy": "landmark_mds", "landmark_mds": {"missing": missing}}}
    for strategy, mult, missing, overlap in itertools.product(
        ["mutual_information", "maxmin"], [1.0, 2.0], ["clip", "chain"], [2.0, 4.0])
]


def label(params: dict) -> str:
    if params.get("embedding", {}).get("strategy") != "landmark_mds":
        return "reference: log_mds, MI, 10 landmarks"
    lm = params["landmarks"]
    return (f"lmds {params['embedding']['landmark_mds']['missing']:5s} "
            f"{'MI' if lm['strategy'] == 'mutual_information' else 'maxmin':6s} "
            f"{lm['count']['dimension']['multiplier']:g}(d+1) overlap {params['scaling']['coverage']['overlap']:g}")


def main() -> None:
    csv_path = HERE / "results.csv"
    rows = read_rows(csv_path)
    done = {(r["params"], int(r["seed"])) for r in rows}
    write_metadata(HERE, {"split": "tune", "n": N, "seeds": list(SEEDS), "reference": REFERENCE, "variants": VARIANTS})
    for params in [REFERENCE, *VARIANTS]:
        for seed in SEEDS:
            key = json.dumps(params, sort_keys=True)
            if (key, seed) in done:
                continue
            row = run_isolated("lvm_gpu", split="tune", n=N, seed=seed, params=params, threads=16)
            rows.append(row)
            write_rows(csv_path, rows)
            if row["status"] != "ok":
                print(f"{label(params):45s} seed={seed}  {row['status']}", flush=True)
                continue
            print(f"{label(params):45s} seed={seed}  trust {row['trustworthiness']:.3f}  cont {row['continuity']:.3f}  "
                  f"knn {row['knn_accuracy']:.3f}  dcorr {row['distance_correlation']:.3f}", flush=True)

    print(f"\n{'variant':45s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6} {'time':>6}")
    by = {}
    for r in read_rows(csv_path):
        if r.get("status", "ok") != "ok":
            continue
        by.setdefault(r["params"], []).append(r)
    for key, rs in sorted(by.items(), key=lambda kv: -statistics.mean(float(r["continuity"]) for r in kv[1])):
        m = lambda k: statistics.mean(float(r[k]) for r in rs)
        print(f"{label(json.loads(key)):45s} {m('trustworthiness'):6.3f} {m('continuity'):6.3f} "
              f"{m('knn_accuracy'):6.3f} {m('distance_correlation'):6.3f} {m('total_s'):6.2f}")


if __name__ == "__main__":
    main()
