"""Compare the `reach` scaling strategy with `coverage` (overlap 8) for the tuned LVM.

Tuned LVM (150 cells, Landmark MDS + chain, MI, d+1 landmarks), tuning split,
20k images, seeds 0-2; every run isolated and memory-guarded.
Writes results.csv; prints a summary with the achieved reach and rho_g.

    python experiments_old/sfv/mnist/diagnostics/reach/run.py
"""

import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

from common.runner import THREAD_VARS  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

import yaml  # noqa: E402

from common.results import read_rows, write_metadata, write_rows  # noqa: E402
from common.runner import run_isolated  # noqa: E402

N, SEEDS = 20000, (0, 1, 2)
BEST = yaml.safe_load((BENCH / "sfv" / "mnist" / "tuning" / "lvm" / "best.yaml").read_text())["params"]
VARIANTS = {
    "coverage overlap 8 (tuned)": {"scaling": {"strategy": "coverage", "coverage": {"overlap": 8.0}}},
    "reach k=all  q=0.5": {"scaling": {"strategy": "reach", "reach": {"k": None, "quantile": 0.5}}},
    "reach k=all  q=0.25": {"scaling": {"strategy": "reach", "reach": {"k": None, "quantile": 0.25}}},
    "reach k=all  q=0.1": {"scaling": {"strategy": "reach", "reach": {"k": None, "quantile": 0.1}}},
    "reach k=8    q=0.5": {"scaling": {"strategy": "reach", "reach": {"k": 8, "quantile": 0.5}}},
}


def params(variant: dict) -> dict:
    return {**BEST, **variant}


def main() -> None:
    csv_path = HERE / "results.csv"
    rows = read_rows(csv_path)
    done = {(r["params"], int(r["seed"])) for r in rows}
    write_metadata(HERE, {"split": "tune", "n": N, "seeds": list(SEEDS), "base": BEST, "variants": VARIANTS})
    for name, variant in VARIANTS.items():
        for seed in SEEDS:
            p = params(variant)
            if (json.dumps(p, sort_keys=True), seed) in done:
                continue
            row = {**run_isolated("lvm_gpu", split="tune", n=N, seed=seed, params=p, threads=16), "variant": name}
            rows.append(row)
            write_rows(csv_path, rows)
            if row["status"] != "ok":
                print(f"{name:28s} seed={seed}  {row['status']}", flush=True)
                continue
            info = json.loads(row["info"])
            print(f"{name:28s} seed={seed}  trust {row['trustworthiness']:.3f}  cont {row['continuity']:.3f}  "
                  f"knn {row['knn_accuracy']:.3f}  dcorr {row['distance_correlation']:.3f}  "
                  f"rho_g {info['rho_g']:.2e}  reach {info['reach']}  L {info['n_landmarks']}", flush=True)

    print(f"\n{'variant':28s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6} {'time':>5}")
    for name in VARIANTS:
        rs = [r for r in read_rows(csv_path) if r.get("variant") == name and r["status"] == "ok"]
        m = lambda k: statistics.mean(float(r[k]) for r in rs)
        print(f"{name:28s} {m('trustworthiness'):6.3f} {m('continuity'):6.3f} {m('knn_accuracy'):6.3f} "
              f"{m('distance_correlation'):6.3f} {m('total_s'):5.2f}")


if __name__ == "__main__":
    main()
