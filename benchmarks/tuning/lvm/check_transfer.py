"""Does LVM's best cell count transfer across dataset sizes?

The comparisons run LVM at up to 50k points, but n_cells was tuned at 20k.
If the best n_cells stays put as n changes, n_cells is the right parameter.
If it grows with n, the right parameter is points per cell (n / n_cells), and
LVM should be re-tuned that way.

Checked entirely on the tuning split (5k, 10k and 20k subsets), so the
evaluation split stays unseen. Uses the tuned n_landmarks from best.yaml.
Writes transfer.csv and transfer.png next to this script.

    python benchmarks/tuning/lvm/check_transfer.py
"""

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.runner import THREAD_VARS  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

import yaml  # noqa: E402

from common.results import read_rows, write_rows  # noqa: E402
from common.runner import run_isolated  # noqa: E402

SIZES = (5000, 10000, 20000)
N_CELLS = (100, 150, 200, 300, 400, 600, 800)
SEEDS = (0, 1, 2)


def run() -> None:
    best = yaml.safe_load((HERE / "best.yaml").read_text())["params"]
    csv_path = HERE / "transfer.csv"
    rows = read_rows(csv_path)
    done = {(int(r["n"]), r["params"], int(r["seed"])) for r in rows}
    for n in SIZES:
        for n_cells in N_CELLS:
            params = {**best, "cells": {**best.get("cells", {}), "n_cells": n_cells}}
            key = json.dumps(params, sort_keys=True)
            for seed in SEEDS:
                if (n, key, seed) in done:
                    continue
                row = {**run_isolated("lvm_gpu", split="tune", n=n, seed=seed, params=params, threads=16),
                       "n_cells": n_cells}
                rows.append(row)
                write_rows(csv_path, rows)
                if row["status"] != "ok":
                    print(f"n={n:6d} n_cells={n_cells:4d} seed={seed}  {row['status']}", flush=True)
                    continue
                print(f"n={n:6d} n_cells={n_cells:4d} seed={seed}  trust {row['trustworthiness']:.3f}", flush=True)


def report() -> None:
    rows = [r for r in read_rows(HERE / "transfer.csv") if r.get("status", "ok") == "ok"]
    trust = defaultdict(list)
    for r in rows:
        trust[(int(r["n"]), int(r["n_cells"]))].append(float(r["trustworthiness"]))
    print(f"\n{'n':>6} {'best n_cells':>12} {'points/cell':>12} {'trust':>7}")
    for n in SIZES:
        cells = [c for c in N_CELLS if (n, c) in trust]
        best = max(cells, key=lambda c: statistics.mean(trust[(n, c)]))
        print(f"{n:6d} {best:12d} {n / best:12.0f} {statistics.mean(trust[(n, best)]):7.3f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in SIZES:
        cells = [c for c in N_CELLS if (n, c) in trust]
        means = [statistics.mean(trust[(n, c)]) for c in cells]
        axes[0].plot(cells, means, marker="o", label=f"n={n}")
        axes[1].plot([n / c for c in cells], means, marker="o", label=f"n={n}")
    axes[0].set_xlabel("n_cells")
    axes[1].set_xlabel("points per cell (n / n_cells)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel("trustworthiness")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("LVM: does the best cell count transfer across n? (tuning split, mean over seeds)")
    fig.tight_layout()
    fig.savefig(HERE / "transfer.png", dpi=150)
    print(f"plot: {HERE / 'transfer.png'}")


if __name__ == "__main__":
    run()
    report()
