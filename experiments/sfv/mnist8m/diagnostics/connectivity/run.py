"""The cell graph per digit (one.py), MNIST and MNIST8M, 3 LVM seeds each; each fit in its own memory-guarded process.

    python experiments/sfv/mnist8m/diagnostics/connectivity/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.runner import thread_env  # noqa: E402

DATASETS, SEEDS = ("mnist", "mnist8m"), range(3)
COLUMNS = ("data_share", "cell_share", "degree", "connectivity", "nearest_over_r", "magnification", "pieces",
           "largest_piece", "detour", "global_stretch", "unreached")


def main() -> None:
    path = HERE / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    for dataset in DATASETS:
        for seed in SEEDS:
            if any(r["dataset"] == dataset and r["seed"] == seed for r in rows):
                continue
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
                   str(HERE / "one.py"), "--dataset", dataset, "--seed", str(seed)]
            out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
            lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
            if not lines:
                print(f"{dataset} seed {seed}: failed\n{out.stderr[-800:]}", flush=True)
                continue
            rows.append(json.loads(lines[-1][7:]))
            with open(path, "a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
    for dataset in DATASETS:
        rs = [r for r in rows if r["dataset"] == dataset]
        if not rs:
            continue
        print(f"\n== {dataset}: {len(rs)} seeds, {rs[0]['n_cells']} cells; per digit, mean over seeds of the median "
              "over its cells (magnification relative to all cells)")
        print(f"{'digit':>5} " + " ".join(f"{c[:13]:>13}" for c in COLUMNS))
        for d in range(10):
            print(f"{d:>5} " + " ".join(f"{np.mean([r['per_digit'][str(d)][c] for r in rs]):13.3f}" for c in COLUMNS))
        for key in ("connectivity", "degree", "mass", "nearest"):
            print(f"rank correlation over cells, magnification vs {key:12s} "
                  + " ".join(f"{r[f'corr_magnification_{key}']:+.2f}" for r in rs))


if __name__ == "__main__":
    main()
