"""Density vs room per digit, MNIST and MNIST8M, 5 LVM seeds each; each fit in its own memory-guarded process.

Prints, per dataset, the per-digit table averaged over seeds, and across all (digit, seed) pairs the rank
correlations that answer: are dense digits squeezed (density vs local / room), and does a digit's room follow its
landmark count (landmarks vs room in 2-D)?

    python benchmarks/mnist8m/diagnostics/density_room/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common.runner import thread_env  # noqa: E402

DATASETS, SEEDS = ("mnist", "mnist8m"), range(5)
REPS = ("lvm_distances", "lvm_2d", "umap_2d")


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
        print(f"\n== {dataset}: {len(rs)} seeds; mean over seeds (local / room: 1 = as in pixels, < 1 = squeezed)")
        print(f"{'digit':>5} {'share':>6} {'density':>8} {'landmarks':>9}  " +
              "  ".join(f"{'local ' + n:>19} {'room ' + n:>18}" for n in REPS))
        mean = lambda d, key: float(np.mean([r["per_digit"][str(d)][key] for r in rs]))
        for d in range(10):
            print(f"{d:>5} {mean(d, 'share'):6.3f} {mean(d, 'density'):8.2f} {mean(d, 'landmarks'):9.1f}  " +
                  "  ".join(f"{mean(d, 'local_' + n):19.2f} {mean(d, 'room_' + n):18.2f}" for n in REPS))
        pairs = [r["per_digit"][str(d)] for r in rs for d in range(10)]
        col = lambda key: [p[key] for p in pairs]
        print("rank correlation over (digit, seed) pairs:")
        for n in REPS:
            print(f"  density vs local {n:14s} {spearmanr(col('density'), col('local_' + n))[0]:+.2f}    "
                  f"density vs room {spearmanr(col('density'), col('room_' + n))[0]:+.2f}")
        for n in ("lvm_distances", "lvm_2d"):
            print(f"  landmarks vs room {n:13s} {spearmanr(col('landmarks'), col('room_' + n))[0]:+.2f}")
        print("  landmark digits per seed: " + "; ".join(",".join(map(str, sorted(r["landmark_digits"]))) for r in rs))


if __name__ == "__main__":
    main()
