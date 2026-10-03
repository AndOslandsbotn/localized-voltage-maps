"""Kernel variants for LVM: does a smooth kernel stop points piling up, and what does it do to quality?

  radial: hard kernel 1{d <= r} for the cell graph and the points (current)
  A:      hard kernel for the graph, tapered (1 - d^2/r^2)_+ for the points
  B:      tapered for both
  C:      Gaussian (sigma = r/3, cut at r) for both
  D:      hard kernel for the graph; each point weights its 10 nearest cells adaptively (extension.kernel knn)
  E:      hard kernel for both, with r large enough that a typical point has 10 cells within it (point_knn)
  D_k<k>_s<s>: D with k nearest cells and weight sharpness s (D = D_k10_s1)
  L_s<s>_f<f>: D with 3 nearest cells and sharpness s, plus a local scale per cell (fill f)
  P_f<f>: D with 3 nearest cells, sharpness 16, points placed within their cell by the cell's local PCA (fill f)

Tuned LVM (150 cells), tuning split (20k), GPU, seeds 0-2, each run isolated and
memory-guarded. Writes results.jsonl and Z_<variant>.npz (seed 0); plot.py draws them.

    python benchmarks/mnist/diagnostics/kernels/run.py
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

from common.runner import thread_env  # noqa: E402

VARIANTS = ("radial", "A", "B", "C", "D", "E", *(f"D_k{k}_s{s}" for k in (3, 5, 10) for s in (1, 4, 16)),
            *(f"L_s{s}_f{f}" for s in (4, 16) for f in (0.25, 0.5)), "P_f0.35", "P_f0.5")
SEEDS = (0, 1, 2)
KEYS = ("trustworthiness", "continuity", "knn_accuracy", "distance_correlation", "occupied_squares",
        "share_in_150_fullest", "no_cell_within_r", "rho_g", "seconds")


def main() -> None:
    path = HERE / "results.jsonl"
    results = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done = {(r["variant"], r["seed"]) for r in results}
    for variant in VARIANTS:
        for seed in SEEDS:
            if (variant, seed) in done:
                continue
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "4500", "--",
                   sys.executable, str(HERE / "one.py"), "--variant", variant, "--seed", str(seed)]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16), cwd=BENCH.parent)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if proc.returncode != 0 or not lines:
                print(f"{variant} seed={seed}: failed ({proc.returncode})\n{proc.stderr[-2000:]}", flush=True)
                continue
            results.append(json.loads(lines[-1][len("RESULT "):]))
            with open(path, "a") as f:
                f.write(json.dumps(results[-1]) + "\n")
            print(variant, seed, {k: round(results[-1][k], 4) for k in KEYS}, flush=True)

    print(f"\n{'variant':8s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6} {'squares':>8} {'top150':>7} "
          f"{'no cell':>7} {'rho_g':>8} {'time':>5}")
    for variant in VARIANTS:
        rs = [r for r in results if r["variant"] == variant]
        if not rs:
            continue
        m = {k: statistics.mean(r[k] for r in rs) for k in KEYS}
        print(f"{variant:8s} {m['trustworthiness']:6.3f} {m['continuity']:6.3f} {m['knn_accuracy']:6.3f} "
              f"{m['distance_correlation']:6.3f} {m['occupied_squares']:8.0f} {m['share_in_150_fullest']:7.1%} "
              f"{m['no_cell_within_r']:7.1%} {m['rho_g']:8.2e} {m['seconds']:5.2f}")


if __name__ == "__main__":
    main()
