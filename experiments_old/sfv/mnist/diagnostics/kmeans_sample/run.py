"""How many points per cell does k-means need? (k-means sample size check)

Tuned LVM (150 cells), tuning split (20k), GPU, seeds 0-2, dimension fixed at
12.3 so only k-means changes. Points per cell 10 ... 133 (= all 20k points).
Measures the k-means objective on all points and the embedding metrics.
Question: has quality levelled off well below FAISS's default of 256 per centroid?

    python experiments_old/sfv/mnist/diagnostics/kmeans_sample/run.py
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

from common.results import write_metadata  # noqa: E402
from common.runner import thread_env  # noqa: E402

PPC, SEEDS = (10, 25, 39, 64, 133.34), (0, 1, 2)


def main() -> None:
    path = HERE / "results.jsonl"
    results = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done = {(r["ppc"], r["seed"]) for r in results}
    write_metadata(HERE, {"split": "tune", "n": 20000, "seeds": list(SEEDS), "points_per_cell": list(PPC),
                          "dimension": "fixed 12.3"})
    for ppc in PPC:
        for seed in SEEDS:
            if (ppc, seed) in done:
                continue
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "4500", "--",
                   sys.executable, str(HERE / "one.py"), "--ppc", str(ppc), "--seed", str(seed)]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16), cwd=BENCH.parent)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if proc.returncode != 0 or not lines:
                print(f"ppc={ppc} seed={seed}: failed ({proc.returncode})\n{proc.stderr[-2000:]}", flush=True)
                continue
            r = json.loads(lines[-1][len("RESULT "):])
            results.append(r)
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"ppc={ppc:6.1f} seed={seed}  objective {r['objective']:.2f}  trust {r['trustworthiness']:.3f}  "
                  f"cont {r['continuity']:.3f}  knn {r['knn_accuracy']:.3f}  dcorr {r['distance_correlation']:.3f}  "
                  f"cells {r['cells_s']:.2f}s", flush=True)

    print(f"\n{'ppc':>6} {'sample':>6} {'objective':>9} {'(rel)':>6} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6}")
    full = statistics.mean(r["objective"] for r in results if r["ppc"] == PPC[-1])
    for ppc in PPC:
        rs = [r for r in results if r["ppc"] == ppc]
        m = lambda k: statistics.mean(r[k] for r in rs)
        print(f"{ppc:6.0f} {rs[0]['sample_size']:6d} {m('objective'):9.2f} {m('objective') / full:6.3f} "
              f"{m('trustworthiness'):6.3f} {m('continuity'):6.3f} {m('knn_accuracy'):6.3f} "
              f"{m('distance_correlation'):6.3f}")


if __name__ == "__main__":
    main()
