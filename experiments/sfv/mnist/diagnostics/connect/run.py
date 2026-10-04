"""graph.connect on MNIST: does joining the cell graph's isolated pieces change anything here?

Tuned LVM, tuning split 20k, GPU, seeds 0-2, connect on (default) vs off; each run isolated and memory-guarded.

    python experiments/sfv/mnist/diagnostics/connect/run.py
"""

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.methods import _merge, method_params  # noqa: E402
from common.results import read_rows, write_metadata, write_rows  # noqa: E402
from common.runner import run_isolated  # noqa: E402

N, SEEDS = 20000, (0, 1, 2)


def main() -> None:
    csv = HERE / "results.csv"
    rows = read_rows(csv)
    done = {(r["connect"], int(r["seed"])) for r in rows}
    write_metadata(HERE, {"split": "tune", "n": N, "seeds": list(SEEDS)})
    for connect in ("true", "false"):
        for seed in SEEDS:
            if (connect, seed) in done:
                continue
            params = _merge(method_params("lvm_gpu"), {"graph": {"connect": connect == "true"}})
            rows.append({**run_isolated("lvm_gpu", split="tune", n=N, seed=seed, params=params, threads=16),
                         "connect": connect})
            write_rows(csv, rows)
    print(f"{'connect':8s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6} {'rho_g':>9}")
    import json
    for connect in ("true", "false"):
        rs = [r for r in read_rows(csv) if r["connect"] == connect and r["status"] == "ok"]
        m = lambda k: statistics.mean(float(r[k]) for r in rs)
        rho = statistics.mean(json.loads(r["info"])["rho_g"] for r in rs)
        print(f"{connect:8s} {m('trustworthiness'):6.3f} {m('continuity'):6.3f} {m('knn_accuracy'):6.3f} "
              f"{m('distance_correlation'):6.3f} {rho:9.2e}")


if __name__ == "__main__":
    main()
