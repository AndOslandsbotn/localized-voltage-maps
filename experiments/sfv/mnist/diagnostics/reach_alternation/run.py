"""Check the alternating `reach` search against the old one (re-selecting landmarks at every probe).

Tuned LVM (150 cells, reach k = all, q = 0.1), tuning split, 20k images, seeds 0-2:

1. same_graph.jsonl -- per seed and device, one fitted graph; old search vs the
   alternation from its start, x0.1 and x10: rho_g, landmark overlap,
   selections, seconds (same_graph.py, each run under the memory guard).
2. results.csv -- the tuned LVM end to end on GPU and CPU (isolated, guarded),
   compared with the old search's rows in tuning/lvm/results.csv.

    python experiments/sfv/mnist/diagnostics/reach_alternation/run.py
"""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.runner import THREAD_VARS  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

from common.methods import best_params  # noqa: E402
from common.results import read_rows, write_metadata, write_rows  # noqa: E402
from common.runner import run_isolated, thread_env  # noqa: E402

N, SEEDS, DEVICES = 20000, (0, 1, 2), ("cuda", "cpu")
STARTS = ("new_x1", "new_x0.1", "new_x10")


def same_graph() -> list[dict]:
    path = HERE / "same_graph.jsonl"
    results = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r["seed"], r["device"]) for r in results}
    for device in DEVICES:
        for seed in SEEDS:
            if (seed, device) in done:
                continue
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "4500", "--",
                   sys.executable, str(HERE / "same_graph.py"), "--seed", str(seed), "--device", device]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16), cwd=BENCH.parent)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if proc.returncode != 0 or not lines:
                print(f"same_graph seed={seed} {device}: failed ({proc.returncode})\n{proc.stderr[-2000:]}", flush=True)
                continue
            results.append(json.loads(lines[-1][len("RESULT "):]))
            with open(path, "a") as f:
                f.write(json.dumps(results[-1]) + "\n")
            r = results[-1]
            print(f"same_graph seed={seed} {device:4s}  old rho {r['old']['rho_g']:.2e} "
                  f"({r['old']['seconds']:.2f}s)  " + "  ".join(
                      f"{s}: rho {r[s]['rho_g']:.2e} rounds {r[s]['selections']} overlap {r[s]['overlap_with_old']}/"
                      f"{r['n_landmarks']} ({r[s]['seconds']:.2f}s)" for s in STARTS), flush=True)
    return results


def end_to_end() -> list[dict]:
    path = HERE / "results.csv"
    rows = read_rows(path)
    done = {(r["method"], int(r["seed"])) for r in rows}
    params = best_params("lvm")
    for method in ("lvm_gpu", "lvm_cpu"):
        for seed in SEEDS:
            if (method, seed) in done:
                continue
            row = run_isolated(method, split="tune", n=N, seed=seed, params=params, threads=16)
            rows.append(row)
            write_rows(path, rows)
            print(f"{method} seed={seed}  {row['status']}  guard peak {float(row['guard_peak_mb']):.0f} MB", flush=True)
    return rows


def main() -> None:
    write_metadata(HERE, {"split": "tune", "n": N, "seeds": list(SEEDS), "params": best_params("lvm")})
    graphs = same_graph()
    rows = end_to_end()

    print("\nSame graph (mean over seeds):")
    print(f"{'device':6s} {'search':9s} {'rho_g':>9s} {'vs old':>7s} {'select':>6s} {'overlap':>7s} {'seconds':>7s}")
    for device in DEVICES:
        rs = [r for r in graphs if r["device"] == device]
        if not rs:
            continue
        for s in ("old", *STARTS):
            ratio = statistics.mean(r[s]["rho_g"] / r["old"]["rho_g"] for r in rs)
            overlap = "" if s == "old" else f"{statistics.mean(r[s]['overlap_with_old'] for r in rs):.1f}"
            print(f"{device:6s} {s:9s} {statistics.mean(r[s]['rho_g'] for r in rs):9.2e} {ratio:7.2f} "
                  f"{statistics.mean(r[s]['selections'] for r in rs):6.1f} {overlap:>7s} "
                  f"{statistics.mean(r[s]['seconds'] for r in rs):7.2f}")

    old_rows = [r for r in read_rows(BENCH / "sfv" / "mnist" / "tuning" / "lvm" / "results.csv")
                if r["status"] == "ok" and json.loads(r["params"]) == best_params("lvm")]
    print("\nEnd to end (mean over seeds; old = tuning/lvm/results.csv, GPU only):")
    print(f"{'run':12s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6} {'total':>6} {'scaling':>7} {'rho_g':>9}")
    for name, rs in (("old lvm_gpu", old_rows),
                     ("new lvm_gpu", [r for r in rows if r["method"] == "lvm_gpu" and r["status"] == "ok"]),
                     ("new lvm_cpu", [r for r in rows if r["method"] == "lvm_cpu" and r["status"] == "ok"])):
        if not rs:
            continue
        m = lambda key: statistics.mean(float(r[key]) for r in rs)
        info = [json.loads(r["info"]) for r in rs]
        print(f"{name:12s} {m('trustworthiness'):6.3f} {m('continuity'):6.3f} {m('knn_accuracy'):6.3f} "
              f"{m('distance_correlation'):6.3f} {m('total_s'):6.2f} "
              f"{statistics.mean(i['stages']['scaling'] for i in info):7.3f} "
              f"{statistics.mean(i['rho_g'] for i in info):9.2e}")


if __name__ == "__main__":
    main()
