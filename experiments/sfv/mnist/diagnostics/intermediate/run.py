"""Is the information there before the 2-D picture? Every method's richer representation vs its 2-D result,
on MNIST (tuning split, 20k) and MNIST8M (its first 50k), seed 0; each run in its own memory-guarded process.

    python experiments/sfv/mnist/diagnostics/intermediate/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.runner import thread_env  # noqa: E402


def main() -> None:
    path = HERE / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done = {(r["dataset"], r["method"]) for r in rows}
    plan = [(d, m, None) for d in ("mnist", "mnist8m") for m in ("lvm", "lisomap", "le", "umap")] + \
           [("mnist", "lvm", L) for L in (30, 60)]
    for dataset, method, landmarks in plan:
        name = method + (f"_L{landmarks}" if landmarks else "")
        if (dataset, name) in done:
            continue
        if True:
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
                   str(HERE / "one.py"), "--dataset", dataset, "--method", method] + \
                  (["--landmarks", str(landmarks)] if landmarks else [])
            proc = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
            lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
            if not lines:
                print(f"{dataset} {name}: failed ({proc.returncode})\n{proc.stderr[-800:]}", flush=True)
                continue
            rows.append(json.loads(lines[-1][7:]))
            with open(path, "a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
    print(f"{'dataset':8s} {'method':8s} {'representation':42s} {'':5s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6}")
    for r in rows:
        for part in ("rich", "2d"):
            q = r[part]
            print(f"{r['dataset']:8s} {r['method']:8s} {(r['representation'] if part == 'rich' else '2-D'):42s} {part:5s} "
                  f"{q['trustworthiness']:6.3f} {q['continuity']:6.3f} {q['knn_accuracy']:6.3f} {q['distance_correlation']:6.3f}")


if __name__ == "__main__":
    main()
