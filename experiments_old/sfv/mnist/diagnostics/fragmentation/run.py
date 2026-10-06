"""Fragmentation of each digit across representations; each in its own memory-guarded process.

    python experiments_old/sfv/mnist/diagnostics/fragmentation/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

from common.runner import thread_env  # noqa: E402

RUNS = [("raw", None), ("lvm", None), ("lvm_pca", 0.35), ("lvm_pca", 0.6), ("umap", None), ("tsne", None)]


def main() -> None:
    path = HERE / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done = {r["what"] for r in rows}
    for what, fill in RUNS:
        name = what + (f"_fill{fill:g}" if fill else "")
        if name in done:
            continue
        cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
               str(HERE / "one.py"), "--what", what] + (["--fill", str(fill)] if fill else [])
        out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not lines:
            print(f"{name}: failed\n{out.stderr[-600:]}", flush=True)
            continue
        rows.append(json.loads(lines[-1][7:]))
        with open(path, "a") as f:
            f.write(json.dumps(rows[-1]) + "\n")
    print("share of each digit in its largest connected group (10-NN links within the digit); [groups >= 2%]")
    print(f"{'representation':16s} " + " ".join(f"{d:>9d}" for d in range(10)))
    for r in rows:
        cells = [f"{r['per_digit'][str(d)]['largest_share']:.2f} [{r['per_digit'][str(d)]['groups_over_2pct']}]" for d in range(10)]
        print(f"{r['what']:16s} " + " ".join(f"{c:>9s}" for c in cells))


if __name__ == "__main__":
    main()
