"""Fix A on MNIST: distances read from voltages down to a floor (1e-10) instead of chaining below tau.

Tuned LVM (MNIST tuning, current defaults), tuning split 20k, GPU, seeds 0-2, every
run isolated and memory-guarded; plus, for seed 0, how many point-landmark pairs
and landmark pairs are chained with and without the floor.

    python experiments_old/sfv/mnist/diagnostics/distance_floor/run.py
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

from common.methods import _merge, method_params  # noqa: E402
from common.results import read_rows, write_metadata, write_rows  # noqa: E402
from common.runner import run_isolated, thread_env  # noqa: E402

N, SEEDS, FLOOR = 20000, (0, 1, 2), 1e-10
VARIANTS = {"tau (chain below 1e-3)": {}, "floor 1e-10": {"embedding": {"distance_floor": FLOOR}}}
METHODS = ("lvm_gpu", "lvm_pca_gpu")

CHAIN_COUNT = r'''
import json, sys
import numpy as np
sys.path.insert(0, "{bench}")
from common.datasets import load
from common.methods import _merge, method_params
from lvm_old.config import load_config
from lvm_old.pipeline import fit_level
from lvm_old.stream import array_source
X, _ = load("tune", n={n}, seed=0)
cfg = load_config(overrides=_merge(method_params("lvm_gpu"), _merge(json.loads('{override}'),
                  {{"compute": {{"device": "cuda", "seed": 0}}}})))
m = fit_level(array_source(X, cfg.data.chunk_size), cfg)
v = m._voltages_and_cells(X, for_distances=True)[0].cpu().numpy()
V = m.V if m.V_dist is None else m.V_dist
lm = V[:, m.landmark_cells]
L = lm.shape[0]
print("RESULT " + json.dumps({{"chained_point_pairs": float(np.mean(v < m.distance_floor)),
                              "chained_landmark_pairs": int(((lm < m.distance_floor) & (lm.T < m.distance_floor)).sum() // 2),
                              "landmark_pairs": L * (L - 1) // 2,
                              "min_point_voltage": float(v.min())}}))
'''


def main() -> None:
    csv = HERE / "results.csv"
    rows = read_rows(csv)
    done = {(r["method"], r["variant"], int(r["seed"])) for r in rows}
    write_metadata(HERE, {"split": "tune", "n": N, "seeds": list(SEEDS), "variants": VARIANTS})
    for method in METHODS:
        for name, override in VARIANTS.items():
            for seed in SEEDS:
                if (method, name, seed) in done:
                    continue
                row = run_isolated(method, split="tune", n=N, seed=seed, params=_merge(method_params(method), override),
                                   threads=16)
                rows.append({**row, "variant": name})
                write_rows(csv, rows)
                print(method, name, seed, row["status"], flush=True)
    counts = {}
    for name, override in VARIANTS.items():
        code = CHAIN_COUNT.format(bench=BENCH, n=N, override=json.dumps(override))
        cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "4500", "--", sys.executable, "-c", code]
        out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16)).stdout
        counts[name] = json.loads([l for l in out.splitlines() if l.startswith("RESULT ")][-1][7:])
    (HERE / "chained.json").write_text(json.dumps(counts, indent=1))

    print(f"\n{'method':12s} {'variant':24s} {'trust':>6} {'cont':>6} {'knn':>6} {'dcorr':>6} {'time':>5}")
    for method in METHODS:
        for name in VARIANTS:
            rs = [r for r in read_rows(csv) if r["method"] == method and r["variant"] == name and r["status"] == "ok"]
            m = lambda k: statistics.mean(float(r[k]) for r in rs)
            print(f"{method:12s} {name:24s} {m('trustworthiness'):6.3f} {m('continuity'):6.3f} {m('knn_accuracy'):6.3f} "
                  f"{m('distance_correlation'):6.3f} {m('total_s'):5.2f}")
    print("\nchained (seed 0):", json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
