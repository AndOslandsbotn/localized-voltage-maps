"""Landmark ablation: which selection strategy, then how many landmarks. Each run its own memory-guarded process.

1. Strategy: mi, maxmin, maxmin+mi, mi+maxmin, at the default count (d_hat + 1), 5 seeds, on mnist_tune, mnist,
   mnist8m and half_sphere. best.json records the rule's pick (mean trustworthiness on mnist_tune) and the strategy
   chosen: the metrics tie, so the user chose MI on the pictures (2026-10-03), the default.
2. Count: the chosen strategy with count multiplier 0.5 ... 4 (x (d_hat + 1)), 5 seeds, on mnist, mnist8m, half_sphere.

Results: results.jsonl (finished runs are skipped); then plot.py draws figure_strategy.png and figure_count.png.
--pictures: the seed-0 runs again with LVM's local PCA chart (results_pca.jsonl), for the figures' embeddings. The
chart spreads each cell's points but leaves the landmarks, cells and their layout unchanged, so the 5-seed
comparisons stay with plain LVM.

    python experiments_old/sfv/ablations/landmarks/run.py [--pictures]
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.runner import thread_env  # noqa: E402

STRATEGIES = ("mi", "maxmin", "maxmin+mi", "mi+maxmin")
CHOSEN = "mi"     # the metrics tie; chosen on the pictures (separate clusters rather than arms), the config default
MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SEEDS = range(5)
STRATEGY_DATASETS = ("mnist_tune", "mnist", "mnist8m", "half_sphere")
COUNT_DATASETS = ("mnist", "mnist8m", "half_sphere")


def run(rows: list, path: Path, dataset: str, strategy: str, multiplier: float, seed: int,
        local_pca: bool = False) -> None:
    if any(r["dataset"] == dataset and r["strategy"] == strategy and r["multiplier"] == multiplier
           and r["seed"] == seed for r in rows):
        return
    cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
           str(HERE / "one.py"), "--dataset", dataset, "--strategy", strategy, "--multiplier", str(multiplier),
           "--seed", str(seed)] + (["--local-pca"] if local_pca else [])
    out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    if not lines:
        print(f"{dataset} {strategy} m={multiplier} seed {seed}: failed\n{out.stderr[-800:]}", flush=True)
        return
    rows.append(json.loads(lines[-1][len("RESULT "):]))
    with open(path, "a") as f:
        f.write(json.dumps(rows[-1]) + "\n")
    r = rows[-1]
    print(f"{dataset:11s} {strategy:10s} m={multiplier:<4g} seed {seed}  L={r['n_landmarks']:3d}  trust "
          f"{r['trustworthiness']:.3f}  cont {r['continuity']:.3f}  5-NN {r['knn_accuracy']:.3f}  "
          f"global {r['distance_correlation']:.3f}", flush=True)


def pictures() -> None:
    """The seed-0 runs of both parts again, with the local PCA chart."""
    path = HERE / "results_pca.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    best = json.loads((HERE / "best.json").read_text())["best"]
    for dataset in STRATEGY_DATASETS[1:]:
        for strategy in STRATEGIES:
            run(rows, path, dataset, strategy, 1.0, 0, local_pca=True)
    for dataset in COUNT_DATASETS:
        for multiplier in MULTIPLIERS:
            run(rows, path, dataset, best, multiplier, 0, local_pca=True)
    subprocess.run([sys.executable, str(HERE / "plot.py")])


def main() -> None:
    if "--pictures" in sys.argv:
        pictures()
        return
    (HERE / "embeddings").mkdir(exist_ok=True)
    path = HERE / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    for dataset in STRATEGY_DATASETS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                run(rows, path, dataset, strategy, 1.0, seed)
    trust = {s: float(np.mean([r["trustworthiness"] for r in rows if r["dataset"] == "mnist_tune"
                               and r["strategy"] == s and r["multiplier"] == 1.0])) for s in STRATEGIES}
    rule = max(trust, key=trust.get)
    best = CHOSEN
    (HERE / "best.json").write_text(json.dumps({"best": best, "chosen_on": "the pictures (metrics tie), by the user",
                                                "rule_best": rule, "rule": "mnist_tune, mean trustworthiness",
                                                "trustworthiness": trust}, indent=1) + "\n")
    print(f"rule's pick on mnist_tune: {rule}  ({', '.join(f'{s} {t:.4f}' for s, t in trust.items())}); "
          f"count experiment with the chosen {best}", flush=True)
    for dataset in COUNT_DATASETS:
        for multiplier in MULTIPLIERS:
            for seed in SEEDS:
                run(rows, path, dataset, best, multiplier, seed)
    subprocess.run([sys.executable, str(HERE / "plot.py")])


if __name__ == "__main__":
    main()
