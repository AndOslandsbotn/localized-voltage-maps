import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.metrics import quality

HERE = Path(__file__).resolve().parent
N, SEEDS = 100_000, (0, 1, 2)
SETTINGS = ("uniform", "counted_10000", "counted_20000", "counted_50000", "counted_100000")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows, settings = [], {}
    for seed in SEEDS:
        X, labels = load("eval", n=N, seed=seed, dataset="mnist8m")
        for name in SETTINGS:
            saved = np.load(HERE / "embeddings" / f"{name}_seed{seed}.npz")
            model = json.loads(str(saved["model"]))
            settings[name] = model["settings"]
            rows.append({"setting": name, "seed": seed, "seconds": float(saved["seconds"]),
                         **quality(X, saved["Z"], labels, seed=seed), "fitted": model["fitted"]})
    environment = json.loads(str(saved["environment"]))
    results = {"dataset": "mnist8m", "split": "eval", "n": N, "seeds": SEEDS, "environment": environment,
               "settings": settings, "rows": rows}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
