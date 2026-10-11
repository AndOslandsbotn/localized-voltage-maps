import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.metrics import quality

HERE = Path(__file__).resolve().parent
N, SEEDS = 50_000, (0, 1, 2)
PLACEMENTS, MASSES, METHODS = ("kmeans", "farthest"), ("uniform", "counted"), ("lvm", "lvm_chart")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for seed in SEEDS:
        X, labels = load("eval", n=N, seed=seed)
        for method in METHODS:
            for placement in PLACEMENTS:
                for masses in MASSES:
                    saved = np.load(HERE / "embeddings" / f"{method}_{placement}_{masses}_seed{seed}.npz")
                    model = json.loads(str(saved["model"]))
                    rows.append({"method": method, "placement": placement, "masses": masses, "seed": seed,
                                 "seconds": float(saved["seconds"]), **quality(X, saved["Z"], labels, seed=seed),
                                 "fitted": model["fitted"]})
    environment = json.loads(str(saved["environment"]))
    results = {"dataset": "mnist", "split": "eval", "n": N, "seeds": SEEDS, "environment": environment,
               "rows": rows}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
