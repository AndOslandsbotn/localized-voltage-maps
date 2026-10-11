import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import mnist8m_points
from experiments.common.guard import watch_memory
from experiments.common.metrics import quality
from experiments.sfv.mnist8m.from_mnist.run import CONDITIONS, METHODS, SEEDS

HERE = Path(__file__).resolve().parent
BASELINES = ("umap", "landmark_isomap")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    runs = [(method, condition) for method in METHODS for condition in CONDITIONS] + [(b, None) for b in BASELINES]
    for seed in SEEDS:
        for method, condition in runs:
            name = method if condition is None else f"{method}_{condition}"
            saved = np.load(HERE / "embeddings" / f"{name}_seed{seed}.npz")
            X, labels = mnist8m_points(saved["rows"])
            row = {"method": method, "condition": condition, "seed": seed, "fit_seconds": float(saved["fit_seconds"]),
                   "transform_seconds": float(saved["transform_seconds"]), "n_placed": int(saved["n_placed"]),
                   **quality(X, saved["Z"], labels, seed=seed)}
            if "model" in saved:
                row["fitted"] = json.loads(str(saved["model"]))["fitted"]
            rows.append(row)
    environment = json.loads(str(saved["environment"]))
    (HERE / "results.json").write_text(json.dumps({"fit_on": "mnist eval", "placed": "mnist8m", "seeds": SEEDS,
                                                   "environment": environment, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
