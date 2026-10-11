import json
from pathlib import Path

import numpy as np

from experiments.common import methods
from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.metrics import quality
from experiments.common.record import defaults

HERE = Path(__file__).resolve().parent
SIZES = (5_000, 10_000, 20_000, 50_000, 70_000, 100_000)
SEEDS = (0, 1, 2)
METHODS = {"lvm": methods.lvm, "lvm_chart": methods.lvm_chart, "umap": methods.umap_gpu, "tsne": methods.tsne_gpu,
           "laplacian_eigenmaps": methods.laplacian_eigenmaps_gpu, "landmark_isomap": methods.landmark_isomap_gpu}


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for n in SIZES:
        for seed in SEEDS:
            X, labels = load("eval", n=n, seed=seed, dataset="mnist8m")
            for name in METHODS:
                saved = np.load(HERE / "embeddings" / f"{name}_n{n}_seed{seed}.npz")
                row = {"method": name, "n": n, "seed": seed, "seconds": float(saved["seconds"]),
                       **quality(X, saved["Z"], labels, seed=seed)}
                if "model" in saved:
                    row["fitted"] = json.loads(str(saved["model"]))["fitted"]
                rows.append(row)
    settings = {name: defaults(method) for name, method in METHODS.items()}
    for name in ("lvm", "lvm_chart"):
        models = {n: np.load(HERE / "embeddings" / f"{name}_n{n}_seed0.npz")["model"] for n in SIZES}
        settings[name] = {n: json.loads(str(model))["settings"] for n, model in models.items()}
    environment = json.loads(str(np.load(HERE / "embeddings" / f"lvm_n{SIZES[0]}_seed{SEEDS[0]}.npz")["environment"]))
    results = {"dataset": "mnist8m", "split": "eval", "sizes": SIZES, "seeds": SEEDS, "environment": environment,
               "settings": settings, "rows": rows}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
