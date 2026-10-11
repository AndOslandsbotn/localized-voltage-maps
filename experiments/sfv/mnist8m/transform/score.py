import json
from pathlib import Path

import numpy as np

from experiments.common import methods
from experiments.common.datasets import mnist8m_points
from experiments.common.guard import watch_memory
from experiments.common.metrics import quality
from experiments.common.record import defaults
from lvm.config import load_config

HERE = Path(__file__).resolve().parent
N_FIT, SEEDS, SAMPLE = 50_000, (0, 1, 2), 5_000
METHODS = {"lvm": methods.fit_lvm, "lvm_chart": methods.fit_lvm_chart, "umap": methods.fit_umap_gpu,
           "landmark_isomap": methods.fit_landmark_isomap_gpu}


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for name in METHODS:
        for seed in SEEDS:
            saved = np.load(HERE / "embeddings" / f"{name}_seed{seed}.npz")
            X, labels = mnist8m_points(saved["rows"])
            n_placed, transform_seconds = int(saved["n_placed"]), float(saved["transform_seconds"])
            rows.append({"method": name, "seed": seed, "fit_seconds": float(saved["fit_seconds"]),
                         "transform_seconds": transform_seconds, "n_placed": n_placed,
                         "points_per_second": n_placed / transform_seconds,
                         **quality(X, saved["Z"], labels, seed=seed)})
    settings = {name: defaults(method) for name, method in METHODS.items()}
    settings["lvm"] = settings["lvm_chart"] = load_config().model_dump(mode="json")
    environment = json.loads(str(np.load(HERE / "embeddings" / f"lvm_seed{SEEDS[0]}.npz")["environment"]))
    results = {"dataset": "mnist8m", "n_fit": N_FIT, "seeds": SEEDS, "sample": SAMPLE,
               "not_placing": ["tsne", "laplacian_eigenmaps"], "environment": environment, "settings": settings,
               "rows": rows}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
