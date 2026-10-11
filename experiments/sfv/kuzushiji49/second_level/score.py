import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.metrics import quality
from experiments.sfv.kuzushiji49.second_level.run import DATASET, LEVEL2, MIN_POINTS, N_FIT, SEEDS

HERE = Path(__file__).resolve().parent
PROBE = HERE.parents[1] / "scaling_probe" / "embeddings"


def main() -> None:
    watch_memory(limit_mb=3000)
    X, labels = load("eval", n=N_FIT, seed=0, dataset=DATASET)
    global_runs = {name: np.load(PROBE / f"{DATASET}_{name}_n{N_FIT}.npz") for name in ("umap", "tsne")}
    rows, timings = [], []
    for seed in SEEDS:
        saved = np.load(HERE / "embeddings" / f"seed{seed}.npz")
        embeddings = {name: saved[name] for name in ("level1", *LEVEL2)}
        seconds = {name: value for name, value in json.loads(str(saved["seconds"])).items()}
        chart = np.load(HERE / "embeddings" / f"chart_seed{seed}.npz")
        for data in ("same", "new"):
            embeddings[f"{data}_chart"] = chart[data]
            seconds[f"{data}_chart"] = json.loads(str(chart["seconds"]))[data]
        for method in ("umap", "tsne"):
            local = np.load(HERE / "embeddings" / f"{method}_seed{seed}.npz")
            for data in ("same", "new"):
                embeddings[f"{method}_{data}"] = local[data]
                seconds[f"{method}_{data}"] = json.loads(str(local["seconds"]))[data]
            embeddings[f"{method}_global"] = global_runs[method]["Z"]
            seconds[f"{method}_global"] = float(global_runs[method]["seconds"])
        routing = json.loads((HERE / "routing_seconds.json").read_text())[str(seed)]
        for name in ("same", "new", "same_chart", "new_chart", "umap_same", "umap_new", "tsne_same", "tsne_new"):
            seconds[name] += seconds["level1"] + routing
        timings.append({"seed": seed, "seconds": seconds, "routing": routing})
        purity = json.loads(str(saved["purity"]))
        sizes = json.loads(str(saved["sizes"]))
        kept = 0
        for r, n_fit in enumerate(sizes["fit"]):
            if n_fit < MIN_POINTS:
                rows.append({"seed": seed, "region": r, "n_fit": n_fit, "n_all": sizes["all"][r]})
                continue
            members = np.flatnonzero(saved["region_of_fit"] == r)
            rows.append({"seed": seed, "region": r, "n_fit": n_fit, "n_all": sizes["all"][r],
                         "purity": {k: v[kept] for k, v in purity.items()},
                         "quality": {name: quality(X[members], Z[members], labels[members], seed=seed)
                                     for name, Z in embeddings.items()}})
            kept += 1
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "n_fit": N_FIT, "seeds": SEEDS,
                                                   "min_points": MIN_POINTS, "timings": timings, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
