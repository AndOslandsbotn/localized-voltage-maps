import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.common import methods
from experiments.common.datasets import load, unrolled
from experiments.common.geodesics import graph_geodesics, rank_correlation
from experiments.common.guard import watch_memory
from experiments.common.record import defaults
from lvm.config import load_config

HERE = Path(__file__).resolve().parent
EMBEDDINGS = HERE.parent / "embedding" / "embeddings"
N, SEEDS, SAMPLE, NEIGHBOURS = 20_000, (0, 1, 2), 1_500, 10
GRAPH = {"graph": {"radius": {"adaptive_per_cell": {"k": 5}}}}
METHODS = {"lvm": methods.lvm, "lvm_chart": methods.lvm_chart, "umap": methods.umap_gpu, "tsne": methods.tsne_gpu,
           "laplacian_eigenmaps": methods.laplacian_eigenmaps_gpu, "landmark_isomap": methods.landmark_isomap_gpu}


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0].astype(np.float32)
        sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
        exact = pdist(unrolled(X[sample]))
        graph = graph_geodesics(X, sample, NEIGHBOURS)
        rows.append({"method": "graph", "seed": seed, "exact": rank_correlation(graph, exact)})
        for name in METHODS:
            Z = np.load(EMBEDDINGS / f"{name}_seed{seed}.npz")["Z"]
            embedded = pdist(Z[sample])
            rows.append({"method": name, "seed": seed, "exact": rank_correlation(embedded, exact),
                         "graph": rank_correlation(embedded, graph)})
    settings = {name: defaults(method) for name, method in METHODS.items()}
    settings["lvm"] = settings["lvm_chart"] = load_config(overrides=GRAPH).model_dump(mode="json")
    environment = json.loads(str(np.load(EMBEDDINGS / f"lvm_seed{SEEDS[0]}.npz")["environment"]))
    results = {"dataset": "swiss_roll", "split": "eval", "n": N, "seeds": SEEDS, "sample": SAMPLE,
               "graph_neighbours": NEIGHBOURS, "environment": environment, "settings": settings, "rows": rows}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
