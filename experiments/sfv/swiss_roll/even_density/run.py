import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment
from experiments.sfv.swiss_roll.embedding.run_laplacian_eigenmaps import best_pair

HERE = Path(__file__).resolve().parent
N, SEEDS, DATASETS = 20_000, (0, 1, 2), ("swiss_roll", "swiss_roll_even")
GRAPH = {"graph": {"radius": {"adaptive_per_cell": {"k": 5}}}}
METHODS = {"lvm_3": 3, "lvm_12": 12}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    for dataset in DATASETS:
        X = load("eval", n=N, seed=0, dataset=dataset)[0].astype(np.float32)
        for seed in SEEDS:
            for name, count in METHODS.items():
                path = HERE / "embeddings" / f"{dataset}_{name}_seed{seed}.npz"
                if not path.exists():
                    config = {**GRAPH, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": count}}}}
                    model = fit_lvm(X, seed=seed, config=config)
                    np.savez(path, Z=model.transform(X), environment=json.dumps(environment("cuda")))
                    model.save(HERE / "models" / path.stem)
            path = HERE / "embeddings" / f"{dataset}_laplacian_eigenmaps_seed{seed}.npz"
            if not path.exists():
                Z, k, pair, _, _ = best_pair(X, seed)
                np.savez(path, Z=Z, n_neighbors=k, eigenvectors=np.array(pair))


if __name__ == "__main__":
    main()
