import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_umap_gpu
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N_FIT, SEEDS, SAMPLE = 50_000, (0, 1, 2), 5_000


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = np.concatenate(list(stream("mnist8m", n=N_FIT)))
    rest = stream("mnist8m", start=N_FIT)
    fit_umap_gpu(X[:2000], seed=0).transform(X[:2000])
    for seed in SEEDS:
        path = HERE / "embeddings" / f"umap_seed{seed}.npz"
        if path.exists():
            continue
        sample = np.sort(np.random.default_rng(seed).choice(rest.n, SAMPLE, replace=False))
        start = time.perf_counter()
        model = fit_umap_gpu(X, seed=seed)
        fit_seconds = time.perf_counter() - start
        start, placed, offset = time.perf_counter(), [], 0
        for chunk in rest:
            Z = model.transform(chunk)
            inside = sample[(sample >= offset) & (sample < offset + len(chunk))]
            placed.append(Z[inside - offset])
            offset += len(chunk)
        transform_seconds = time.perf_counter() - start
        np.savez(path, Z=np.concatenate(placed), rows=N_FIT + sample, fit_seconds=fit_seconds,
                 transform_seconds=transform_seconds, n_placed=offset, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
