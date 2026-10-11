import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from experiments.common.datasets import load
from experiments.common.geodesics import rank_correlation
from experiments.common.metrics import quality
from experiments.sfv.half_sphere.tears.run import COUNTS, N, SEEDS, VARIANTS

HERE = Path(__file__).resolve().parent
K, SAMPLE = 10, 1_500


def torn(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Points with one of their K nearest neighbours on the sphere placed > 10% of the embedding's width away."""
    _, neighbours = NearestNeighbors(n_neighbors=K + 1).fit(X).kneighbors(X)
    far = np.linalg.norm(Z[neighbours[:, 1:]] - Z[:, None, :], axis=2).max(axis=1)
    width = np.linalg.norm(np.quantile(Z, 0.99, axis=0) - np.quantile(Z, 0.01, axis=0))
    return far > 0.1 * width


def main() -> None:
    X = load("eval", n=N, seed=0, dataset="half_sphere")[0]
    sample = np.random.default_rng(0).choice(N, SAMPLE, replace=False)
    great_circle = pdist(X[sample], lambda a, b: np.arccos(np.clip(a @ b, -1.0, 1.0)))
    rows = []
    for seed in SEEDS:
        for count in COUNTS:
            for name in VARIANTS:
                Z = np.load(HERE / "embeddings" / f"{name}_{count}_seed{seed}.npz")["Z"]
                scores = quality(X, Z, None, seed=seed)
                rows.append({"variant": name, "landmarks": count, "seed": seed, "torn_share": float(torn(X, Z).mean()),
                             "geodesic": rank_correlation(pdist(Z[sample]), great_circle),
                             "trustworthiness": scores["trustworthiness"], "continuity": scores["continuity"]})
    (HERE / "results.json").write_text(json.dumps({"n": N, "seeds": SEEDS, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
