import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors

from experiments.common.guard import watch_memory
from experiments.sfv.mnist8m.stability.run_lvm import SEEDS

HERE = Path(__file__).resolve().parent
METHODS = ("lvm", "lvm_chart", "umap_seeded", "tsne", "laplacian_eigenmaps", "landmark_isomap")
K, SAMPLE = 10, 10_000


def neighbour_agreement(A: np.ndarray, B: np.ndarray) -> float:
    """Mean overlap (Jaccard) of each point's K nearest neighbours in two embeddings of the same points."""
    a = NearestNeighbors(n_neighbors=K).fit(A).kneighbors()[1]
    b = NearestNeighbors(n_neighbors=K).fit(B).kneighbors()[1]
    shared = np.array([len(np.intersect1d(x, y)) for x, y in zip(a, b)])
    return float(np.mean(shared / (2 * K - shared)))


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for method in METHODS:
        embeddings = [np.load(HERE / "embeddings" / f"{method}_seed{seed}.npz")["Z"] for seed in SEEDS]
        sample = np.random.default_rng(0).choice(len(embeddings[0]), SAMPLE, replace=False)
        for i, j in combinations(range(len(SEEDS)), 2):
            A, B = embeddings[i][sample], embeddings[j][sample]
            rows.append({"method": method, "seeds": [SEEDS[i], SEEDS[j]], "procrustes": float(procrustes(A, B)[2]),
                         "neighbour_agreement": neighbour_agreement(A, B)})
    (HERE / "results.json").write_text(json.dumps({"n": 50_000, "sample": SAMPLE, "k": K, "seeds": SEEDS,
                                                   "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
