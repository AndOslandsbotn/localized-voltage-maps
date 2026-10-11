import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from experiments.common.guard import watch_memory
from experiments.common.metrics import quality, room_per_digit
from experiments.sfv.mnist.distribution_shift.run import COMMON, CONDITIONS, METHODS, SEEDS, shifted

HERE = Path(__file__).resolve().parent
TAU = 1e-3
BASELINES = ("umap", "tsne", "laplacian_eigenmaps", "landmark_isomap")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for seed in SEEDS:
        data, labels = shifted(seed)
        common = np.isin(labels, COMMON)
        rare_digits = [d for d in range(10) if d not in COMMON]
        for method in METHODS:
            for condition in CONDITIONS:
                saved = np.load(HERE / "embeddings" / f"{method}_{condition}_seed{seed}.npz")
                _, room = room_per_digit(data, saved["Z"], labels, seed=seed)
                rows.append({"method": method, "condition": condition, "seed": seed,
                             **quality(data, saved["Z"], labels, seed=seed),
                             "room_common": float(np.mean(room[list(COMMON)]) / np.median(room[rare_digits])),
                             "decay_common": decay_ratio(saved, data, common),
                             "fitted": json.loads(str(saved["model"]))["fitted"]})
        for method in BASELINES:
            Z = np.load(HERE / "embeddings" / f"{method}_seed{seed}.npz")["Z"]
            _, room = room_per_digit(data, Z, labels, seed=seed)
            rows.append({"method": method, "condition": None, "seed": seed, **quality(data, Z, labels, seed=seed),
                         "room_common": float(np.mean(room[list(COMMON)]) / np.median(room[rare_digits]))})
    (HERE / "results.json").write_text(json.dumps({"dataset": "mnist", "seeds": SEEDS, "common": COMMON,
                                                   "rows": rows}, indent=2))


def decay_ratio(saved, data: np.ndarray, common: np.ndarray) -> float:
    """Median decay rate of -log v per unit distance along graph edges, in cells mostly holding the common digits,
    over the same in the other cells (edges where both ends have v >= tau, landmarks excluded)."""
    centroids, radius, maps, landmarks = saved["centroids"], saved["radius"], saved["maps"], saved["landmarks"]
    nearest = cdist(data, centroids, "sqeuclidean").argmin(axis=1)
    held = np.bincount(nearest, minlength=len(centroids))
    common_cell = np.bincount(nearest, weights=common, minlength=len(centroids)) > held / 2
    distance = cdist(centroids, centroids)
    i, j = np.nonzero(np.triu(distance <= np.maximum(radius[:, None], radius[None, :]), k=1))
    valid = (maps[:, i] >= TAU) & (maps[:, j] >= TAU)
    valid &= (i[None, :] != landmarks[:, None]) & (j[None, :] != landmarks[:, None])
    rate = np.abs(np.log(maps[:, i]) - np.log(maps[:, j])) / distance[i, j]
    both_common = common_cell[i] & common_cell[j]
    neither = ~common_cell[i] & ~common_cell[j]
    return float(np.median(rate[valid & both_common[None, :]]) / np.median(rate[valid & neither[None, :]]))


if __name__ == "__main__":
    main()
