import json
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from experiments.common.datasets import load, unrolled
from experiments.common.guard import watch_memory

HERE = Path(__file__).resolve().parent
EMBEDDINGS = HERE.parent / "embedding" / "embeddings"
N, SEEDS = 20_000, (0, 1, 2)
METHODS = ("lvm", "lvm_chart", "umap", "tsne", "laplacian_eigenmaps", "landmark_isomap")


def along_roll(sheet: np.ndarray, Z: np.ndarray) -> float:
    """Absolute rank correlation between a point's position along the roll (its arc length on the flat sheet) and
    its position along the map's main axis (first principal direction): 1 when the roll is unrolled in order."""
    centred = Z - Z.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return float(abs(spearmanr(sheet[:, 0], centred @ axis)[0]))


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0]
        sheet = unrolled(X)
        for name in METHODS:
            Z = np.load(EMBEDDINGS / f"{name}_seed{seed}.npz")["Z"]
            rows.append({"method": name, "seed": seed, "procrustes": float(procrustes(sheet, Z)[2]),
                         "along_roll": along_roll(sheet, Z)})
    (HERE / "results.json").write_text(json.dumps({"dataset": "swiss_roll", "split": "eval", "n": N, "seeds": SEEDS,
                                                   "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
