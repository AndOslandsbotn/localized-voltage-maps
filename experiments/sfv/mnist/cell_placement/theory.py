import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm, uniform_masses
from experiments.common.metrics import room_per_digit

HERE = Path(__file__).resolve().parent
N, SEEDS, SAMPLE, K, TAU = 50_000, (0, 1, 2), 5_000, 20, 1e-3
PLACEMENTS, MASSES = ("kmeans", "farthest"), ("uniform", "counted")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for seed in SEEDS:
        X, labels = load("eval", n=N, seed=seed)
        X = X.astype(np.float32)
        for placement in PLACEMENTS:
            for masses in MASSES:
                uniform = patch("lvm.region.cell_masses", uniform_masses) if masses == "uniform" else nullcontext()
                with uniform:
                    region = fit_lvm(X, seed=seed, config={"cells": {"placement": placement,
                                                                     "masses": {"rel_error": None}}}).root_
                log_density, log_rate = decay_against_density(region, X, seed)
                Z = np.load(HERE / "embeddings" / f"lvm_{placement}_{masses}_seed{seed}.npz")["Z"]
                digit_density, room = room_per_digit(X, Z, labels, seed=seed, size=SAMPLE)
                rows.append({"placement": placement, "masses": masses, "seed": seed,
                             "decay_slope": float(np.polyfit(log_density, log_rate, 1)[0]),
                             "log_density": log_density.tolist(), "log_rate": log_rate.tolist(),
                             "digit_log_density": digit_density.tolist(), "room": room.tolist(),
                             "room_vs_density": float(spearmanr(digit_density, np.log(room)).statistic)})
    (HERE / "theory.json").write_text(json.dumps({"n": N, "seeds": SEEDS, "k": K, "tau": TAU, "rows": rows}))


def decay_against_density(region, X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per cell: the data density around its centre, log p ~ -d log r_K (r_K the distance to the K-th nearest data
    point), and the rate at which -log v changes per unit distance along its graph edges, median over the edges and
    landmark maps where both ends have v >= tau."""
    centroids = region.centroids.cpu().numpy()
    radius = region.radius.cpu().numpy()
    maps = region.maps.cpu().numpy()
    landmarks = region.landmarks.cpu().numpy()
    sample = X[np.random.default_rng(seed).choice(len(X), 20_000, replace=False)]
    r_k = NearestNeighbors(n_neighbors=K).fit(sample).kneighbors(centroids)[0][:, -1]
    log_density = -region.dimension * np.log(r_k)

    distance = cdist(centroids, centroids)
    i, j = np.nonzero(np.triu(distance <= np.maximum(radius[:, None], radius[None, :]), k=1))
    log_v = np.log(np.maximum(maps, 1e-300))
    valid = (maps[:, i] >= TAU) & (maps[:, j] >= TAU)
    valid &= (i[None, :] != landmarks[:, None]) & (j[None, :] != landmarks[:, None])
    rate = np.abs(log_v[:, i] - log_v[:, j]) / distance[i, j]
    per_cell = [[] for _ in range(len(centroids))]
    for landmark, edge in zip(*np.nonzero(valid)):
        per_cell[i[edge]].append(rate[landmark, edge])
        per_cell[j[edge]].append(rate[landmark, edge])
    has = np.array([len(rates) > 0 for rates in per_cell])
    log_rate = np.log([np.median(rates) for rates, h in zip(per_cell, has) if h])
    return log_density[has], log_rate


if __name__ == "__main__":
    main()
