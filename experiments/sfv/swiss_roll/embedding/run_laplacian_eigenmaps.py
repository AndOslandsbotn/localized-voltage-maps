import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import SpectralEmbedding

from experiments.common.datasets import load, unrolled
from experiments.common.guard import watch_memory
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEEDS = 20_000, (0, 1, 2)
NEIGHBOURS, EIGENVECTORS = (5, 10), 10


def best_pair(X: np.ndarray, seed: int):
    """Laplacian eigenmaps at its best on the swiss roll: of the first EIGENVECTORS (scikit-learn's ARPACK solver,
    stable where cuML's is not for this many), the one following the position along the roll and the one following
    its height, for the neighbour count whose weaker of the two correlations is highest."""
    position, height = unrolled(X).T
    best = (-np.inf,)
    for k in NEIGHBOURS:
        Z = SpectralEmbedding(EIGENVECTORS, affinity="nearest_neighbors", n_neighbors=k,
                              random_state=seed).fit_transform(X)
        along = np.array([abs(spearmanr(z, position).statistic) for z in Z.T])
        across = np.array([abs(spearmanr(z, height).statistic) for z in Z.T])
        a = int(along.argmax())
        b = int(np.where(np.arange(EIGENVECTORS) == a, -1.0, across).argmax())
        if min(along[a], across[b]) > best[0]:
            best = (min(along[a], across[b]), Z[:, [a, b]], k, (a, b), along[a], across[b])
    return best[1:]


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    for seed in SEEDS:
        path = HERE / "embeddings" / f"laplacian_eigenmaps_seed{seed}.npz"
        if path.exists():
            continue
        X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0].astype(np.float32)
        Z, k, pair, along, across = best_pair(X, seed)
        np.savez(path, Z=Z, n_neighbors=k, eigenvectors=np.array(pair), position_correlation=along,
                 height_correlation=across, environment=json.dumps(environment("cpu")))


if __name__ == "__main__":
    main()
