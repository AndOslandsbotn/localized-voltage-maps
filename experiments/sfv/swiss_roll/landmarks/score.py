import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.common.datasets import load, unrolled
from experiments.common.geodesics import rank_correlation
from experiments.sfv.swiss_roll.landmarks.run import COUNTS, N, SEEDS

HERE = Path(__file__).resolve().parent
SAMPLE = 1_500


def aspect(points: np.ndarray) -> float:
    """Short side over long side: the ratio of the two principal standard deviations."""
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return float(spread[1] / spread[0])


def main() -> None:
    rows = []
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0]
        sheet = unrolled(X)
        sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
        truth = pdist(sheet[sample])
        for count in COUNTS:
            Z = np.load(HERE / "embeddings" / f"landmarks_{count}_seed{seed}.npz")["Z"]
            rows.append({"landmarks": count, "seed": seed, "exact": rank_correlation(pdist(Z[sample]), truth),
                         "aspect": aspect(Z), "true_aspect": aspect(sheet)})
    (HERE / "results.json").write_text(json.dumps({"n": N, "seeds": SEEDS, "sample": SAMPLE, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
