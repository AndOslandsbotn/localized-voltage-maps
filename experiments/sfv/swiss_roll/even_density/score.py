import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.common.datasets import load, unrolled
from experiments.common.geodesics import rank_correlation
from experiments.sfv.swiss_roll.even_density.run import DATASETS, N, SEEDS
from experiments.sfv.swiss_roll.landmarks.score import aspect

HERE = Path(__file__).resolve().parent
SAMPLE = 1_500
NAMES = ("lvm_3", "lvm_12", "lvm_chart_3", "lvm_chart_12", "laplacian_eigenmaps")


def main() -> None:
    rows = []
    for dataset in DATASETS:
        X = load("eval", n=N, seed=0, dataset=dataset)[0]
        sheet = unrolled(X)
        sample = np.random.default_rng(0).choice(N, SAMPLE, replace=False)
        truth = pdist(sheet[sample])
        for seed in SEEDS:
            for name in NAMES:
                Z = np.load(HERE / "embeddings" / f"{dataset}_{name}_seed{seed}.npz")["Z"]
                rows.append({"dataset": dataset, "method": name, "seed": seed,
                             "geodesic": rank_correlation(pdist(Z[sample]), truth), "aspect": aspect(Z),
                             "true_aspect": aspect(sheet)})
    (HERE / "results.json").write_text(json.dumps({"n": N, "seeds": SEEDS, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
