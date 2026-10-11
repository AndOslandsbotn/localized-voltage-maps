import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.common.datasets import load, unrolled
from experiments.common.geodesics import rank_correlation

HERE = Path(__file__).resolve().parent
SAMPLE = 1_500


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    for row in results["rows"]:
        X = load("eval", n=results["n"], seed=row["seed"], dataset="swiss_roll")[0]
        sample = np.random.default_rng(row["seed"]).choice(results["n"], SAMPLE, replace=False)
        Z = np.load(HERE / "embeddings" / f"{row['setting']}_seed{row['seed']}.npz")["Z"]
        row["exact"] = rank_correlation(pdist(Z[sample]), pdist(unrolled(X[sample])))
    results["sample"] = SAMPLE
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
