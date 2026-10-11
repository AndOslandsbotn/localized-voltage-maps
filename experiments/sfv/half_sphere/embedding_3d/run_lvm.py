import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEED = 20_000, 0


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=N, seed=SEED, dataset="half_sphere")[0].astype(np.float32)
    model = fit_lvm(X, seed=SEED, n_components=3)
    Z = model.transform(X)
    np.savez(HERE / "embeddings" / "lvm.npz", Z=Z, environment=json.dumps(environment("cuda")))
    model.save(HERE / "models" / "lvm")


if __name__ == "__main__":
    main()
