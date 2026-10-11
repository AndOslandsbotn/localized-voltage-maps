import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment, model_record

HERE = Path(__file__).resolve().parent
N_FIT, SEED = 50_000, 0


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = np.concatenate(list(stream("mnist8m", n=N_FIT)))
    model = fit_lvm(X, seed=SEED)
    Z = np.concatenate([model.transform(chunk) for chunk in stream("mnist8m")]).astype(np.float32)
    np.savez(HERE / "embeddings" / "lvm.npz", Z=Z, n_fit=N_FIT, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))
    model.save(HERE / "models" / "lvm")


if __name__ == "__main__":
    main()
