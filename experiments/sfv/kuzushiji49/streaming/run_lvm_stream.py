import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment, model_record
from experiments.sfv.kuzushiji49.streaming.run_lvm import DATASET, SEEDS, SIZES

HERE = Path(__file__).resolve().parent
VARIANTS = {"lvm_minibatch": {"cells": {"refine": {"strategy": "minibatch", "passes": 1},
                                        "masses": {"rel_error": None}}},
            "lvm_lloyd": {"cells": {"refine": {"strategy": "lloyd", "passes": 10, "tolerance": 0.05},
                                    "masses": {"rel_error": None}}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    warm_up = stream(DATASET, n=2000)
    for config in VARIANTS.values():
        fit_lvm(warm_up, seed=0, config=config).transform(warm_up)
    for n in SIZES:
        for seed in SEEDS:
            for name, config in VARIANTS.items():
                path = HERE / "embeddings" / f"{name}_n{n}_seed{seed}.npz"
                if not path.exists():
                    embed(n, seed, config, path)


def embed(n: int, seed: int, config: dict, path: Path) -> None:
    data = stream(DATASET, n=n)
    start = time.perf_counter()
    model = fit_lvm(data, seed=seed, config=config)
    Z = model.transform(data)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))
    model.save(HERE / "models" / path.stem)


if __name__ == "__main__":
    main()
