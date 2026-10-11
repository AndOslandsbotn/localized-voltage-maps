import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEEDS, COUNTS = 20_000, (0, 1, 2), (3, 6, 9, 12, 20)
PINNED = {"graph": {"radius": {"strategy": "knn", "knn": {"k": 10}}, "connect": False},
          "extension": {"strategy": "grounded"}}
VARIANTS = {"before": {**PINNED, "embedding": {"distance_floor": 1e-3}},
            "fixed": {**PINNED, "embedding": {"distance_floor": 1e-10}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=N, seed=0, dataset="half_sphere")[0].astype(np.float32)
    for seed in SEEDS:
        for count in COUNTS:
            for name, variant in VARIANTS.items():
                path = HERE / "embeddings" / f"{name}_{count}_seed{seed}.npz"
                if path.exists():
                    continue
                config = {**variant, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": count}}}}
                model = fit_lvm(X, seed=seed, config=config)
                region = model.root_
                np.savez(path, Z=model.transform(X), landmarks=region.centroids[region.landmarks].cpu().numpy(),
                         environment=json.dumps(environment("cuda")))
                model.save(HERE / "models" / path.stem)


if __name__ == "__main__":
    main()
