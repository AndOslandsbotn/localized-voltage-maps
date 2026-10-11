import json
from pathlib import Path

import numpy as np

from experiments.common.guard import watch_memory
from experiments.common.metrics import quality
from experiments.sfv.kuzushiji49.streaming.run_lvm import DATASET, SEEDS, SIZES, labels, read

HERE = Path(__file__).resolve().parent
METHODS = ("lvm", "lvm_minibatch", "lvm_lloyd", "umap", "tsne")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for n in SIZES:
        X, y = read(n), labels(n)
        for method in METHODS:
            for seed in SEEDS:
                path = HERE / "embeddings" / f"{method}_n{n}_seed{seed}.npz"
                if path.exists():
                    saved = np.load(path)
                    rows.append({"method": method, "n": n, "seed": seed, "seconds": float(saved["seconds"]),
                                 **quality(X, saved["Z"], y, seed=seed)})
        del X
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "sizes": SIZES, "seeds": SEEDS,
                                                   "data": "first n points, the same for every seed and method",
                                                   "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
