import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.metrics import quality
from experiments.sfv.mnist.distribution_shift.run_mass_curve import COUNTED, METHODS, SEEDS, shifted_rows

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    mnist, labels = load("eval")
    rows = []
    for seed in SEEDS:
        chosen = shifted_rows(labels, seed)
        data, data_labels = mnist[chosen], labels[chosen]
        for method in METHODS:
            for counted in COUNTED:
                saved = np.load(HERE / "embeddings" / f"curve_{method}_{counted}_seed{seed}.npz")
                rows.append({"method": method, "counted": counted, "seed": seed,
                             **quality(data, saved["Z"], data_labels, seed=seed)})
    (HERE / "mass_curve.json").write_text(json.dumps({"seeds": SEEDS, "counted": COUNTED, "smoothing": "add-one",
                                                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
