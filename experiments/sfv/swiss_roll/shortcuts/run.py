import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from experiments.common.datasets import load, unrolled
from experiments.common.guard import watch_memory
from experiments.common.record import environment, model_record
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent
N, SEEDS, SHORTCUT = 20_000, (0, 1, 2), 2.0
SETTINGS = {"default": {},
            "cells_1000": {"cells": {"n_cells": 1000}},
            "k_5": {"graph": {"radius": {"adaptive_per_cell": {"k": 5}}}}}


def shortcut_edges(model: LocalizedVoltageMaps) -> dict:
    centroids = model.root_.centroids.cpu().numpy()
    radius = model.root_.radius.cpu().numpy()
    ambient = cdist(centroids, centroids)
    sheet = cdist(unrolled(centroids), unrolled(centroids))
    joined = ambient <= np.maximum(radius[:, None], radius[None, :])
    np.fill_diagonal(joined, False)
    i, j = np.nonzero(np.triu(joined))
    ratio = sheet[i, j] / ambient[i, j]
    shortcuts = ratio > SHORTCUT
    return {"cells": len(centroids), "edges": len(i), "shortcuts": int(shortcuts.sum()),
            "cells_with_shortcut": len(np.unique(np.concatenate([i[shortcuts], j[shortcuts]]))),
            "radius_median": float(np.median(radius)), "radius_max": float(radius.max()),
            "ratio_max": float(ratio.max())}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    rows, records = [], {}
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0].astype(np.float32)
        for name, config in SETTINGS.items():
            model = LocalizedVoltageMaps(local_chart="none", random_state=seed, config=config)
            Z = model.fit(X).transform(X)
            np.savez(HERE / "embeddings" / f"{name}_seed{seed}.npz", Z=Z)
            model.save(HERE / "models" / f"{name}_seed{seed}")
            rows.append({"setting": name, "seed": seed, **shortcut_edges(model)})
            records[f"{name}_seed{seed}"] = model_record(model)
    results = {"dataset": "swiss_roll", "n": N, "seeds": SEEDS, "shortcut_ratio": SHORTCUT,
               "layer_spacing": 2 * np.pi, "settings": SETTINGS, "environment": environment(model.device_),
               "models": records, "rows": rows}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
