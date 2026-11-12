import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import build_state, distances, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from experiments.hsfv_first_attempt.kuzushiji49.coarse_root.run import DATASET, N, ROOT_CELLS, SEEDS
from experiments.hsfv_first_attempt.kuzushiji49.overlapping_regions.score import QUERIES, neighbour_scores, true_neighbours
from lvm import LocalizedVoltageMaps
from lvm.region import load_region

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    X, y = read(DATASET, N)
    rows = []
    for seed in SEEDS:
        level1_any = None
        for cells in ROOT_CELLS:
            folder = HERE / "models" / f"root{cells}" / f"seed{seed}"
            level1 = LocalizedVoltageMaps.load(folder / "level1")
            config, device = level1.config_, level1.device_
            points, labels = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)
            queries = torch.as_tensor(np.random.default_rng(seed).choice(N, QUERIES, replace=False), device=device)
            if level1_any is None:
                truth, level1_any = true_neighbours(points, queries), True
            everyone = torch.arange(N, device=device)
            changes = json.loads((folder / "repair.json").read_text())
            runs = {"plain": (level1.root_, folder / "plain"),
                    "repaired": (load_region(folder / "root_repaired.npz", device), folder / "repaired")}
            for name, (root, models) in runs.items():
                level2, regions, strengths = load_overlapping(models, device)
                for within in ((False, True) if name == "plain" else (True,)):
                    state = build_state(X, root, level2, regions, strengths, config, within=within)
                    tag = "_cells" if within else ""
                    methods = {f"level1_{name}{tag}": lambda q: level1_distances(state, q, everyone)}
                    if within:
                        methods[f"level2_{name}{tag}"] = lambda q: distances(state, q, everyone, "weighted")
                    for method, distance in methods.items():
                        rows.append({"root_cells": cells, "seed": seed, "method": method, **changes,
                                     **neighbour_scores(distance, queries, truth, labels)})
                        print(rows[-1], flush=True)
                    del state
                    torch.cuda.empty_cache()
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "n": N, "queries": QUERIES, "rows": rows},
                                                  indent=2))


if __name__ == "__main__":
    main()
