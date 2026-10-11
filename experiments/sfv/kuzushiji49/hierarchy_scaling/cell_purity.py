import json
import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.common.datasets import DATA, FileChunks, stream
from experiments.common.guard import watch_memory
from experiments.sfv.kuzushiji49.hierarchy_scaling.run_lvm import DATASET, N_FEATURES, REFINE, bucket
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells, refine_cells, split_cells
from lvm.config import load_config

HERE = Path(__file__).resolve().parent
SIZES, SEEDS = (5_000, 20_000, 50_000, 100_000, 200_000, 270_912), (0, 1, 2)


def main() -> None:
    watch_memory(limit_mb=3000)
    labels = np.fromfile(DATA / DATASET / "y.i8", dtype=np.int8).astype(np.int64)
    config = load_config(overrides=REFINE)
    rows = []
    for n in SIZES:
        for seed in SEEDS:
            rows.append({"n": n, "seed": seed, **purities(n, seed, labels, config)})
            print(rows[-1], flush=True)
    (HERE / "cell_purity.json").write_text(json.dumps({"dataset": DATASET, "scored_on": "all 270,912 images",
                                                       "rows": rows}, indent=2))


def purities(n: int, seed: int, labels: np.ndarray, config) -> dict:
    """Cell purity of level 1 and of level 2 (cells split inside each region), both scored on every image."""
    data = stream(DATASET, n=n)
    root = LocalizedVoltageMaps(local_chart="none", random_state=seed, config=REFINE).fit(data).root_
    cell_region = root.maps.argmax(dim=0)
    folder = HERE / "buckets_purity"
    region_of_point = bucket(data, root.centroids, cell_region, folder)
    children = {}
    for r in np.unique(region_of_point):
        size = int((region_of_point == r).sum())
        parents = root.centroids[cell_region == r]
        if size < 3:
            children[int(r)] = parents
            continue
        chunks = FileChunks(folder / f"region_{r}.u8", 0, size, N_FEATURES, 10_000)
        sample = torch.as_tensor(np.concatenate(list(chunks)), device="cuda")
        cells = split_cells(sample, parents, config=config, device="cuda", seed=seed)
        children[int(r)] = refine_cells(chunks, sample, cells, config=config, device="cuda")[0]
    shutil.rmtree(folder)
    for r in range(len(root.landmarks)):
        children.setdefault(r, root.centroids[cell_region == r])

    n_classes = int(labels.max()) + 1
    level1 = torch.zeros((len(root.centroids), n_classes), dtype=torch.float64, device="cuda")
    level2 = {r: torch.zeros((len(c), n_classes), dtype=torch.float64, device="cuda") for r, c in children.items()}
    offset = 0
    for chunk in stream(DATASET):
        points = torch.as_tensor(chunk, device="cuda")
        y = torch.as_tensor(labels[offset:offset + len(chunk)], device="cuda")
        cells = assign_cells(points, root.centroids)
        level1.index_put_((cells, y), torch.ones(len(y), dtype=torch.float64, device="cuda"), accumulate=True)
        region = cell_region[cells]
        for r in torch.unique(region).tolist():
            inside = region == r
            child = assign_cells(points[inside], children[r].to(points.dtype))
            level2[r].index_put_((child, y[inside]), torch.ones(int(inside.sum()), dtype=torch.float64, device="cuda"),
                                 accumulate=True)
        offset += len(chunk)
    total = float(level1.sum())
    return {"level1": float(level1.max(dim=1).values.sum()) / total,
            "level2": sum(float(c.max(dim=1).values.sum()) for c in level2.values()) / total,
            "level2_cells": sum(len(c) for c in children.values())}


if __name__ == "__main__":
    main()
