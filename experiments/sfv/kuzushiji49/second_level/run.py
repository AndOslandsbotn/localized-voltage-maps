import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.datasets import DATA, load, stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment, model_record
from lvm.cells import assign_cells

HERE = Path(__file__).resolve().parent
DATASET, N_FIT, SEEDS, MIN_POINTS = "kuzushiji49", 50_000, (0, 1, 2), 1_000
MASSES = {"cells": {"masses": {"rel_error": None}}}
LEVEL2 = {"same": (fit_lvm, "same"), "new": (fit_lvm, "new")}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    labels = np.fromfile(DATA / DATASET / "y.i8", dtype=np.int8).astype(np.int64)
    warm_up = load("eval", n=2000, seed=0, dataset=DATASET)[0]
    fit_lvm(warm_up, seed=0).transform(warm_up)
    for seed in SEEDS:
        path = HERE / "embeddings" / f"seed{seed}.npz"
        if not path.exists():
            run(seed, labels, path)


def run(seed: int, labels: np.ndarray, path: Path) -> None:
    X = load("eval", n=N_FIT, seed=0, dataset=DATASET)[0]
    start = time.perf_counter()
    level1 = fit_lvm(X, seed=seed, config=MASSES)
    Z1 = level1.transform(X)
    seconds = {"level1": time.perf_counter() - start}
    root = level1.root_
    cell_region = root.maps.argmax(dim=0)
    region_of_row = route(stream(DATASET), root.centroids, cell_region)
    region_of_fit = region_of_row[:N_FIT]
    del X
    embeddings = {name: np.zeros_like(Z1) for name in LEVEL2}
    purity = {"level1": [], "same": [], "new": []}
    sizes = {"fit": [], "all": []}
    for name in LEVEL2:
        seconds[name] = 0.0
    for r in range(len(root.landmarks)):
        fit_rows = np.flatnonzero(region_of_fit == r)
        all_rows = np.flatnonzero(region_of_row == r)
        sizes["fit"].append(len(fit_rows))
        sizes["all"].append(len(all_rows))
        if len(fit_rows) < MIN_POINTS:
            continue
        new_points = rows_of(all_rows)
        points, region_labels = new_points[np.searchsorted(all_rows, fit_rows)], labels[fit_rows]
        purity["level1"].append(cell_purity(points, region_labels, root.centroids))
        for name, (fit, data) in LEVEL2.items():
            fitted_on = points if data == "same" else new_points
            config = MASSES if data == "same" else {"sample": {"size": len(all_rows)}, **MASSES}
            start = time.perf_counter()
            model = fit(fitted_on, seed=seed, config=config)
            embeddings[name][fit_rows] = model.transform(points)
            seconds[name] += time.perf_counter() - start
            if name in purity:
                purity[name].append(cell_purity(points, region_labels, model.root_.centroids))
        del new_points, points
    np.savez(path, level1=Z1, **embeddings, region_of_fit=region_of_fit, region_of_row=region_of_row,
             sizes=json.dumps(sizes), purity=json.dumps(purity), seconds=json.dumps(seconds),
             environment=json.dumps(environment("cuda")), level1_model=json.dumps(model_record(level1)))


def route(data, centroids: torch.Tensor, cell_region: torch.Tensor) -> np.ndarray:
    """The level-1 region of every point: the landmark basin of its nearest cell."""
    regions = [cell_region[assign_cells(torch.as_tensor(chunk, device=centroids.device), centroids)].cpu().numpy()
               for chunk in data]
    return np.concatenate(regions)


def rows_of(rows: np.ndarray) -> np.ndarray:
    """The given rows of the dataset, read in one streamed pass."""
    kept, offset = [], 0
    for chunk in stream(DATASET):
        inside = rows[(rows >= offset) & (rows < offset + len(chunk))]
        kept.append(chunk[inside - offset])
        offset += len(chunk)
    return np.concatenate(kept)


def cell_purity(points: np.ndarray, labels: np.ndarray, centroids: torch.Tensor) -> float:
    """Share of the points whose label is the majority label of their nearest cell."""
    cells = assign_cells(torch.as_tensor(points, device=centroids.device), centroids).cpu().numpy()
    majority = 0
    for c in np.unique(cells):
        majority += np.bincount(labels[cells == c]).max()
    return float(majority / len(points))


if __name__ == "__main__":
    main()
