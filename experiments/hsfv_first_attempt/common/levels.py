import shutil
from pathlib import Path

import numpy as np
import torch

from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells
from lvm.config import load_config
from lvm.data import as_chunks
from lvm.region import fit_region, load_region, save_region

MIN_POINTS = 100


def fit_two_levels(X: np.ndarray, *, config: dict, level1_cells: int, seed: int, folder: Path) -> None:
    """Level 1 on all points with ``level1_cells`` cells; its landmarks' basins are the regions; level 2 is fitted inside each region with its
    cells split from the region's level-1 cells. Saves both levels' models and each point's region in ``folder``.
    Regions with fewer than MIN_POINTS points get no level 2: their points keep level 1."""
    if folder.exists():
        shutil.rmtree(folder)
    level1 = LocalizedVoltageMaps(local_chart="none", random_state=seed,
                                  config={**config, "cells": {"n_cells": level1_cells}}).fit(X)
    level1.save(folder / "level1")
    fit_level2(X, level1.root_, config=config, seed=seed, folder=folder)


def fit_level2(X: np.ndarray, root, *, config: dict, seed: int, folder: Path) -> None:
    """The regions are the basins of the root's landmarks (each cell goes to the landmark with the highest voltage
    there, each point to its nearest cell's region); level 2 is fitted inside each region with its cells split from
    the region's root cells. Saves each region's model and each point's region in ``folder``. Regions with fewer
    than MIN_POINTS points get no level 2: their points keep level 1."""
    cell_region = root.maps.argmax(dim=0)
    points = torch.as_tensor(X, device=root.centroids.device)
    region_of_point = cell_region[assign_cells(points, root.centroids.to(points.dtype))].cpu().numpy()
    region_config = load_config(overrides={**config, "embedding": {"local_chart": "none"}})
    (folder / "level2").mkdir(parents=True)
    for r in np.unique(region_of_point):
        members = np.flatnonzero(region_of_point == r)
        if len(members) < MIN_POINTS:
            continue
        region = fit_region(as_chunks(X[members], region_config.data.chunk_size), region_config,
                            device=str(root.centroids.device), region_id=(int(r),), seed=seed,
                            parents=root.centroids[cell_region == r])
        save_region(region, folder / "level2" / f"region_{r}.npz")
    np.save(folder / "region_of_point.npy", region_of_point)


def load_two_levels(folder: Path) -> tuple[LocalizedVoltageMaps, dict, np.ndarray]:
    """(level 1, {region: its level-2 model}, each point's region), as fit_two_levels saved them."""
    level1 = LocalizedVoltageMaps.load(folder / "level1")
    level2 = {int(p.stem.split("_")[1]): load_region(p, level1.device_) for p in (folder / "level2").glob("region_*.npz")}
    return level1, level2, np.load(folder / "region_of_point.npy")
