import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.datasets import half_sphere
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import (geodesic_curves, global_coordinates, global_within_root,
                                                    landmark_graph)
from experiments.hsfv_first_attempt.common.overlap import load_overlapping, within_cell
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, N, SEEDS
from experiments.hsfv_first_attempt.half_sphere.chained_distances.score import SAMPLE, scores, truth
from experiments.hsfv_first_attempt.half_sphere.noise.run import NOISE, noisy
from experiments.hsfv_first_attempt.half_sphere.overlapping_regions.score import condensed
from lvm.cells import assign_cells
from lvm.config import load_config
from lvm.embedding import classical_mds, triangulate_distances
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
DEVICE = "cuda"
EXTRA, CAP, K = (0, 100_000, 500_000), 256, 10
GROUPS = ("all", "short", "middle", "long")


def extra_points(seed: int, sigma: float, n: int) -> np.ndarray:
    """New points from the same shape with the same noise (the stream)."""
    X = half_sphere(n, 5000 + seed).astype(np.float32)
    return X + np.random.default_rng(6000 + seed).normal(scale=sigma, size=X.shape).astype(np.float32)


def leaf_within(graph: dict, root, level2: dict, points: torch.Tensor, regions: np.ndarray, config,
                coords: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    """Each point inside its strongest region's level-2 cell in the global frame, the cell's local PCA from its own
    points plus up to CAP streamed points routed to it (root cell, strongest region, nearest level-2 cell)."""
    n = sum(graph["sizes"])
    mean_sq, projection = classical_mds(graph["paths"], n)
    primary = torch.as_tensor(regions[:, 0], device=points.device)
    strongest = root.maps.argmax(dim=0)
    extra_region = strongest[assign_cells(extra, root.centroids.to(extra.dtype))] if len(extra) else extra[:0, 0].long()
    out = coords.clone()
    for r in graph["keys"]:
        members = torch.nonzero(primary == r).squeeze(1)
        if len(members) == 0:
            continue
        child = level2[r]
        o, k = graph["offset"][r], graph["sizes"][graph["keys"].index(r)]
        v = child.maps.T.double()
        cells = triangulate_distances(v @ graph["paths"][o:o + k] / v.sum(dim=1, keepdim=True), mean_sq, projection)
        mine = extra[extra_region == r]
        if len(mine):
            cell = assign_cells(mine, child.centroids.to(mine.dtype))
            order = torch.argsort(cell, stable=True)
            rank = torch.arange(len(cell), device=cell.device) - (torch.cumsum(torch.bincount(cell, minlength=len(child.centroids)), 0)
                                                                 - torch.bincount(cell, minlength=len(child.centroids)))[cell[order]]
            mine = mine[order[rank < CAP]]
        pca = torch.cat([points[members], mine])
        out[members] = within_cell(child, points[members], config, directions="per_cell", cell_coords=cells,
                                   pca_points=pca).to(out.dtype)
    return out


def leaf_within_mean(level2: dict, points: torch.Tensor, regions: np.ndarray, config,
                     coords: torch.Tensor) -> torch.Tensor:
    """Like leaf_within without streamed points, but each level-2 cell anchored where its points are on average in
    the global frame (as the root cells are), instead of placed by its voltages."""
    primary = torch.as_tensor(regions[:, 0], device=points.device)
    out = coords.clone()
    for r, child in level2.items():
        members = torch.nonzero(primary == r).squeeze(1)
        if len(members) == 0:
            continue
        cell = assign_cells(points[members], child.centroids.to(points.dtype))
        counts = torch.bincount(cell, minlength=len(child.centroids)).clamp(min=1)
        anchors = torch.zeros((len(child.centroids), coords.shape[1]), dtype=coords.dtype, device=coords.device)
        anchors = anchors.index_add_(0, cell, coords[members]) / counts[:, None].to(coords.dtype)
        out[members] = within_cell(child, points[members], config, directions="per_cell",
                                   cell_coords=anchors).to(out.dtype)
    return out


def recall(coords: torch.Tensor, S: torch.Tensor, true_full: torch.Tensor) -> float:
    """Recall@K of each sampled point's true K nearest neighbours (clean geodesic distances, among all points)."""
    d = torch.cdist(coords[S], coords)
    d[torch.arange(len(S)), S] = torch.inf
    found = torch.topk(d, K, dim=1, largest=False).indices
    t = true_full.clone()
    t[torch.arange(len(S)), S] = torch.inf
    true = torch.topk(t, K, dim=1, largest=False).indices
    return float((found[:, :, None] == true[:, None, :]).any(dim=2).float().mean())


def clean_distances(clean: np.ndarray, sample: np.ndarray) -> torch.Tensor:
    """(len(sample), N) true geodesic distances from the sampled points to all points."""
    from experiments.common.datasets import unrolled
    if "half_sphere" == "half_sphere":
        U = torch.as_tensor(clean / np.linalg.norm(clean, axis=1, keepdims=True), device=DEVICE, dtype=torch.float64)
        return torch.arccos((U[torch.as_tensor(sample, device=DEVICE)] @ U.T).clamp(-1, 1))
    U = torch.as_tensor(unrolled(clean), device=DEVICE, dtype=torch.float64)
    return torch.cdist(U[torch.as_tensor(sample, device=DEVICE)], U)


def main() -> None:
    """Toys: within-cell positions from the level-2 cells with their local PCA filled by streamed points (0, 100k,
    500k new points), against root cells and no within-cell step; true distances and true neighbours."""
    watch_memory(limit_mb=3000)
    config = load_config(overrides=CONFIG)
    rows = []
    for sigma in NOISE:
        for seed in SEEDS:
            folder = HERE / "models" / f"noise{sigma}" / f"seed{seed}" / "default"
            clean, X = noisy(seed, sigma)
            points = torch.as_tensor(X, device=DEVICE)
            sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
            S = torch.as_tensor(sample, device=DEVICE)
            true = truth(clean[sample])
            true_full = clean_distances(clean, sample)
            root = load_region(folder / "root.npz", DEVICE)
            level2, regions, strengths = load_overlapping(folder / "m2", DEVICE)
            curves = {r: geodesic_curves(level2[r], config) for r in level2}
            graph = landmark_graph(root, level2, points, regions, config, curves=curves)
            coords = global_coordinates(graph, level2, points, regions, strengths, torch.arange(N, device=DEVICE), config)
            variants = {"hierarchy": coords, "root_cells": global_within_root(root, points, coords, config)}
            for extra in EXTRA:
                more = torch.as_tensor(extra_points(seed, sigma, extra), device=DEVICE) if extra else points[:0]
                variants[f"leaf_cells_extra{extra}"] = leaf_within(graph, root, level2, points, regions, config, coords, more)
            variants["leaf_cells_mean_anchor"] = leaf_within_mean(level2, points, regions, config, coords)
            thirds = np.quantile(true, [1 / 3, 2 / 3])
            masks = {"all": np.ones(len(true), dtype=bool), "short": true <= thirds[0],
                     "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
            for method, c in variants.items():
                d = condensed(torch.cdist(c[S], c[S]))
                r10 = recall(c, S, true_full)
                for group in GROUPS:
                    rows.append({"noise": sigma, "seed": seed, "method": method, "group": group, "recall10": r10,
                                 **scores(d, true, masks[group])})
            print(sigma, seed, flush=True)
    (HERE / "results_streamed.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
