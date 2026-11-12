import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import (geodesic_curves, global_coordinates, global_within_root,
                                                    landmark_graph)
from experiments.hsfv_first_attempt.common.overlap import load_overlapping, within_cell
from experiments.common.datasets import stream
from experiments.hsfv_first_attempt.common.real_data import read, read_rows
from experiments.hsfv_first_attempt.kuzushiji49.overlapping_regions.score import QUERIES, neighbour_scores, true_neighbours
from experiments.hsfv_first_attempt.kuzushiji49.root_choice.run import DATASET, N, SAVED, SEEDS
from experiments.hsfv_first_attempt.kuzushiji49.root_choice.score_within import REFERENCE, wide_scores
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells
from lvm.embedding import classical_mds, triangulate_distances
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
EXTRA, CAP, CHUNK = (0, 170_912), 256, 50_000


def route(root, level2: dict, extra: int) -> dict:
    """Stream the images after the first N: each goes to its nearest root cell, that cell's strongest region, and
    the nearest level-2 cell there; keep up to CAP per level-2 cell. Returns, per region, the kept rows."""
    kept = {r: [] for r in level2}
    filled = {r: torch.zeros(len(level2[r].centroids), dtype=torch.long, device="cuda") for r in level2}
    strongest = root.maps.argmax(dim=0)
    start = N
    for chunk in (stream(DATASET, start=N, n=extra, chunk_size=CHUNK) if extra else []):
        rows = np.arange(start, start + len(chunk))
        start += len(chunk)
        points = torch.as_tensor(chunk, device="cuda")
        region = strongest[assign_cells(points, root.centroids.to(points.dtype))]
        for r in level2:
            mine = torch.nonzero(region == r).squeeze(1)
            if len(mine) == 0:
                continue
            cell = assign_cells(points[mine], level2[r].centroids.to(points.dtype))
            for c in torch.unique(cell).tolist():
                room = CAP - int(filled[r][c])
                if room <= 0:
                    continue
                take = mine[cell == c][:room]
                kept[r].append(rows[take.cpu().numpy()])
                filled[r][c] += len(take)
    return {r: np.sort(np.concatenate(v)) if v else np.zeros(0, dtype=np.int64) for r, v in kept.items()}


def streamed_within(graph: dict, level2: dict, points: torch.Tensor, regions: np.ndarray, config,
                    coords: torch.Tensor, extra_rows: dict) -> torch.Tensor:
    """Every point inside its strongest region's LEVEL-2 cell, in the global frame, with each cell's local PCA
    estimated from its own points plus the streamed ones (per-cell direction bound)."""
    n = sum(graph["sizes"])
    mean_sq, projection = classical_mds(graph["paths"], n)
    primary = torch.as_tensor(regions[:, 0], device=points.device)
    out = coords.clone()
    for r in graph["keys"]:
        members = torch.nonzero(primary == r).squeeze(1)
        if len(members) == 0:
            continue
        child = level2[r]
        o, k = graph["offset"][r], graph["sizes"][graph["keys"].index(r)]
        v = child.maps.T.double()
        cells = triangulate_distances(v @ graph["paths"][o:o + k] / v.sum(dim=1, keepdim=True), mean_sq, projection)
        extra = torch.as_tensor(read_rows(DATASET, extra_rows[r]), device=points.device)
        pca = torch.cat([points[members], extra]) if len(extra) else points[members]
        out[members] = within_cell(child, points[members], config, directions="per_cell", cell_coords=cells,
                                   pca_points=pca).to(out.dtype)
    return out


def main() -> None:
    """Within-cell positions from the LEVEL-2 cells, their local PCA filled by streaming more images (0 up to all
    extra), against the root-cell version (= the single level locally)."""
    watch_memory(limit_mb=3000)
    X, y = read(DATASET, N)
    rows = []
    for seed in SEEDS:
        level1 = LocalizedVoltageMaps.load(SAVED / "root300" / f"seed{seed}" / "level1")
        config, device = level1.config_, level1.device_
        points, labels = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)
        queries = torch.as_tensor(np.random.default_rng(seed).choice(N, QUERIES, replace=False), device=device)
        reference = torch.as_tensor(np.random.default_rng(100 + seed).choice(N, REFERENCE, replace=False), device=device)
        truth = true_neighbours(points, queries)
        everyone = torch.arange(N, device=device)
        root = load_region(SAVED / "root300" / f"seed{seed}" / "root_repaired.npz", device)
        level2, regions, strengths = load_overlapping(HERE / "models" / f"seed{seed}" / "default_m2", device)
        curves = {r: geodesic_curves(level2[r], config) for r in level2}
        graph = landmark_graph(root, level2, points, regions, config, curves=curves)
        coords = torch.cat([global_coordinates(graph, level2, points, regions, strengths, everyone[s:s + 20_000], config)
                            for s in range(0, N, 20_000)])
        rooted = global_within_root(root, points, coords, config)
        f = lambda q: torch.cdist(rooted[q], rooted)
        result = {"root_cells": neighbour_scores(f, queries, truth, labels) | wide_scores(f, points, queries, reference)}
        del rooted
        for extra in EXTRA:
            extra_rows = route(root, level2, extra)
            leaf = streamed_within(graph, level2, points, regions, config, coords, extra_rows)
            f = lambda q: torch.cdist(leaf[q], leaf)
            result[f"leaf_cells_extra{extra}"] = (neighbour_scores(f, queries, truth, labels)
                                                    | wide_scores(f, points, queries, reference)
                                                    | {"streamed_kept": int(sum(len(v) for v in extra_rows.values()))})
            del leaf
            torch.cuda.empty_cache()
        for method, s in result.items():
            rows.append({"seed": seed, "method": method, **s})
            print(rows[-1], flush=True)
    (HERE / "results_streamed.json").write_text(json.dumps({"dataset": DATASET, "n": N, "cap": CAP, "rows": rows},
                                                           indent=2))


if __name__ == "__main__":
    main()
