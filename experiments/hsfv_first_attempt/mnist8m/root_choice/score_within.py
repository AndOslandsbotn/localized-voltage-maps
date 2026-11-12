import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import (geodesic_curves, global_coordinates, global_within,
                                                    global_within_root, landmark_graph)
from experiments.hsfv_first_attempt.common.overlap import build_state, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from experiments.hsfv_first_attempt.mnist8m.overlapping_regions.score import QUERIES, neighbour_scores, true_neighbours
from experiments.hsfv_first_attempt.mnist8m.root_choice.run import DATASET, N, SAVED, SEEDS
from lvm import LocalizedVoltageMaps
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
DIRECTIONS = ("per_cell", "dimension")
WIDE, REFERENCE = 100, 2_000


def wide_scores(distance, points: torch.Tensor, queries: torch.Tensor, reference: torch.Tensor, block: int = 100) -> dict:
    """Beyond the closest neighbours: recall@WIDE of the true WIDE nearest neighbours, and the mean over queries of
    the Spearman correlation between our distances and the original-space distances to REFERENCE random points."""
    recall, spearman = [], []
    for start in range(0, len(queries), block):
        q = queries[start:start + block]
        ours = distance(q)
        ours[torch.arange(len(q)), q] = torch.inf
        orig = torch.cdist(points[q], points)
        orig[torch.arange(len(q)), q] = torch.inf
        found = torch.topk(ours, WIDE, dim=1, largest=False).indices
        true = torch.topk(orig, WIDE, dim=1, largest=False).indices
        recall.append((found[:, :, None] == true[:, None, :]).any(dim=2).float().mean(dim=1))
        a = torch.argsort(torch.argsort(ours[:, reference], dim=1), dim=1).double()
        b = torch.argsort(torch.argsort(orig[:, reference], dim=1), dim=1).double()
        a, b = a - a.mean(dim=1, keepdim=True), b - b.mean(dim=1, keepdim=True)
        spearman.append((a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)))
    return {"recall_wide": float(torch.cat(recall).mean()), "global_rank": float(torch.cat(spearman).mean())}


def main() -> None:
    """The hierarchy on the default 300-cell root with within-cell positions in the global frame (two ways of
    choosing each cell's directions), against the same without them and the best single level so far."""
    watch_memory(limit_mb=3000)
    X, y = read(DATASET, N)
    rows = []
    for seed in SEEDS:
        level1 = LocalizedVoltageMaps.load(SAVED / "root300" / f"seed{seed}" / "level1")
        config, device = level1.config_, level1.device_
        points, labels = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)
        queries = torch.as_tensor(np.random.default_rng(seed).choice(N, QUERIES, replace=False), device=device)
        truth = true_neighbours(points, queries)
        plain, p_regions, p_strengths = load_overlapping(SAVED / "root300" / f"seed{seed}" / "plain", device)
        state = build_state(X, level1.root_, plain, p_regions, p_strengths, config, within=True, directions="dimension")
        everyone = torch.arange(N, device=device)
        single = level1_distances(state, everyone, everyone[:1])
        del single
        reference = torch.as_tensor(np.random.default_rng(100 + seed).choice(N, REFERENCE, replace=False), device=device)
        f = lambda q: level1_distances(state, q, everyone)
        result = {"level1_300_within": neighbour_scores(f, queries, truth, labels) | wide_scores(f, points, queries, reference)}
        del state
        torch.cuda.empty_cache()
        root = load_region(SAVED / "root300" / f"seed{seed}" / "root_repaired.npz", device)
        level2, regions, strengths = load_overlapping(HERE / "models" / f"seed{seed}" / "default_m2", device)
        curves = {r: geodesic_curves(level2[r], config) for r in level2}
        graph = landmark_graph(root, level2, points, regions, config, curves=curves)
        coords = torch.cat([global_coordinates(graph, level2, points, regions, strengths, everyone[s:s + 20_000], config)
                            for s in range(0, N, 20_000)])
        f = lambda q: torch.cdist(coords[q], coords)
        result["hierarchy"] = neighbour_scores(f, queries, truth, labels) | wide_scores(f, points, queries, reference)
        for directions in DIRECTIONS:
            within = global_within(graph, level2, points, regions, config, coords, directions=directions)
            f = lambda q: torch.cdist(within[q], within)
            result[f"hierarchy_within_{directions}"] = (neighbour_scores(f, queries, truth, labels)
                                                        | wide_scores(f, points, queries, reference))
            del within
            torch.cuda.empty_cache()
        rooted = global_within_root(root, points, coords, config)
        f = lambda q: torch.cdist(rooted[q], rooted)
        result["hierarchy_within_root"] = neighbour_scores(f, queries, truth, labels) | wide_scores(f, points, queries,
                                                                                                    reference)
        del rooted
        torch.cuda.empty_cache()
        for method, s in result.items():
            rows.append({"seed": seed, "method": method, **s})
            print(rows[-1], flush=True)
    (HERE / "results_within.json").write_text(json.dumps({"dataset": DATASET, "n": N, "queries": QUERIES,
                                                          "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
