import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import build_state, distances, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from experiments.hsfv_first_attempt.kuzushiji49.overlapping_regions.run import DATASET, MS, N, SEEDS
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent
QUERIES, K, BATCH = 2_000, 10, 100


def true_neighbours(points: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """(Q, K) each query's K nearest other points in the original space."""
    found = []
    for start in range(0, len(queries), BATCH):
        q = queries[start:start + BATCH]
        d = torch.cdist(points[q], points)
        d[torch.arange(len(q)), q] = torch.inf
        found.append(torch.topk(d, K, dim=1, largest=False).indices)
    return torch.cat(found)


def neighbour_scores(distance, queries: torch.Tensor, truth: torch.Tensor, labels: torch.Tensor) -> dict:
    """Recall@K of the true K nearest neighbours among the K nearest by distance, and kNN label accuracy."""
    recall, correct = [], []
    for start in range(0, len(queries), BATCH):
        q = queries[start:start + BATCH]
        d = distance(q)
        d[torch.arange(len(q)), q] = torch.inf
        found = torch.topk(d, K, dim=1, largest=False).indices
        recall.append((found[:, :, None] == truth[start:start + BATCH][:, None, :]).any(dim=2).float().mean(dim=1))
        correct.append((torch.mode(labels[found], dim=1).values == labels[q]).float())
    return {"recall": float(torch.cat(recall).mean()), "knn_accuracy": float(torch.cat(correct).mean())}


def main() -> None:
    watch_memory(limit_mb=3000)
    X, y = read(DATASET, N)
    rows = []
    for seed in SEEDS:
        folder = HERE / "models" / f"seed{seed}"
        level1 = LocalizedVoltageMaps.load(folder / "level1")
        root, config, device = level1.root_, level1.config_, level1.device_
        points, labels = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)
        queries = torch.as_tensor(np.random.default_rng(seed).choice(N, QUERIES, replace=False), device=device)
        truth = true_neighbours(points, queries)
        everyone = torch.arange(N, device=device)
        for m, within in [(m, within) for m in MS for within in (False, True)]:
            level2, regions, strengths = load_overlapping(folder / f"m{m}", device)
            state = build_state(X, root, level2, regions, strengths, config, within=within)
            tag = "_cells" if within else ""
            methods = ({f"level1{tag}": lambda q: level1_distances(state, q, everyone),
                        f"m1{tag}": lambda q: distances(state, q, everyone, "weighted")} if m == 1 else
                       {f"m{m}_weighted{tag}": lambda q: distances(state, q, everyone, "weighted")})
            for method, distance in methods.items():
                rows.append({"seed": seed, "method": method, **neighbour_scores(distance, queries, truth, labels)})
                print(rows[-1], flush=True)
            del state
            torch.cuda.empty_cache()
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "n": N, "queries": QUERIES, "k": K,
                                                   "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
