import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.datasets import unrolled
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import load_two_levels
from experiments.hsfv_first_attempt.common.repair import _detour_flags, children, voltage_flags
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import SEEDS
from experiments.hsfv_first_attempt.swiss_roll.noise.run import NOISE, noisy
from lvm.cells import assign_cells

HERE = Path(__file__).resolve().parent


def true_shortcuts(root, X: np.ndarray, clean: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Short-cuts judged with the clean positions of each root cell's points: close in space, far along the sheet."""
    cells = assign_cells(torch.as_tensor(X, device=root.centroids.device), root.centroids.to(torch.float32)).cpu().numpy()
    space = np.array([clean[cells == c].mean(axis=0) for c in range(len(root.centroids))])
    sheet = unrolled(space)
    along = np.linalg.norm(sheet[edges[:, 0]] - sheet[edges[:, 1]], axis=1)
    return along > 2 * np.linalg.norm(space[edges[:, 0]] - space[edges[:, 1]], axis=1) + 1.0


def main() -> None:
    """Short-cut detection on the noisy swiss roll's coarse roots: level 2's grounded voltages against the detour test
    (level 2's graph shortest paths), scored on the true short-cuts."""
    watch_memory(limit_mb=3000)
    rows = []
    for sigma in NOISE:
        for seed in SEEDS:
            clean, X = noisy(seed, sigma)
            level1, level2, _ = load_two_levels(HERE / "models" / f"noise{sigma}" / f"seed{seed}" / "base")
            root, config = level1.root_, level1.config_
            edges, _, by_voltage = voltage_flags(root, level2, config)
            centroids, radius, parent = children(root, level2)
            by_detour = _detour_flags(edges, root.centroids.double().cpu().numpy(), centroids, radius, parent, config, 2.0)
            true = true_shortcuts(root, X, clean, edges)
            for name, flags in (("detour", by_detour), ("voltage", by_voltage)):
                rows.append({"noise": sigma, "seed": seed, "test": name, "edges": len(edges),
                             "true_shortcuts": int(true.sum()), "flagged": int(flags.sum()),
                             "recall": float((flags & true).sum() / max(true.sum(), 1)),
                             "precision": float((flags & true).sum() / max(flags.sum(), 1))})
                print(rows[-1], flush=True)
    (HERE / "detection_voltage.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
