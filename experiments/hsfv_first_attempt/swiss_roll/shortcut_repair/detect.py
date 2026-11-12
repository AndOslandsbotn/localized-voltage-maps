import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import unrolled
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import load_two_levels
from experiments.hsfv_first_attempt.common.repair import children, flagged_by_detour, flagged_edges, straddling
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import LEVEL1_CELLS, SEEDS

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "chained_distances" / "models"


def detours(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True where two locations are close in space but far along the sheet: a short-cut."""
    along = np.linalg.norm(unrolled(a) - unrolled(b), axis=1)
    return along > 2 * np.linalg.norm(a - b, axis=1) + 1.0


def spans_layers(cells: np.ndarray) -> bool:
    """True if some two of the cells are close in space but far along the sheet (they lie on different layers)."""
    if len(cells) < 2:
        return False
    i, j = np.triu_indices(len(cells), 1)
    return bool(detours(cells[i], cells[j]).any())


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for setting in LEVEL1_CELLS:
        for seed in SEEDS:
            level1, level2, _ = load_two_levels(MODELS / setting / f"seed{seed}")
            root, config = level1.root_, level1.config_
            C = root.centroids.double().cpu().numpy()
            edges, touch = flagged_edges(root, level2, config)
            _, flags = flagged_by_detour(root, level2, config)
            true = detours(C[edges[:, 0]], C[edges[:, 1]])
            groups = straddling(root, level2, config)
            kids, _, parent = children(root, level2)
            straddles = np.array([spans_layers(kids.cpu().numpy()[parent == a]) for a in range(len(C))])
            rows.append({"setting": setting, "seed": seed, "cells": len(C), "edges": len(edges),
                         "true_shortcuts": int(true.sum()), "touch_flagged": int(touch.sum()),
                         "touch_flagged_and_true": int((touch & true).sum()), "flagged": int(flags.sum()),
                         "flagged_and_true": int((flags & true).sum()),
                         "recall": float((flags & true).sum() / max(true.sum(), 1)),
                         "precision": float((flags & true).sum() / max(flags.sum(), 1)),
                         "straddling_detected": int((groups > 1).sum()), "straddling_true": int(straddles.sum()),
                         "straddling_detected_and_true": int(((groups > 1) & straddles).sum())})
            print(rows[-1], flush=True)
    (HERE / "detection.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
