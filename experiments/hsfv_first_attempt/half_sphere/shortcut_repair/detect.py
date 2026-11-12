import json
from pathlib import Path

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import load_two_levels
from experiments.hsfv_first_attempt.common.repair import flagged_by_detour, flagged_edges, straddling
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import LEVEL1_CELLS, SEEDS

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "chained_distances" / "models"


def main() -> None:
    """The half sphere has no short-cuts: every flagged edge is a false alarm."""
    watch_memory(limit_mb=3000)
    rows = []
    for setting in LEVEL1_CELLS:
        for seed in SEEDS:
            level1, level2, _ = load_two_levels(MODELS / setting / f"seed{seed}")
            root, config = level1.root_, level1.config_
            edges, touch = flagged_edges(root, level2, config)
            _, flags = flagged_by_detour(root, level2, config)
            rows.append({"setting": setting, "seed": seed, "cells": len(root.centroids), "edges": len(edges),
                         "touch_flagged": int(touch.sum()), "flagged": int(flags.sum()),
                         "straddling_detected": int((straddling(root, level2, config) > 1).sum())})
            print(rows[-1], flush=True)
    (HERE / "detection.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
