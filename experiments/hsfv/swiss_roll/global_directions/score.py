import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv.common.directions import level1_graph, level2_graph, nested
from experiments.hsfv.common.levels import score_regions
from experiments.hsfv.swiss_roll.level2_regions.run import DATASET, SEED
from experiments.hsfv.swiss_roll.level2_regions.score import geodesic
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent
RUNS = {20_000: "level2_regions", 60_000: "level2_regions_60k"}
SAMPLE = 500
SHOWN = ("rank", "rank_close", "rank_middle", "rank_far", "recall@5", "recall@20", "recall@78", "recall@305",
         "recall@1203")


def main() -> None:
    watch_memory(limit_mb=3000)
    for n, run in RUNS.items():
        folder = HERE.parent / run
        X = load("eval", n=n, seed=SEED, dataset=DATASET)[0].astype(np.float32)
        saved = np.load(folder / "maps.npz")
        level1 = LocalizedVoltageMaps.load(folder / "models" / "level1")
        root, device, config = level1.root_, level1.device_, level1.config_
        Z_nested = nested(saved["level1_chart"], saved["level2_chart"], saved["region"])
        np.savez(HERE / f"nested_n{n}.npz", Z=Z_nested)
        level1_distances, graph1 = level1_graph(root, X, config, device)
        level2_distances, graph2 = level2_graph(folder / "models", X, saved["region"], config, device)
        maps = {"level1_chart": saved["level1_chart"], "nested": Z_nested,
                "level1_graph": level1_distances, "level2_graph": level2_distances}
        rows = score_regions(X, maps, np.zeros(n, dtype=int), geodesic, sample=SAMPLE, seed=SEED)
        graphs = {"level1": graph1, "level2": graph2}
        results = {"dataset": DATASET, "n": n, "seed": SEED, "sample": SAMPLE, "maps_from": run, "graphs": graphs,
                   "rows": rows}
        (HERE / f"results_n{n}.json").write_text(json.dumps(results, indent=2))
        print(f"== n = {n}, graphs {graphs}")
        print(f"{'':14s}" + "".join(f"{metric:>12s}" for metric in SHOWN))
        for row in rows:
            print(f"{row['map']:14s}" + "".join(f"{row[metric]:12.3f}" for metric in SHOWN))


if __name__ == "__main__":
    main()
