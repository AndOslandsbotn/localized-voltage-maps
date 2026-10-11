import json
import time
from pathlib import Path

from experiments.common.datasets import load, stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.sfv.kuzushiji49.second_level.run import DATASET, MASSES, N_FIT, SEEDS, route

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    X = load("eval", n=N_FIT, seed=0, dataset=DATASET)[0]
    fit_lvm(X[:2000], seed=0).transform(X[:2000])
    seconds = {}
    for seed in SEEDS:
        root = fit_lvm(X, seed=seed, config=MASSES).root_
        start = time.perf_counter()
        route(stream(DATASET), root.centroids, root.maps.argmax(dim=0))
        seconds[seed] = time.perf_counter() - start
    (HERE / "routing_seconds.json").write_text(json.dumps(seconds, indent=2))


if __name__ == "__main__":
    main()
