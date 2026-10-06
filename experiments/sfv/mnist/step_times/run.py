import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.timing import StepTimer
from lvm.config import load_config
from lvm.data import as_chunks
from lvm.region import fit_region

HERE = Path(__file__).resolve().parent
N, SEEDS, DEVICE = 50_000, (0, 1, 2), "cuda"


def main() -> None:
    watch_memory(limit_mb=3000)
    config = load_config()
    X = load("eval", n=N, seed=0)[0].astype(np.float32)
    fit_region(as_chunks(X[:2000], 2000), config, device=DEVICE, region_id=(), seed=0)              # warm-up
    seconds = {}
    for seed in SEEDS:
        timer = StepTimer(DEVICE)
        fit_region(as_chunks(X, config.data.chunk_size), config, device=DEVICE, region_id=(), seed=seed, timer=timer)
        for step, s in timer.seconds.items():
            seconds.setdefault(step, []).append(s)
    off = ["refine"] if config.cells.refine.passes == 0 else []
    results = {"dataset": "mnist", "n": N, "seeds": list(SEEDS), "device": DEVICE, "off": off, "seconds": seconds}
    (HERE / "step_times.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
