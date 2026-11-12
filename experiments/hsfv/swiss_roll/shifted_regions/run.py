import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment
from experiments.hsfv.common.shifted import level1_kernel, shifted_versions
from experiments.hsfv.common.tree import fit_children
from experiments.hsfv.swiss_roll.level2_regions.run import CONFIG as PAPER1_CONFIG, DATASET
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent
N, SEED, LANDMARKS = 864_000, 4, 12
CONFIG = {**PAPER1_CONFIG, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": LANDMARKS}}}}


def main() -> None:
    watch_memory(limit_mb=4000)
    if (HERE / "versions.npz").exists():
        return
    X = load("eval", n=N, seed=SEED, dataset=DATASET)[0].astype(np.float32)
    level1 = LocalizedVoltageMaps(local_chart="none", random_state=SEED, config=CONFIG).fit(X)
    level1.save(HERE / "models" / "level1")
    versions = shifted_versions(level1.root_, level1_kernel(level1, X), level1.config_)
    saved = {}
    for version, info in versions.items():
        fit_children(X, level1.root_, (), np.arange(len(X)), info["maps"].argmax(dim=0), config=CONFIG, seed=SEED,
                     folder=HERE / "models" / f"version{version}", device=level1.device_)
        saved |= {f"version{version}_landmarks": info["landmarks"].cpu().numpy(),
                  f"version{version}_rho_g": info["rho_g"], f"version{version}_maps": info["maps"].cpu().numpy()}
    np.savez(HERE / "versions.npz", **saved, config=json.dumps(CONFIG),
             environment=json.dumps(environment(level1.device_)))


if __name__ == "__main__":
    main()
