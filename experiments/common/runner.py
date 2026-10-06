from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from experiments.common.guard import run_guarded

ROOT = Path(__file__).resolve().parents[2]


def embed(method: str, *, dataset: str, n: int, seed: int, folder: Path, split: str = "eval",
          recompute: bool = False) -> dict[str, np.ndarray]:
    """A method's embedding of n points of a dataset split: Z, labels, seconds, device. Computed in its own process
    under the memory guard, saved in the experiment's embeddings/ folder (not in git) and reused unless
    ``recompute``."""
    path = folder / "embeddings" / f"{method}_{dataset}_{split}_n{n}_seed{seed}.npz"
    if recompute or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "experiments.common.embed", method, dataset, split, str(n), str(seed),
                   str(path)]
        if run_guarded(command, cwd=ROOT) != 0:
            raise RuntimeError(f"{method} on {dataset} (n={n}, seed={seed}) failed")
    with np.load(path) as saved:
        return dict(saved)
