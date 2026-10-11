import json
import pickle
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_umap_gpu

HERE = Path(__file__).resolve().parent
N = 50_000


def main() -> None:
    watch_memory(limit_mb=3000)
    X = np.concatenate(list(stream("mnist8m", n=N)))
    sizes = json.loads((HERE / "model_size.json").read_text())
    sizes["bytes"]["umap"] = len(pickle.dumps(fit_umap_gpu(X, seed=0)))
    (HERE / "model_size.json").write_text(json.dumps(sizes, indent=2))


if __name__ == "__main__":
    main()
