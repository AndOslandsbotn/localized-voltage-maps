from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    X, labels = load("eval", n=50_000, seed=0)
    X = X.astype(np.float32)
    lvm = LocalizedVoltageMaps(local_chart="none", random_state=0).fit(X).transform(X)
    lvm_chart = LocalizedVoltageMaps(local_chart="last", random_state=0).fit(X).transform(X)
    np.savez(HERE / "embeddings.npz", lvm=lvm, lvm_chart=lvm_chart, labels=labels)


if __name__ == "__main__":
    main()
