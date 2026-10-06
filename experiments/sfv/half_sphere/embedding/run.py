import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment, model_record
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    X, _ = load("eval", n=20_000, seed=0, dataset="half_sphere")
    X = X.astype(np.float32)

    lvm = LocalizedVoltageMaps(local_chart="none", random_state=0)
    lvm_embedding = lvm.fit(X).transform(X)

    lvm_chart = LocalizedVoltageMaps(local_chart="last", random_state=0)
    lvm_chart_embedding = lvm_chart.fit(X).transform(X)

    np.savez(HERE / "embeddings.npz", X=X, lvm=lvm_embedding, lvm_chart=lvm_chart_embedding)
    record = {"dataset": "half_sphere", "split": "eval", "n": len(X), "n_features": X.shape[1],
              "environment": environment(lvm.device_),
              "models": {"lvm": model_record(lvm), "lvm_chart": model_record(lvm_chart)}}
    (HERE / "settings.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
