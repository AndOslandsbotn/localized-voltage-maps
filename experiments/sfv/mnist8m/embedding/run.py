import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import mnist8m_labels, stream
from experiments.common.guard import watch_memory
from experiments.common.record import environment, model_record
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    X = stream("mnist8m")                       # 8.1 million images, read from disk chunk by chunk
    labels = mnist8m_labels()

    lvm = LocalizedVoltageMaps(local_chart="none", random_state=0)
    lvm_embedding = lvm.fit(X).transform(X)

    lvm_chart = LocalizedVoltageMaps(local_chart="last", random_state=0)
    lvm_chart_embedding = lvm_chart.fit(X).transform(X)

    np.savez(HERE / "embeddings.npz", labels=labels, lvm=lvm_embedding, lvm_chart=lvm_chart_embedding)
    record = {"dataset": "mnist8m", "n": len(labels), "n_features": X.n_features,
              "environment": environment(lvm.device_),
              "models": {"lvm": model_record(lvm), "lvm_chart": model_record(lvm_chart)}}
    (HERE / "settings.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
