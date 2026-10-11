import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_landmark_isomap_gpu, fit_lvm, fit_lvm_chart
from experiments.common.record import defaults

HERE = Path(__file__).resolve().parent
N = 50_000


def saved_bytes(model) -> int:
    with tempfile.TemporaryDirectory() as folder:
        model.save(folder)
        return sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file())


def main() -> None:
    watch_memory(limit_mb=3000)
    X = np.concatenate(list(stream("mnist8m", n=N)))
    n_landmarks = defaults(fit_landmark_isomap_gpu)["n_landmarks"]
    sizes = {"data": X.nbytes,
             "lvm": saved_bytes(fit_lvm(X, seed=0)),
             "lvm_chart": saved_bytes(fit_lvm_chart(X, seed=0)),
             "landmark_isomap": X.nbytes + 8 * (N * n_landmarks + n_landmarks + 2 * n_landmarks)}
    (HERE / "model_size.json").write_text(json.dumps({"n": N, "bytes": sizes,
                                                      "note": "bytes kept to place new points; landmark_isomap "
                                                              "counts its distance table (n x landmarks, float64) "
                                                              "plus the training data its neighbour search needs"}, indent=2))


if __name__ == "__main__":
    main()
