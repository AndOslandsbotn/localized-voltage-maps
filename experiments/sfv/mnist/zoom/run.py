import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment, model_record
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells

HERE = Path(__file__).resolve().parent
ZOOMS = ((7, 9), (3, 5), (2, 5, 8))


def landmark_positions(model: LocalizedVoltageMaps) -> np.ndarray:
    """Where each landmark sits in the picture: its cell's centroid, placed like any point."""
    region = model.root_
    return model.transform(region.centroids[region.landmarks].cpu().numpy())


def main() -> None:
    watch_memory(limit_mb=3000)
    X, labels = load("eval", n=50_000, seed=0)
    X = X.astype(np.float32)

    # Level 0: LVM on all points. Each cell goes to the landmark with the highest voltage there, and each landmark is
    # named by the majority digit of its own cell (the labels only name landmarks; they never define the regions).
    level0 = LocalizedVoltageMaps(random_state=0)
    level0_embedding = level0.fit(X).transform(X)
    root = level0.root_
    cells = assign_cells(torch.as_tensor(X, device=level0.device_), root.centroids).cpu().numpy()
    region = root.maps.argmax(dim=0).cpu().numpy()[cells]
    landmark_cells = root.landmarks.cpu().numpy()
    landmark_digits = np.array([np.bincount(labels[cells == c], minlength=10).argmax() for c in landmark_cells])

    # Level 1: for each pair of digits, LVM fitted again on the points in the regions of their landmarks.
    results = {"labels": labels, "level0": level0_embedding, "region": region, "landmark_digits": landmark_digits,
             "level0_landmarks": landmark_positions(level0)}
    models = {"level0": model_record(level0)}
    for digits in ZOOMS:
        name = "zoom_" + "_".join(map(str, digits))
        landmarks = np.flatnonzero(np.isin(landmark_digits, digits))
        rows = np.flatnonzero(np.isin(region, landmarks))
        level1 = LocalizedVoltageMaps(random_state=0)
        results[f"{name}_rows"] = rows
        results[f"{name}_landmarks"] = landmarks
        results[f"{name}_level1"] = level1.fit(X[rows]).transform(X[rows])
        results[f"{name}_level1_landmarks"] = landmark_positions(level1)
        level1.save(HERE / "models" / name)
        models[name] = model_record(level1)

    np.savez(HERE / "embeddings.npz", **results)
    level0.save(HERE / "models" / "level0")
    record = {"dataset": "mnist", "split": "eval", "n": len(X), "n_features": X.shape[1], "zooms": ZOOMS,
              "environment": environment(level0.device_), "models": models}
    (HERE / "settings.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
