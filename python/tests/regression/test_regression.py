"""Regression: LVM's results on a fixed synthetic dataset, pinned so restructuring the code cannot change them.

The data: 3000 points in 20 dimensions, three groups of different spread (a tight blob, a wide blob, a curved strip),
so the per-cell radius, chaining and landmark selection all matter. Every intermediate result is compared, so a
failure says where results changed: cells, masses, radii, graph, rho_g, landmarks, voltage maps, embedding.

* cpu: CPU path with scikit-learn's k-means; reproducible to ~1e-6, compared tightly.
* cuda: the defaults as the experiments run them (cuML k-means, local PCA chart). cuML's k-means is not
  bit-reproducible (centroids move by ~0.01 between runs, the embedding by ~0.3% of its range), so it is compared
  with tolerances above that noise.

When a change of results is intended, regenerate the references (and say why in the commit):

    LVM_UPDATE_REFERENCE=1 pytest tests/regression
"""

import os
from pathlib import Path

import numpy as np
import pytest

from lvm.config import load_config
from lvm.pipeline import fit_level
from lvm.stream import array_source

REFERENCE = Path(__file__).with_name("reference")
UPDATE = os.environ.get("LVM_UPDATE_REFERENCE") == "1"
BASE = {"cells": {"n_cells": 150, "sample_size": 3000}}
CASES = {
    "cpu": {**BASE, "compute": {"device": "cpu"}, "cells": {**BASE["cells"], "kmeans": {"strategy": "sklearn"}}},
    "cuda": {**BASE, "compute": {"device": "cuda"}, "embedding": {"local_scale": {"strategy": "pca"}}},
}
# Relative tolerance per quantity, as a share of that quantity's largest absolute value.
TOLERANCE = {"cpu": 1e-5, "cuda": 0.05}


def _data() -> np.ndarray:
    rng = np.random.default_rng(0)
    rotation = np.linalg.qr(rng.normal(size=(20, 20)))[0]
    t = rng.uniform(0, 3 * np.pi, 1000)
    groups = [rng.normal(0, 0.3, (1000, 3)),                                       # tight blob
              rng.normal(0, 1.5, (1000, 3)) + [8, 0, 0],                           # wide blob
              np.c_[np.cos(t) * 4, np.sin(t) * 4, t] + [0, 10, 0] + rng.normal(0, 0.1, (1000, 3))]   # strip
    X = np.vstack([np.c_[g, rng.normal(0, 0.05, (len(g), 17))] for g in groups]) @ rotation
    return X[rng.permutation(len(X))]


def _results(case: str) -> dict[str, np.ndarray]:
    X = _data()
    model = fit_level(array_source(X, 500), load_config(overrides=CASES[case]))
    return {
        "d_hat": np.array(model.dimension.d),
        "centroids": model.centroids,
        "masses": model.masses.p,
        "radius_per_cell": model.r_cells,
        "kernel": np.asarray(model.K),
        "rho_g": np.array(model.rho.rho_g),
        "landmarks": model.landmarks.indices,
        "voltages": model.V,
        "voltages_for_distances": model.V if model.V_dist is None else model.V_dist,
        "landmark_distances": model.embedding.landmark_D,
        "embedding": model.transform(X),
    }


@pytest.mark.parametrize("case", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_results_match_the_pinned_reference(case):
    results = _results(case)
    path = REFERENCE / f"regression_{case}.npz"
    if UPDATE:
        REFERENCE.mkdir(exist_ok=True)
        np.savez_compressed(path, **results)
        pytest.skip(f"reference written to {path}")
    expected = np.load(path)
    assert set(expected.files) == set(results), "the set of pinned quantities changed"
    np.testing.assert_array_equal(results["landmarks"], expected["landmarks"], err_msg="landmarks")
    for name in sorted(set(results) - {"landmarks"}):
        got, want = np.asarray(results[name], dtype=np.float64), expected[name].astype(np.float64)
        assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
        scale = max(float(np.abs(want).max()), 1e-12)
        np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE[case] * scale, err_msg=name)
