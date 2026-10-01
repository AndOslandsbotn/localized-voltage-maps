import numpy as np
import pytest
from sklearn.manifold import trustworthiness

from lvm.config import load_config
from lvm.pipeline import fit_level
from lvm.stream import array_source


def _strip_in_20d(n=6000, seed=0):
    # A flat 4 x 1 strip, rotated into 20 dimensions, points in shuffled order.
    rng = np.random.default_rng(seed)
    Y = rng.random((n, 2)) * [4.0, 1.0]
    Q, _ = np.linalg.qr(rng.normal(size=(20, 2)))
    return Y, Y @ Q.T + 0.01 * rng.normal(size=(n, 20))


def _config(**overrides):
    base = {
        "compute": {"device": "cpu"},
        "cells": {"n_cells": 150, "sample_size": 3000, "masses": {"rel_error": 0.2}},
        "landmarks": {"n_landmarks": 10},
    }
    for key, value in overrides.items():
        base.setdefault(key, {}).update(value)
    return load_config(overrides=base)


@pytest.fixture(scope="module")
def strip_model():
    Y, X = _strip_in_20d()
    return Y, X, fit_level(array_source(X, chunk_size=1000), _config())


def test_fit_level_shapes(strip_model):
    _, X, model = strip_model
    n = model.centroids.shape[0]
    assert model.centroids.shape == (150, 20)
    assert model.K.shape == (n, n)
    assert model.V.shape == (10, n)
    assert len(set(model.landmark_cells.tolist())) == 10
    assert model.embedding.cell_coords.shape == (n, 2)
    assert set(model.timings) == {"sample", "cells", "masses", "graph", "scaling", "voltages", "landmarks", "embedding"}


def test_embedding_preserves_neighbourhoods_of_a_flat_strip(strip_model):
    Y, X, model = strip_model
    Z = model.transform(X)
    assert Z.shape == (X.shape[0], 2) and np.all(np.isfinite(Z))
    # Neighbours in the embedding should mostly be neighbours on the strip.
    idx = np.random.default_rng(1).choice(X.shape[0], 2000, replace=False)
    assert trustworthiness(Y[idx], Z[idx], n_neighbors=10) > 0.9


def test_streamed_transform_matches_in_memory(strip_model):
    _, X, model = strip_model
    streamed = np.concatenate(list(model.transform_source(array_source(X[:2500], chunk_size=700))))
    assert np.allclose(streamed, model.transform(X[:2500]))
