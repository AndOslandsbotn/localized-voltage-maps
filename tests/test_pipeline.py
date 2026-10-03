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
        "landmarks": {"n_landmarks": 10, "count": {"strategy": "fixed"}},
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
    assert model.rho.landmarks is model.landmarks       # reach chose them with rho_g; not chosen again
    assert model.embedding.cell_coords.shape == (n, 2)
    assert set(model.timings) == {"sample", "dimension", "cells", "masses", "graph", "scaling", "voltages", "landmarks", "embedding"}


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


def test_gpu_and_cpu_point_paths_agree(strip_model):
    # Same fitted model; only where the per-point steps run differs.
    import dataclasses

    _, X, model = strip_model
    cpu = dataclasses.replace(model, device="cpu")
    gpu = dataclasses.replace(model, device="cuda")
    Xs = X[:1000]
    # float32 on both; an entry within rounding of tau may be zeroed on one side only.
    np.testing.assert_allclose(gpu.voltages(Xs), cpu.voltages(Xs), rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(gpu.transform(Xs), cpu.transform(Xs), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("landmarks_strategy", ["mutual_information", "maxmin"])
@pytest.mark.parametrize("missing", ["clip", "chain"])
def test_landmark_mds_pipeline_runs_and_embeds_the_strip(landmarks_strategy, missing):
    Y, X = _strip_in_20d()
    config = _config(
        landmarks={"strategy": landmarks_strategy, "count": {"strategy": "dimension"}},
        embedding={"strategy": "landmark_mds", "landmark_mds": {"missing": missing}},
    )
    model = fit_level(array_source(X, chunk_size=1000), config)
    # The strip is 2-D but noisy in all 20 dimensions, so the estimate sits a bit above 2.
    assert 1.5 < model.dimension.d < 5.0
    assert model.V.shape[0] == int(np.ceil(model.dimension.d + 1))
    Z = model.transform(X)
    assert np.all(np.isfinite(Z))
    assert np.abs(Z).max() < 1e3     # no blow-up from a degenerate landmark configuration
    idx = np.random.default_rng(1).choice(X.shape[0], 2000, replace=False)
    trust = trustworthiness(Y[idx], Z[idx], n_neighbors=10)
    # Max-min spreads landmarks into general position, which triangulation needs; MI may
    # place the few landmarks almost on a line, leaving only one usable component.
    assert trust > (0.85 if landmarks_strategy == "maxmin" else 0.7)


def test_local_scale_pipeline_spreads_points_and_streams():
    Y, X = _strip_in_20d(n=3000, seed=1)
    config = _config(extension={"kernel": "knn", "knn": {"k": 3, "sharpness": 16.0}},
                     embedding={"local_scale": {"strategy": "cell", "cell": {"fill": 0.5}}})
    model = fit_level(array_source(X, chunk_size=1000), config)
    assert model.local_scale is not None and "local_scale" in model.timings
    Z = np.concatenate(list(model.transform_source(array_source(X, chunk_size=700))))
    assert np.allclose(Z, model.transform(X))
    assert len(np.unique(np.round(Z, 6), axis=0)) > 0.9 * len(Z)       # points no longer piled up


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_local_pca_pipeline_runs_on_its_device_and_streams(device):
    Y, X = _strip_in_20d(n=3000, seed=2)
    config = _config(compute={"device": device},
                     embedding={"local_scale": {"strategy": "pca", "pca": {"fill": 0.35}}})
    model = fit_level(array_source(X, chunk_size=1000), config)
    assert model.local_scale.bases.device.type == device
    Z = np.concatenate(list(model.transform_source(array_source(X, chunk_size=700))))
    assert np.allclose(Z, model.transform(X), atol=1e-5)
    assert len(np.unique(np.round(Z, 5), axis=0)) > 0.9 * len(Z)
