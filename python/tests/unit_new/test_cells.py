import numpy as np

from lvm_new.cells import fit_cells, refine_cells
from lvm_new.config import load_config


def _blobs(n_blobs, per_blob=200, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.random((n_blobs, 5)) * 20
    X = np.vstack([c + 0.1 * rng.standard_normal((per_blob, 5)) for c in centres])
    order = rng.permutation(len(X))
    return centres, X[order].astype(np.float32), np.repeat(np.arange(n_blobs), per_blob)[order]


def test_kmeans_gives_every_blob_a_cell_on_each_device(device):
    centres, X, _ = _blobs(8)
    cells = load_config(overrides={"cells": {"n_cells": 24}}).cells
    centroids = fit_cells(X, config=cells, device=device, seed=0)
    assert centroids.shape == (24, 5)
    nearest = np.linalg.norm(centres[:, None] - centroids[None], axis=2).min(axis=1)
    assert np.all(nearest < 0.5)


def test_a_sample_no_bigger_than_n_cells_gives_one_cell_per_point():
    X = np.random.default_rng(0).random((10, 3)).astype(np.float32)
    cells = load_config(overrides={"cells": {"n_cells": 20}}).cells
    np.testing.assert_allclose(fit_cells(X, config=cells, device="cpu", seed=0), X)


def test_stream_refinement_ends_at_each_blobs_mean_with_the_prefix_counted_once(device):
    # Separated blobs: assignments never change, so the running means end exactly at each blob's mean.
    _, X, blob = _blobs(4, per_blob=500)
    sample = X[:300]
    start = np.array([sample[blob[:300] == b].mean(axis=0) for b in range(4)])
    chunks = [X[a:a + 128] for a in range(0, len(X), 128)]
    refine = load_config(overrides={"cells": {"refine": {"strategy": "stream"}}}).cells.refine
    centroids = refine_cells(chunks, sample, start, config=refine, device=device, skip=300)
    np.testing.assert_allclose(centroids, [X[blob == b].mean(axis=0) for b in range(4)], atol=1e-4)
