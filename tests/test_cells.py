import numpy as np
import pytest

from scipy.spatial.distance import cdist

from lvm.cells import assign_cells, fit_cells, sq_distances
from lvm.config import load_config
from lvm.regions import Region, estimate_masses, min_count_for, route
from lvm.stream import array_source, iter_array_chunks


STRATEGIES = ["cuml", "faiss", "sklearn"]


def _fit(sample, n_cells, strategy="sklearn", seed=0):
    config = load_config(overrides={"cells": {"kmeans": {"strategy": strategy}}}).cells.kmeans
    return fit_cells(sample, config=config, n_cells=n_cells, seed=seed).centroids


def _blobs(n_blobs=8, per_blob=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.random((n_blobs, 5)) * 20
    return centers, np.vstack([c + 0.1 * rng.standard_normal((per_blob, 5)) for c in centers])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_cluster_gets_cells_when_cells_outnumber_clusters(strategy):
    # The regime LVM uses: many more cells than clusters. Random init
    # (cuml, faiss) may split one blob more finely than another, but it
    # must not leave a blob without a cell.
    centers, sample = _blobs()
    centroids = _fit(sample, 24, strategy)
    assert centroids.shape == (24, 5) and centroids.dtype == np.float64
    nearest = centroids[assign_cells(centers, centroids)]
    assert np.all(np.linalg.norm(nearest - centers, axis=1) < 0.5)


def test_sklearn_recovers_exactly_k_separated_blobs():
    # With k = number of blobs only k-means++ (sklearn) guarantees one centroid
    # per blob; random init (cuml, faiss) can put two in one blob.
    centers, sample = _blobs()
    centroids = _fit(sample, 8, "sklearn")
    np.testing.assert_allclose(centroids[assign_cells(centers, centroids)], centers, atol=0.05)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_fit_cells_is_deterministic_for_a_seed(strategy):
    sample = np.random.default_rng(1).random((2000, 3))
    np.testing.assert_allclose(_fit(sample, 10, strategy, seed=3), _fit(sample, 10, strategy, seed=3), rtol=1e-5)


def test_fit_cells_small_sample_gets_one_cell_per_point():
    sample = np.random.default_rng(2).random((5, 2))
    np.testing.assert_array_equal(_fit(sample, 10), sample)


def test_fit_cells_rejects_empty_sample():
    with pytest.raises(ValueError, match="empty sample"):
        _fit(np.empty((0, 2)), 10)


@pytest.mark.parametrize("device", [None, "cuda"])
def test_assign_cells_matches_brute_force_on_both_devices(device):
    rng = np.random.default_rng(4)
    X, C = rng.random((500, 6)), rng.random((30, 6))
    expected = cdist(X, C).argmin(axis=1)
    np.testing.assert_array_equal(assign_cells(X, C, device=device), expected)


def test_min_count_for():
    assert min_count_for(0.1) == 100
    assert min_count_for(0.05) == 400


def _uniform_line_region(n_cells=4):
    # Cells centred at 0.5, 1.5, ... on [0, n_cells]: uniform data gives equal masses.
    return Region(centroids=(np.arange(n_cells) + 0.5)[:, None])


def test_shuffled_masses_stop_early_and_are_accurate():
    X = np.random.default_rng(3).uniform(0, 4, size=(200_000, 1))
    chunks_read = 0

    def source():
        nonlocal chunks_read
        for chunk in iter_array_chunks(X, 1000):
            chunks_read += 1
            yield chunk

    masses = estimate_masses(source, _uniform_line_region(), min_count=400, shuffled=True)[()]
    assert masses.converged
    assert masses.counts.min() >= 400
    assert chunks_read < 5  # ~1600 points needed, far less than the 200 chunks available
    np.testing.assert_allclose(masses.p, 0.25, rtol=0.15)
    assert masses.p.sum() == pytest.approx(1.0)


def test_unshuffled_masses_read_everything():
    X = np.random.default_rng(4).uniform(0, 4, size=(10_000, 1))
    masses = estimate_masses(array_source(X, 999), _uniform_line_region(), min_count=1, shuffled=False)[()]
    assert masses.n_seen == 10_000
    np.testing.assert_array_equal(masses.counts, np.bincount(assign_cells(X, _uniform_line_region().centroids)))


def test_masses_per_overlapping_region_match_brute_force():
    root = Region(centroids=np.array([[0.0], [1.0], [2.0], [3.0]]))
    root.add_child([0, 1, 2]).centroids = np.array([[0.0], [2.0]])
    root.add_child([2, 3]).centroids = np.array([[2.0], [3.0], [4.0]])
    X = np.random.default_rng(5).uniform(-0.4, 3.4, size=(5000, 1))
    masses = estimate_masses(array_source(X, 700), root, min_count=1, shuffled=False)
    for leaf, rows in route(X, root):
        expected = np.bincount(assign_cells(X[rows], leaf.centroids), minlength=leaf.centroids.shape[0])
        np.testing.assert_array_equal(masses[leaf.id].counts, expected)


def test_max_points_caps_the_scan():
    X = np.random.default_rng(6).uniform(0, 4, size=(10_000, 1))
    masses = estimate_masses(array_source(X, 100), _uniform_line_region(), min_count=10_000, shuffled=True, max_points=500)[()]
    assert masses.n_seen == 500
    assert not masses.converged


def test_masses_require_cells():
    with pytest.raises(ValueError, match="no cells yet"):
        estimate_masses(array_source(np.zeros((3, 1)), 2), Region(), min_count=1, shuffled=True)


@pytest.mark.parametrize("dtype, rtol", [(np.float64, 1e-10), (np.float32, 1e-4)])
def test_sq_distances_match_cdist(dtype, rtol):
    rng = np.random.default_rng(5)
    X, C = rng.random((300, 20)), rng.random((40, 20))
    np.testing.assert_allclose(sq_distances(X, C, dtype=dtype), cdist(X, C, "sqeuclidean"), rtol=rtol, atol=rtol)


def test_sq_distances_are_never_negative():
    X = np.full((5, 3), 1e4) + np.random.default_rng(6).random((5, 3)) * 1e-6  # near-identical rows
    assert np.all(sq_distances(X, X) >= 0.0)


def test_sq_distances_on_gpu_matches_numpy():
    import torch

    rng = np.random.default_rng(8)
    X, C = rng.random((200, 10)), rng.random((25, 10))
    gpu = sq_distances(torch.as_tensor(X, device="cuda"), torch.as_tensor(C, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(gpu, sq_distances(X, C), rtol=1e-9, atol=1e-9)
