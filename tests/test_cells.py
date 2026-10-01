import numpy as np
import pytest

from scipy.spatial.distance import cdist

from lvm.cells import assign_cells, fit_cells, sq_distances
from lvm.regions import Region, estimate_masses, min_count_for, route
from lvm.stream import array_source, iter_array_chunks


def test_fit_cells_finds_separated_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    sample = np.vstack([c + 0.1 * rng.standard_normal((200, 2)) for c in centers])
    centroids = fit_cells(sample, 3, seed=0)
    matched = centroids[assign_cells(centers, centroids)]  # nearest centroid to each true centre
    np.testing.assert_allclose(matched, centers, atol=0.05)


def test_fit_cells_is_deterministic_for_a_seed():
    sample = np.random.default_rng(1).random((500, 3))
    np.testing.assert_array_equal(fit_cells(sample, 10, seed=3), fit_cells(sample, 10, seed=3))


def test_fit_cells_small_sample_gets_one_cell_per_point():
    sample = np.random.default_rng(2).random((5, 2))
    assert fit_cells(sample, 10).shape == (5, 2)


def test_fit_cells_rejects_empty_sample():
    with pytest.raises(ValueError, match="empty sample"):
        fit_cells(np.empty((0, 2)), 10)


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fit_cells_on_both_devices_finds_the_blobs(device):
    rng = np.random.default_rng(7)
    centers = rng.random((8, 5)) * 20
    sample = np.vstack([c + 0.1 * rng.standard_normal((100, 5)) for c in centers])
    centroids = fit_cells(sample, 8, seed=0, device=device, init_sample_size=300)
    np.testing.assert_allclose(centroids[assign_cells(centers, centroids)], centers, atol=0.05)
