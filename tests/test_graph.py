import numpy as np
import pytest

from helpers import kernel as _kernel
from lvm.config import load_config
from lvm.graph import choose_kernel, choose_radius, grounded_laplacian


def _radius(centroids, strategy, **options):
    config = load_config(overrides={"graph": {"radius": {"strategy": strategy, strategy: options}}}).graph.radius
    return choose_radius(centroids, config=config).r


def _grid(n_side, spacing):
    xs = np.arange(n_side) * spacing
    return np.array([(x, y) for x in xs for y in xs], dtype=float)


# --- radius -----------------------------------------------------------------

def test_centroid_spacing_scales_median_nn_distance():
    # On a square grid every nearest-neighbour distance equals the spacing.
    assert _radius(_grid(5, spacing=0.2), "centroid_spacing", multiplier=3.0) == pytest.approx(0.6)


def test_knn_radius_on_a_line():
    # Spacing 1: an interior point's 2nd neighbour is at 1, an end point's at 2.
    # 9 of the 11 points are interior, so the median is 1.
    centroids = np.arange(11, dtype=float)[:, None]
    assert _radius(centroids, "knn", k=2) == pytest.approx(1.0)


@pytest.mark.parametrize("dim", [2, 10, 50])
def test_knn_radius_keeps_degree_near_k_in_any_dimension(dim):
    rng = np.random.default_rng(dim)
    centroids = rng.random((500, dim))
    K = _kernel(centroids, _radius(centroids, "knn", k=10))
    # r is the median k-th-neighbour distance, so the median degree is about k.
    assert np.median(K.sum(axis=1)) == pytest.approx(10, abs=1)


def test_knn_radius_caps_k():
    centroids = np.array([[0.0], [1.0], [3.0]])
    assert _radius(centroids, "knn", k=100) == _radius(centroids, "knn", k=2)
    with pytest.raises(ValueError):
        _radius(centroids, "knn", k=0)


@pytest.mark.parametrize("strategy, options", [("knn", {"k": 5}), ("centroid_spacing", {"multiplier": 3.0})])
def test_single_cell_radius_is_zero(strategy, options):
    assert _radius(np.zeros((1, 3)), strategy, **options) == 0.0


def test_default_radius_strategy_is_knn():
    rng = np.random.default_rng(1)
    centroids = rng.random((100, 3))
    assert choose_radius(centroids, config=load_config().graph.radius).r == _radius(centroids, "knn", k=10)


# --- kernel -----------------------------------------------------------------

def test_radial_kernel_is_symmetric_indicator_without_self_loops():
    K = _kernel(np.array([[0.0], [1.0], [2.5]]), r=1.5)
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert np.array_equal(K, expected)


def test_radial_kernel_includes_the_boundary():
    assert _kernel(np.array([[0.0], [1.0]]), r=1.0)[0, 1] == 1.0


# --- grounded Laplacian -----------------------------------------------------

def test_grounded_laplacian_mass_weights():
    rng = np.random.default_rng(0)
    K = _kernel(rng.random((20, 2)), r=0.4)
    p = rng.dirichlet(np.ones(20))
    rho_g = 0.7

    L = grounded_laplacian(K, p, rho_g)

    assert np.allclose(L, L.T)
    off = ~np.eye(20, dtype=bool)
    assert np.allclose(L[off], -(np.outer(p, p) * K)[off])
    # The edges cancel in L @ 1, leaving only each node's ground weight.
    assert np.allclose(L @ np.ones(20), rho_g * p)
    assert np.all(np.linalg.eigvalsh(L) > 0)


def test_grounded_laplacian_zero_mass_row_is_empty():
    K = np.ones((3, 3)) - np.eye(3)
    L = grounded_laplacian(K, np.array([0.5, 0.5, 0.0]), 1.0)
    assert np.all(L[2] == 0.0) and np.all(L[:, 2] == 0.0)


@pytest.mark.parametrize(
    "K, p, rho_g",
    [
        (np.zeros((3, 3)), np.ones(2) / 2, 1.0),         # shape mismatch
        (np.zeros((2, 2)), np.ones(2) / 2, 0.0),         # rho_g must be > 0
        (np.zeros((2, 2)), np.array([1.5, -0.5]), 1.0),  # negative mass
    ],
)
def test_grounded_laplacian_rejects_bad_input(K, p, rho_g):
    with pytest.raises(ValueError):
        grounded_laplacian(K, p, rho_g)


def test_kernel_to_points_keeps_every_entry():
    sq = np.array([[0.0, 4.0], [1.0, 0.25]])  # 2 points x 2 cells
    K = choose_kernel(sq, r=1.0, config=load_config().graph.kernel).K
    assert np.array_equal(K, [[1.0, 0.0], [1.0, 1.0]])


def test_exclude_self_needs_a_square_matrix():
    with pytest.raises(ValueError, match="square"):
        choose_kernel(np.zeros((2, 3)), r=1.0, config=load_config().graph.kernel, exclude_self=True)
