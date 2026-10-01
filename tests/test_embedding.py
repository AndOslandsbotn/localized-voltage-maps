import numpy as np
import pytest

from lvm.config import load_config
from lvm.embedding import fit_embedding

TAU = 1e-3


def _fit(V, p, n_components=2):
    config = load_config(overrides={"embedding": {"log_mds": {"n_components": n_components}}}).embedding
    return fit_embedding(V, p, config=config, tau=TAU)


def test_log_mds_is_weighted_pca_of_log_features():
    rng = np.random.default_rng(0)
    V = rng.random((6, 50))
    p = rng.dirichlet(np.ones(50))
    emb = _fit(V, p, n_components=3)

    assert np.allclose(emb.components @ emb.components.T, np.eye(3))
    assert np.allclose(emb.transform(V), emb.cell_coords)
    # Weighted mean of the coordinates is zero, and the axes are ordered by variance.
    assert np.allclose(p @ emb.cell_coords, 0.0)
    var = p @ emb.cell_coords**2
    assert np.all(np.diff(var) <= 1e-12)


def test_zeroed_voltages_become_finite_far_features():
    V = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    emb = _fit(V, np.full(3, 1 / 3), n_components=1)
    assert np.all(np.isfinite(emb.cell_coords))
    # The two end cells are mirror images, the middle one sits between them.
    c = emb.cell_coords[:, 0]
    assert c[0] == pytest.approx(-c[2]) and c[1] == pytest.approx(0.0, abs=1e-12)


def test_zero_mass_cells_get_coordinates_without_moving_the_axes():
    rng = np.random.default_rng(1)
    V = rng.random((4, 30))
    p = rng.dirichlet(np.ones(30))
    p_zero = p.copy()
    p_zero[5] = 0.0
    V_moved = V.copy()
    V_moved[:, 5] = rng.random(4)  # changing a zero-mass cell's voltages...
    a, b = _fit(V, p_zero), _fit(V_moved, p_zero)
    assert np.allclose(a.components, b.components)  # ...doesn't change the fit
    assert np.all(np.isfinite(b.cell_coords[5]))
