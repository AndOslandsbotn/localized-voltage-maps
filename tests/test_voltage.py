import importlib.util

import numpy as np
import pytest

from lvm.graph import apply_ground_resistance, build_graph
from lvm.voltage import METHODS, solve_voltage, solve_voltage_maps

HAVE_CHOLMOD = importlib.util.find_spec("sksparse") is not None


def _random_grounded_laplacian(rng, n=40, rho=0.3):
    points = rng.random((n, 2))
    W = build_graph(points, r=0.35).astype(float)
    np.fill_diagonal(W, 0.0)
    return apply_ground_resistance(W, rho=rho)


@pytest.mark.parametrize("method", METHODS)
def test_solve_voltage_satisfies_kirchhoff_condition(method):
    if method == "cholmod" and not HAVE_CHOLMOD:
        pytest.skip("scikit-sparse not installed")
    rng = np.random.default_rng(0)
    L_rho = _random_grounded_laplacian(rng)
    n = L_rho.shape[0]
    source = [0, 1]
    free = np.setdiff1d(np.arange(n), source)

    v = solve_voltage(L_rho, source, method=method)

    assert np.allclose(v[source], 1.0)
    # Definition 3's stationarity condition (Eq. 4 with zero external
    # current): at every floating node the weighted residual must vanish.
    residual = L_rho @ v
    assert np.allclose(residual[free], 0.0, atol=1e-6)


def test_all_methods_agree():
    rng = np.random.default_rng(1)
    L_rho = _random_grounded_laplacian(rng)
    source = [3, 4, 5]

    reference = solve_voltage(L_rho, source, method="dense")
    for method in ("cholesky", "cg"):
        v = solve_voltage(L_rho, source, method=method, tol=1e-10)
        assert np.allclose(v, reference, atol=1e-6), method


def test_all_source_nodes_returns_ones():
    rng = np.random.default_rng(2)
    L_rho = _random_grounded_laplacian(rng, n=5)
    v = solve_voltage(L_rho, list(range(5)), method="dense")
    assert np.allclose(v, 1.0)


def test_unknown_method_raises():
    rng = np.random.default_rng(3)
    L_rho = _random_grounded_laplacian(rng, n=5)
    with pytest.raises(ValueError):
        solve_voltage(L_rho, [0], method="not-a-method")


def test_solve_voltage_maps_stacks_independent_solves():
    rng = np.random.default_rng(4)
    L_rho = _random_grounded_laplacian(rng, n=20)
    source_sets = [[0], [1, 2], [5]]

    V = solve_voltage_maps(L_rho, source_sets, method="dense")

    assert V.shape == (len(source_sets), L_rho.shape[0])
    for row, sources in zip(V, source_sets):
        expected = solve_voltage(L_rho, sources, method="dense")
        assert np.allclose(row, expected)


def test_cholmod_missing_raises_actionable_error(monkeypatch):
    if HAVE_CHOLMOD:
        pytest.skip("scikit-sparse is installed; nothing to test here")
    rng = np.random.default_rng(5)
    L_rho = _random_grounded_laplacian(rng, n=10)
    with pytest.raises(ImportError, match="libsuitesparse"):
        solve_voltage(L_rho, [0], method="cholmod")
