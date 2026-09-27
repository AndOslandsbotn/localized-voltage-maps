import numpy as np
import pytest

from lvm.graph import apply_ground_resistance, build_graph
from lvm.voltage import solve_voltage_maps

DEVICES = ("cpu", "cuda")


def _random_grounded_laplacian(rng, n=40, rho=0.3):
    points = rng.random((n, 2))
    W = build_graph(points, r=0.35).astype(float)
    np.fill_diagonal(W, 0.0)
    return apply_ground_resistance(W, rho=rho)


def _reference_voltage(L_rho, sources):
    # Independent reference: plain numpy solve of the reduced system
    # L_rho[F, F] v[F] = -L_rho[F, S] 1, with no torch or batching involved.
    n = L_rho.shape[0]
    free = np.setdiff1d(np.arange(n), sources)
    v = np.ones(n)
    v[free] = np.linalg.solve(L_rho[np.ix_(free, free)], -L_rho[np.ix_(free, sources)].sum(axis=1))
    return v


@pytest.mark.parametrize("device", DEVICES)
def test_solve_voltage_satisfies_kirchhoff_condition(device):
    rng = np.random.default_rng(0)
    L_rho = _random_grounded_laplacian(rng)
    n = L_rho.shape[0]
    source = [0, 1]
    free = np.setdiff1d(np.arange(n), source)

    v = solve_voltage_maps(L_rho, [source], device=device)[0]

    assert np.allclose(v[source], 1.0)
    # Definition 3's stationarity condition (Eq. 4 with zero external
    # current): at every floating node the weighted residual must vanish.
    residual = L_rho @ v
    assert np.allclose(residual[free], 0.0, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_voltage_maps_match_reference(device):
    rng = np.random.default_rng(6)
    L_rho = _random_grounded_laplacian(rng, n=30)
    source_sets = [[0], [1, 2], [5], [7, 8, 9], [29]]

    V = solve_voltage_maps(L_rho, source_sets, device=device)

    assert V.shape == (len(source_sets), L_rho.shape[0])
    for row, sources in zip(V, source_sets):
        assert np.allclose(row, _reference_voltage(L_rho, sources), atol=1e-10)


def test_all_source_nodes_returns_ones():
    rng = np.random.default_rng(2)
    L_rho = _random_grounded_laplacian(rng, n=5)
    v = solve_voltage_maps(L_rho, [list(range(5))], device="cpu")[0]
    assert np.allclose(v, 1.0)


def test_out_of_range_source_raises():
    rng = np.random.default_rng(3)
    L_rho = _random_grounded_laplacian(rng, n=5)
    with pytest.raises(ValueError):
        solve_voltage_maps(L_rho, [[5]], device="cpu")
