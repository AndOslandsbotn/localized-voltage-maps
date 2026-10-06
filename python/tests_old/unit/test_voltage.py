import numpy as np
import pytest

from helpers import kernel
from lvm_old.graph import grounded_laplacian
from lvm_old.config import load_config
from lvm_old.voltage import (
    choose_sources,
    extend_voltages,
    solve_grounded_voltage_maps,
    solve_voltage_maps,
    support_mask,
    threshold_voltages,
)


def _random_grounded_laplacian(rng, n=40, rho_g=0.3):
    points = rng.random((n, 2))
    p = rng.dirichlet(np.ones(n))
    return grounded_laplacian(kernel(points, r=0.35), p, rho_g)


def _random_region(rng, n=40):
    points = rng.random((n, 2))
    return kernel(points, r=0.3), rng.dirichlet(np.ones(n))


def _reference_voltage(L_rho, sources):
    # Independent reference: plain numpy solve of the reduced system
    # L_rho[F, F] v[F] = -L_rho[F, S] 1, with no torch or batching involved.
    n = L_rho.shape[0]
    free = np.setdiff1d(np.arange(n), sources)
    v = np.ones(n)
    v[free] = np.linalg.solve(L_rho[np.ix_(free, free)], -L_rho[np.ix_(free, sources)].sum(axis=1))
    return v


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


def _neighbour_average(K, p, rho_g, v):
    # v_i = sum_j K_ij p_j v_j / (rho_g + sum_j K_ij p_j): the stationarity
    # condition at a free cell once its own mass p_i is divided out.
    Kp = K * p[None, :]
    return (Kp @ v) / (rho_g + Kp.sum(axis=1))


def test_grounded_maps_match_reference(device):
    rng = np.random.default_rng(10)
    K, p = _random_region(rng)
    rho_g = 0.5
    source_sets = [[0], [3, 4], [39]]

    V = solve_grounded_voltage_maps(K, p, rho_g, source_sets, device=device)

    L_rho = grounded_laplacian(K, p, rho_g)
    for row, sources in zip(V, source_sets):
        assert np.allclose(row, _reference_voltage(L_rho, sources), atol=1e-10)


def test_free_cells_are_mass_weighted_neighbour_averages():
    # Include very light cells: their own mass cancels, so they must still
    # satisfy the averaging condition to high precision.
    rng = np.random.default_rng(11)
    K, p = _random_region(rng)
    p[[5, 6, 7]] = 1e-9
    rho_g = 0.5
    source = [0]
    free = np.setdiff1d(np.arange(p.size), source)

    v = solve_grounded_voltage_maps(K, p, rho_g, [source], device="cpu")[0]

    assert np.allclose(v[free], _neighbour_average(K, p, rho_g, v)[free], atol=1e-10)


def test_zero_mass_cells_get_neighbour_average():
    rng = np.random.default_rng(12)
    K, p = _random_region(rng)
    empty = [5, 6, 20]
    p[empty] = 0.0
    rho_g = 0.5

    v = solve_grounded_voltage_maps(K, p, rho_g, [[0]], device="cpu")[0]

    assert np.allclose(v[empty], _neighbour_average(K, p, rho_g, v)[empty], atol=1e-12)
    assert np.all((v >= 0.0) & (v <= 1.0))


def test_zero_mass_is_the_limit_of_vanishing_mass():
    # Filling in a p = 0 cell must agree with solving with a tiny p: the
    # voltage map is continuous as a cell's mass goes to zero.
    rng = np.random.default_rng(13)
    K, p = _random_region(rng)
    rho_g = 0.5
    p_zero, p_tiny = p.copy(), p.copy()
    p_zero[[5, 6]] = 0.0
    p_tiny[[5, 6]] = 1e-12

    v_zero = solve_grounded_voltage_maps(K, p_zero, rho_g, [[0]], device="cpu")[0]
    v_tiny = solve_grounded_voltage_maps(K, p_tiny, rho_g, [[0]], device="cpu")[0]

    assert np.allclose(v_zero, v_tiny, atol=1e-8)


def test_isolated_zero_mass_cell_has_zero_voltage():
    K = np.zeros((3, 3))
    K[0, 1] = K[1, 0] = 1.0  # cell 2 has no neighbours
    p = np.array([0.5, 0.5, 0.0])
    v = solve_grounded_voltage_maps(K, p, 1.0, [[0]], device="cpu")[0]
    assert v[2] == 0.0


def test_zero_mass_source_raises():
    rng = np.random.default_rng(14)
    K, p = _random_region(rng, n=10)
    p[3] = 0.0
    with pytest.raises(ValueError, match="zero-mass"):
        solve_grounded_voltage_maps(K, p, 0.5, [[3]], device="cpu")


def test_threshold_zeroes_small_voltages_and_keeps_the_rest():
    V = np.array([[1.0, 0.2, 1e-3, 5e-4, 0.0]])
    out = threshold_voltages(V, tau=1e-3)
    assert np.array_equal(out, [[1.0, 0.2, 1e-3, 0.0, 0.0]])  # tau itself is kept
    assert V[0, 3] == 5e-4  # input untouched


def test_support_mask_matches_threshold():
    rng = np.random.default_rng(15)
    K, p = _random_region(rng)
    V = solve_grounded_voltage_maps(K, p, 2.0, [[0], [10]], device="cpu")
    mask = support_mask(V, tau=1e-2)
    assert np.array_equal(mask, threshold_voltages(V, 1e-2) > 0)
    assert mask[0, 0] and mask[1, 10]  # sources are always in their own support


def test_larger_ground_weight_shrinks_support():
    # Corollary 14: a stronger ground drains more current, so maps are more local.
    rng = np.random.default_rng(16)
    K, p = _random_region(rng, n=60)
    weak = support_mask(solve_grounded_voltage_maps(K, p, 0.1, [[0]], device="cpu"), 1e-3).sum()
    strong = support_mask(solve_grounded_voltage_maps(K, p, 10.0, [[0]], device="cpu"), 1e-3).sum()
    assert strong < weak


@pytest.mark.parametrize("tau", [-0.1, 1.0])
def test_threshold_rejects_bad_tau(tau):
    with pytest.raises(ValueError):
        threshold_voltages(np.ones((1, 3)), tau)


def test_single_node_sources_are_the_cells_with_mass():
    p = np.array([0.5, 0.0, 0.3, 0.2])
    sources = choose_sources(p, config=load_config().sources)
    assert sources.sets == [[0], [2], [3]]
    assert np.array_equal(sources.cells, [0, 2, 3])


def test_grounded_extension_matches_zero_mass_fill_in():
    # A data point is a zero-mass cell: placing one exactly where a zero-mass
    # cell sits must give that cell's fill-in voltage.
    rng = np.random.default_rng(20)
    points = rng.random((40, 2))
    p = rng.dirichlet(np.ones(40))
    p[7] = 0.0
    K = kernel(points, r=0.3)
    rho_g = 0.5
    V = solve_grounded_voltage_maps(K, p, rho_g, [[0], [12]], device="cpu")

    Kx = kernel(points, r=0.3)[[7]]  # row 7: cell 7 to every cell, zero on itself
    v = extend_voltages(Kx, V, p, config=load_config(overrides={"extension": {"strategy": "grounded"}}).extension, rho_g=rho_g, nearest=np.array([7]))

    assert np.allclose(v[:, 0], V[:, 7], atol=1e-12)


def test_isolated_point_takes_its_nearest_cells_voltages():
    V = np.array([[1.0, 0.4, 0.1]])
    p = np.full(3, 1 / 3)
    Kx = np.zeros((1, 3))  # no cell within reach
    v = extend_voltages(Kx, V, p, config=load_config(overrides={"extension": {"strategy": "grounded"}}).extension, rho_g=1.0, nearest=np.array([1]))
    assert v[0, 0] == 0.4


def test_voltage_distances_are_minus_log_and_unknown_below_tau():
    from lvm_old.voltage import voltage_distances

    d = voltage_distances(np.array([[1.0, np.exp(-2.0), 1e-4]]), tau=1e-3)
    assert d[0, 0] == 0.0 and d[0, 1] == pytest.approx(2.0) and np.isinf(d[0, 2])


def test_chained_distances_fill_gaps_with_shortest_paths():
    from lvm_old.voltage import chained_distances

    inf = np.inf
    D = np.array([[0.0, 1.0, inf], [1.0, 0.0, 2.0], [inf, 3.0, 0.0]])   # 0-2 unknown; 1-2 asymmetric
    C = chained_distances(D)
    assert np.allclose(C, C.T)
    assert C[1, 2] == pytest.approx(2.5)        # mean of the two directions
    assert C[0, 2] == pytest.approx(3.5)        # through 1


def test_chained_distances_keep_disconnected_pairs_infinite():
    from lvm_old.voltage import chained_distances

    D = np.array([[0.0, np.inf], [np.inf, 0.0]])
    assert np.isinf(chained_distances(D)[0, 1])


def test_average_extension_interpolates_cell_voltages_without_ground(device):
    import torch

    from lvm_old.config import load_config
    from lvm_old.voltage import extend_voltages

    V = np.array([[1.0, 0.5, 0.2]])                     # one landmark map over 3 cells
    p = np.array([0.2, 0.3, 0.5])
    Kx = np.array([[1.0, 0.0, 0.0],                     # a point sitting on cell 0 only
                   [0.0, 1.0, 1.0],                     # a point between cells 1 and 2
                   [0.0, 0.0, 0.0]])                    # an isolated point (nearest cell 2)
    nearest = np.array([0, 1, 2])
    config = load_config(overrides={"extension": {"strategy": "average"}}).extension
    t = (lambda a: torch.as_tensor(a, device=device)) if device == "cuda" else (lambda a: a)
    out = extend_voltages(t(Kx), t(V), t(p), config=config, rho_g=0.5, nearest=t(nearest))
    out = out.cpu().numpy() if device == "cuda" else out
    assert np.allclose(out[0], [1.0, (0.3 * 0.5 + 0.5 * 0.2) / 0.8, 0.2])   # no ground: no halving
    grounded = load_config(overrides={"extension": {"strategy": "grounded"}}).extension
    h = extend_voltages(Kx, V, p, config=grounded, rho_g=0.5, nearest=nearest)
    assert h[0, 0] < out[0, 0]                                               # the ground lowers it
