import numpy as np
import pytest

from helpers import kernel, region_kernel
from lvm.config import load_config
from lvm.landmarks import choose_landmarks, mutual_information
from lvm.voltage import solve_grounded_voltage_maps, threshold_voltages


def _landmarks_config(precision="float64", **overrides):
    # float64 by default here so results can be compared with exact references.
    mi = {"precision": precision, **overrides.pop("mutual_information", {})}
    return load_config(overrides={"landmarks": {**overrides, "mutual_information": mi}}).landmarks


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def test_mi_is_zero_when_all_cells_look_the_same():
    p = np.full(5, 0.2)
    assert mutual_information(np.ones((3, 5)), p, noise_std=0.01) == pytest.approx(0.0, abs=1e-12)


def test_mi_is_mass_entropy_when_cells_are_separated():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    V = np.array([[0.0, 0.3, 0.6, 0.9]])  # spacing 0.3 >> noise_std
    assert mutual_information(V, p, noise_std=0.01) == pytest.approx(_entropy(p), abs=1e-9)


def test_mi_ignores_zero_mass_cells():
    V = np.array([[0.0, 0.5, 0.5]])
    assert mutual_information(V, np.array([0.5, 0.5, 0.0]), 0.01) == pytest.approx(np.log(2))


def _line_maps(n=50):
    # A chain of cells; weak ground so maps reach along the whole line.
    centroids = np.arange(n, dtype=float)[:, None]
    K = kernel(centroids, 1.0)
    p = np.full(n, 1.0 / n)
    V = solve_grounded_voltage_maps(K, p, 0.05, [[i] for i in range(n)], device="cpu")
    return threshold_voltages(V, 1e-3), p


def test_first_landmark_on_a_line_is_near_an_end():
    # A map from an end is monotone along the line, so it tells every cell
    # apart; a map from the middle is symmetric and confuses mirror cells.
    V, p = _line_maps()
    first = choose_landmarks(V, p, config=_landmarks_config(n_landmarks=1), device="cpu").indices[0]
    assert first < 5 or first >= 45


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_greedy_selection_is_distinct_and_mi_never_decreases(device):
    rng = np.random.default_rng(0)
    centroids = rng.random((200, 2))
    K = region_kernel(centroids)
    p = rng.dirichlet(np.ones(200))
    V = threshold_voltages(solve_grounded_voltage_maps(K, p, 5e-3, [[i] for i in range(200)], device="cpu"), 1e-3)

    sel = choose_landmarks(V, p, config=_landmarks_config(n_landmarks=15), device=device)

    assert len(set(sel.indices.tolist())) == 15
    assert np.all(np.diff(sel.scores) >= -1e-12)
    assert sel.scores[-1] <= _entropy(p) + 1e-9
    # The reported MI is what the chosen maps actually give.
    assert sel.scores[-1] == pytest.approx(mutual_information(V[sel.indices], p, 0.01))


def test_cpu_and_cuda_pick_the_same_landmarks():
    # A symmetric line has exact ties (mirror cells score the same), which the
    # devices may break differently in the last bit; random masses remove them.
    V, _ = _line_maps(40)
    p = np.random.default_rng(1).dirichlet(np.ones(40))
    config = _landmarks_config(n_landmarks=6)
    cpu = choose_landmarks(V, p, config=config, device="cpu")
    cuda = choose_landmarks(V, p, config=config, device="cuda")
    assert np.array_equal(cpu.indices, cuda.indices)


def test_n_landmarks_is_capped_at_the_number_of_candidates():
    V, p = _line_maps(5)
    sel = choose_landmarks(V, p, config=_landmarks_config(n_landmarks=20), device="cpu")
    assert sorted(sel.indices.tolist()) == list(range(5))


def _brute_force_greedy(V, p, k, noise_std):
    chosen = []
    for _ in range(k):
        rest = [c for c in range(V.shape[0]) if c not in chosen]
        chosen.append(max(rest, key=lambda c: mutual_information(V[chosen + [c]], p, noise_std)))
    return chosen


@pytest.mark.parametrize("thresholded", [True, False])
def test_incremental_greedy_matches_brute_force(thresholded):
    rng = np.random.default_rng(2)
    centroids = rng.random((60, 2))
    K = region_kernel(centroids)
    p = rng.dirichlet(np.ones(60))
    V = solve_grounded_voltage_maps(K, p, 5e-3, [[i] for i in range(60)], device="cpu")
    if thresholded:
        V = threshold_voltages(V, 1e-3)
    noise_std = 0.05

    sel = choose_landmarks(V, p, config=_landmarks_config(n_landmarks=5, mutual_information={"noise_std": noise_std}), device="cpu")

    assert sel.indices.tolist() == _brute_force_greedy(V, p, 5, noise_std)
    for k in range(1, 6):
        assert sel.scores[k - 1] == pytest.approx(mutual_information(V[sel.indices[:k]], p, noise_std), rel=1e-9)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_float32_picks_the_same_landmarks_as_float64(device):
    rng = np.random.default_rng(3)
    centroids = rng.random((300, 5))
    p = rng.dirichlet(np.ones(300))
    V = threshold_voltages(
        solve_grounded_voltage_maps(region_kernel(centroids), p, 5e-3, [[i] for i in range(300)], device="cpu"), 1e-3
    )
    sel64 = choose_landmarks(V, p, config=_landmarks_config("float64", n_landmarks=15), device=device)
    sel32 = choose_landmarks(V, p, config=_landmarks_config("float32", n_landmarks=15), device=device)
    assert np.array_equal(sel32.indices, sel64.indices)
    np.testing.assert_allclose(sel32.scores, sel64.scores, rtol=1e-4)


def test_scratch_budget_does_not_change_the_result():
    V, p = _line_maps(40)
    small = choose_landmarks(V, p, config=_landmarks_config(n_landmarks=6, mutual_information={"scratch_mb": 1}), device="cpu")
    large = choose_landmarks(V, p, config=_landmarks_config(n_landmarks=6, mutual_information={"scratch_mb": 512}), device="cpu")
    assert np.array_equal(small.indices, large.indices)


def _maxmin_config():
    return load_config(overrides={"landmarks": {"strategy": "maxmin"}}).landmarks


def test_maxmin_spreads_landmarks_along_a_line():
    # On a chain of cells, farthest-point selection takes both ends first,
    # then fills in the middle.
    V, p = _line_maps(41)
    sel = choose_landmarks(V, p, config=_maxmin_config(), n_landmarks=3, tau=1e-3, device="cpu")
    assert set(sel.indices[:2].tolist()) == {0, 40}
    assert 15 <= sel.indices[2] <= 25
    assert np.all(np.diff(sel.scores[1:]) <= 1e-9)    # each new landmark is closer to the chosen set


def test_maxmin_returns_distinct_landmarks_and_respects_the_count():
    rng = np.random.default_rng(4)
    centroids = rng.random((150, 3))
    p = rng.dirichlet(np.ones(150))
    V = threshold_voltages(solve_grounded_voltage_maps(region_kernel(centroids), p, 5e-3, [[i] for i in range(150)], device="cpu"), 1e-3)
    sel = choose_landmarks(V, p, config=_maxmin_config(), n_landmarks=12, tau=1e-3, device="cpu")
    assert len(sel.indices) == 12 == len(set(sel.indices.tolist()))
