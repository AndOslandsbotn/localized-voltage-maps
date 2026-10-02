import numpy as np
import pytest

from helpers import region_kernel
from lvm.config import load_config
from lvm.scaling import choose_rho_g, typical_support_fraction

TAU = 1e-3


def _region(seed=0, n=300):
    rng = np.random.default_rng(seed)
    centroids = rng.random((n, 2))
    K = region_kernel(centroids, "centroid_spacing")
    return K, rng.dirichlet(np.ones(n))


def _scaling(strategy="support_fraction", **options):
    return load_config(overrides={"scaling": {"strategy": strategy, strategy: options}}).scaling


def test_typical_fraction_decreases_with_rho_g():
    K, p = _region()
    fractions = [typical_support_fraction(K, p, r, tau=TAU, statistic="median", device="cpu") for r in (1e-4, 1e-2, 1.0)]
    assert fractions[0] >= fractions[1] >= fractions[2]
    assert fractions[0] > fractions[2]


@pytest.mark.parametrize("statistic", ["median", "mean"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_choose_rho_g_hits_target(statistic, device):
    K, p = _region()
    config = _scaling(target=0.1, statistic=statistic, tolerance=0.01)

    choice = choose_rho_g(K, p, tau=TAU, config=config, n_landmarks=20, device=device)

    assert abs(choice.support_fraction - 0.1) <= 0.01
    # The reported fraction is the one the returned rho_g actually gives.
    assert choice.support_fraction == pytest.approx(
        typical_support_fraction(K, p, choice.rho_g, tau=TAU, statistic=statistic, device="cpu")
    )


def test_unreachable_target_returns_nearest_bound():
    K, p = _region()
    # Maps can't cover more than every cell, so a target of 1.0 is only reached
    # in the weak-ground limit; the lower bound is the closest we can get.
    config = _scaling(target=0.999, rho_g_bounds=[1e-2, 1e2])
    assert choose_rho_g(K, p, tau=TAU, config=config, n_landmarks=20, device="cpu").rho_g == 1e-2
    # A tiny target below the smallest possible support (the source alone).
    config = _scaling(target=1e-6, rho_g_bounds=[1e-2, 1e2])
    assert choose_rho_g(K, p, tau=TAU, config=config, n_landmarks=20, device="cpu").rho_g == 1e2


def test_zero_mass_cells_are_not_used_as_sources():
    K, p = _region(n=100)
    p[:10] = 0.0
    p /= p.sum()
    # Would raise if a zero-mass cell were passed as a source.
    choice = choose_rho_g(K, p, tau=TAU, config=_scaling(target=0.1, tolerance=0.02), n_landmarks=20, device="cpu")
    assert choice.rho_g > 0


def test_bad_bounds_raise():
    K, p = _region(n=50)
    with pytest.raises(ValueError, match="rho_g_bounds"):
        choose_rho_g(K, p, tau=TAU, config=_scaling(rho_g_bounds=[1.0, 0.1]), n_landmarks=20, device="cpu")


@pytest.mark.parametrize("n_landmarks", [5, 10, 20])
def test_coverage_targets_overlap_over_n_landmarks(n_landmarks):
    K, p = _region()
    config = _scaling("coverage", overlap=2.0, tolerance=0.01)
    choice = choose_rho_g(K, p, tau=TAU, config=config, n_landmarks=n_landmarks, device="cpu")
    assert abs(choice.support_fraction - 2.0 / n_landmarks) <= 0.01


def test_reach_with_all_landmarks_is_the_default_strategy():
    scaling = load_config().scaling
    assert scaling.strategy == "reach" and scaling.reach.k is None


def test_coverage_target_is_capped_at_every_cell():
    # overlap / n_landmarks > 1 asks for more than the whole region: weakest ground.
    K, p = _region()
    config = _scaling("coverage", overlap=3.0, rho_g_bounds=[1e-2, 1e2])
    assert choose_rho_g(K, p, tau=TAU, config=config, n_landmarks=2, device="cpu").rho_g == 1e-2


def _reach_setup(seed=0):
    from scipy.sparse.csgraph import connected_components

    rng = np.random.default_rng(seed)
    grid = np.array([(x, y) for x in range(14) for y in range(14)], dtype=float) / 14
    K = region_kernel(grid + 0.01 * rng.standard_normal(grid.shape))   # jittered grid: evenly dense
    assert connected_components(K)[0] == 1                   # reach is only meaningful on a connected graph
    p = rng.dirichlet(np.ones(196))
    landmarks = load_config(overrides={"landmarks": {"n_landmarks": 8, "count": {"strategy": "fixed"}}}).landmarks
    return K, p, landmarks


def _chosen_sets(p, choice):
    cells = np.flatnonzero(p > 0)
    return [[int(cells[i])] for i in choice.landmarks.indices]


@pytest.mark.parametrize("k", [3, 8])
def test_reach_meets_its_target_with_the_most_local_maps(k):
    from lvm.scaling import landmark_reach

    K, p, landmarks = _reach_setup()
    config = _scaling("reach", k=k, quantile=0.5, rel_tolerance=0.02)
    choice = choose_rho_g(K, p, tau=TAU, config=config, n_landmarks=8, landmarks=landmarks, device="cpu")
    chosen = _chosen_sets(p, choice)
    # The returned landmarks reach k at the returned rho_g ...
    assert choice.reach >= k
    assert landmark_reach(K, p, choice.rho_g, chosen, tau=TAU, quantile=0.5, device="cpu") >= k
    # ... and these are the most local maps that do: a noticeably stronger ground no longer reaches k.
    assert landmark_reach(K, p, choice.rho_g * 1.5, chosen, tau=TAU, quantile=0.5, device="cpu") < k


def test_reach_alternation_settles_within_its_rounds():
    K, p, landmarks = _reach_setup()
    choice = choose_rho_g(K, p, tau=TAU, config=_scaling("reach", max_rounds=5), n_landmarks=8, landmarks=landmarks,
                          device="cpu")
    assert 1 <= choice.n_rounds <= 5
    assert len(set(choice.landmarks.indices.tolist())) == 8


@pytest.mark.parametrize("start_factor", [0.1, 10.0])
def test_reach_from_a_poor_start_still_meets_its_target(start_factor):
    K, p, landmarks = _reach_setup()
    choice = choose_rho_g(K, p, tau=TAU, config=_scaling("reach", start_factor=start_factor), n_landmarks=8,
                          landmarks=landmarks, device="cpu")
    assert choice.reach >= 8


@pytest.mark.parametrize("guess", [None, 1e-3, 1.0, 1e3])
def test_largest_passing_finds_the_boundary_from_any_side(guess):
    from lvm.scaling import _largest_passing

    # value falls through the target 0 at rho_g = 2
    rho, value, n = _largest_passing(lambda r: -np.log(r / 2.0), 0.0, bounds=(1e-6, 1e6), rel_tolerance=0.01,
                                     max_iter=60, guess=guess)
    assert 2.0 / 1.01 <= rho <= 2.0 and value >= 0.0


@pytest.mark.parametrize("guess", [None, 1.0])
def test_largest_passing_returns_a_bound_when_the_target_is_met_everywhere_or_nowhere(guess):
    from lvm.scaling import _largest_passing

    kw = dict(bounds=(1e-2, 1e2), rel_tolerance=0.01, max_iter=60, guess=guess)
    assert _largest_passing(lambda r: 1.0, 0.0, **kw)[0] == 1e2
    assert _largest_passing(lambda r: -1.0, 0.0, **kw)[0] == 1e-2


def test_reach_with_fewer_landmarks_required_gives_more_local_maps():
    K, p, landmarks = _reach_setup(seed=1)
    rho = {k: choose_rho_g(K, p, tau=TAU, config=_scaling("reach", k=k), n_landmarks=8, landmarks=landmarks,
                           device="cpu").rho_g for k in (2, 8)}
    assert rho[2] > rho[8]


def test_reach_defaults_to_all_landmarks_and_needs_the_landmarks_config():
    K, p, landmarks = _reach_setup()
    choice = choose_rho_g(K, p, tau=TAU, config=_scaling("reach"), n_landmarks=8, landmarks=landmarks, device="cpu")
    assert choice.reach >= 8
    with pytest.raises(ValueError, match="landmarks config"):
        choose_rho_g(K, p, tau=TAU, config=_scaling("reach"), n_landmarks=8, device="cpu")
