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


def test_coverage_is_the_default_strategy():
    assert load_config().scaling.strategy == "coverage"


def test_coverage_target_is_capped_at_every_cell():
    # overlap / n_landmarks > 1 asks for more than the whole region: weakest ground.
    K, p = _region()
    config = _scaling("coverage", overlap=3.0, rho_g_bounds=[1e-2, 1e2])
    assert choose_rho_g(K, p, tau=TAU, config=config, n_landmarks=2, device="cpu").rho_g == 1e-2
