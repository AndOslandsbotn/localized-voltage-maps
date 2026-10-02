import numpy as np
import pytest

from lvm.config import load_config
from lvm.dimension import choose_dimension
from lvm.landmarks import landmark_count


def _dimension_config(strategy, **options):
    return load_config(overrides={"dimension": {"strategy": strategy, **({strategy: options} if options else {})}}).dimension


@pytest.mark.parametrize("strategy", ["mle", "twonn"])
@pytest.mark.parametrize("d", [2, 5])
def test_estimators_recover_the_dimension_of_a_flat_subspace(strategy, d):
    rng = np.random.default_rng(d)
    Q, _ = np.linalg.qr(rng.normal(size=(30, d)))
    X = rng.random((4000, d)) @ Q.T                     # d-dimensional cube embedded in 30-D
    est = choose_dimension(X, config=_dimension_config(strategy)).d
    assert est == pytest.approx(d, rel=0.25)


def test_fixed_dimension_is_returned_as_is():
    assert choose_dimension(np.zeros((10, 3)), config=_dimension_config("fixed", d=7.5)).d == 7.5


@pytest.mark.parametrize("d, multiplier, expected", [(12.3, 1.0, 14), (12.3, 2.0, 27), (0.0, 1.0, 1)])
def test_dimension_count_is_ceil_multiplier_times_d_plus_one(d, multiplier, expected):
    config = load_config(overrides={"landmarks": {"count": {"strategy": "dimension",
                                                            "dimension": {"multiplier": multiplier}}}}).landmarks
    assert landmark_count(d, config=config) == expected


def test_fixed_count_uses_n_landmarks():
    config = load_config(overrides={"landmarks": {"n_landmarks": 17, "count": {"strategy": "fixed"}}}).landmarks
    assert landmark_count(12.3, config=config) == 17
