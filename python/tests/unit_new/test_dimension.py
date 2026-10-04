import numpy as np
import pytest

from lvm_new.config import load_config
from lvm_new.dimension import estimate_dimension

MLE = load_config(overrides={"dimension": {"strategy": "mle"}}).dimension


def _flat(d, n=4000, scale=1.0, offset=0.0, seed=0):
    # A d-dimensional cube embedded in 30 dimensions.
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(30, d)))[0]
    return (scale * rng.random((n, d)) @ basis.T + offset).astype(np.float32)


@pytest.mark.parametrize("d", [2, 5])
def test_mle_recovers_the_dimension_of_a_flat_subspace(d, device):
    assert estimate_dimension(_flat(d), config=MLE, device=device, seed=0) == pytest.approx(d, rel=0.25)


@pytest.mark.gpu
def test_mle_on_the_gpu_survives_large_norms_with_close_neighbours():
    # Far from the origin with neighbours close together: float32 distances round to 0 here.
    X = _flat(3, scale=0.01, offset=1000.0)
    assert estimate_dimension(X, config=MLE, device="cuda", seed=0) == pytest.approx(3, rel=0.25)
