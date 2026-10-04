import numpy as np

from lvm_new.config import load_config
from lvm_new.sample import sample_region


class _Counting:
    """Chunks of 100 points whose single feature is the point's index; counts the chunks handed out."""

    def __init__(self, n_points):
        self.n_points, self.handed_out = n_points, 0

    def __iter__(self):
        for start in range(0, self.n_points, 100):
            self.handed_out += 1
            yield np.arange(start, min(start + 100, self.n_points), dtype=np.float32)[:, None]


def _config(strategy, size):
    return load_config(overrides={"sample": {"strategy": strategy, "size": size}}).sample


def test_prefix_takes_the_first_points_and_stops_reading():
    source = _Counting(1000)
    sample = sample_region(source, config=_config("prefix", 250), seed=0)
    np.testing.assert_array_equal(sample[:, 0], np.arange(250))
    assert source.handed_out == 3


def test_reservoir_draws_from_the_whole_data_and_repeats_with_the_seed():
    first = sample_region(_Counting(10_000), config=_config("reservoir", 1000), seed=0)[:, 0]
    again = sample_region(_Counting(10_000), config=_config("reservoir", 1000), seed=0)[:, 0]
    assert len(np.unique(first)) == 1000 and (first >= 5000).any()
    np.testing.assert_array_equal(first, again)
