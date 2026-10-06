import numpy as np
import pytest
import torch

from lvm_new.cells import cell_masses, fit_cells, refine_cells
from lvm_new.config import load_config


# Far from the origin with cells close together, float32 |c|^2 - 2 x.c rounds badly enough to pick the wrong cell.
FAR = pytest.mark.parametrize("offset, scale", [(0.0, 1.0), (1000.0, 0.001)], ids=["near_origin", "far_from_origin"])


def _blobs(n_blobs, per_blob=200, seed=0, offset=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    centres = offset + scale * rng.random((n_blobs, 5)) * 20
    X = np.vstack([c + scale * 0.1 * rng.standard_normal((per_blob, 5)) for c in centres])
    order = rng.permutation(len(X))
    return centres, X[order].astype(np.float32), np.repeat(np.arange(n_blobs), per_blob)[order]


@FAR
def test_kmeans_gives_every_blob_a_cell_on_each_device(device, offset, scale):
    centres, X, _ = _blobs(8, offset=offset, scale=scale)
    config = load_config(overrides={"cells": {"n_cells": 24}})
    centroids = fit_cells(torch.as_tensor(X, device=device), config=config, device=device, seed=0).cpu().numpy()
    assert centroids.shape == (24, 5)
    nearest = np.linalg.norm(centres[:, None] - centroids[None], axis=2).min(axis=1)
    assert np.all(nearest < 0.5 * scale)


def test_a_sample_no_bigger_than_n_cells_gives_one_cell_per_point():
    X = np.random.default_rng(0).random((10, 3)).astype(np.float32)
    config = load_config(overrides={"cells": {"n_cells": 20}})
    np.testing.assert_allclose(fit_cells(torch.as_tensor(X), config=config, device="cpu", seed=0).numpy(), X)


def test_stream_refinement_ends_at_each_blobs_mean_with_the_prefix_counted_once(device):
    # Separated blobs: assignments never change, so the running means end exactly at each blob's mean.
    _, X, blob = _blobs(4, per_blob=500)
    sample = X[:300]
    start = np.array([sample[blob[:300] == b].mean(axis=0) for b in range(4)])
    chunks = [X[a:a + 128] for a in range(0, len(X), 128)]
    config = load_config(overrides={"cells": {"refine": {"passes": 1}}})       # prefix sample: the first 300 points
    centroids = refine_cells(chunks, torch.as_tensor(sample, device=device), torch.as_tensor(start, device=device),
                             config=config, device=device).cpu().numpy()
    np.testing.assert_allclose(centroids, [X[blob == b].mean(axis=0) for b in range(4)], atol=1e-4)


class _Counting:
    """Re-iterable chunks of the given points, 100 at a time; counts the chunks handed out."""

    def __init__(self, X):
        self.X, self.handed_out = X, 0

    def __iter__(self):
        for start in range(0, len(self.X), 100):
            self.handed_out += 1
            yield self.X[start:start + 100]


@FAR
def test_masses_are_each_cells_share_of_the_data(device, offset, scale):
    rng = np.random.default_rng(0)
    centroids = offset + scale * np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    X = np.vstack([c + scale * 0.1 * rng.standard_normal((n, 2)) for c, n in zip(centroids, (100, 300, 600))])
    config = load_config(overrides={"sample": {"strategy": "reservoir"}})     # data in any order: count it all
    masses = cell_masses(_Counting(X[rng.permutation(len(X))].astype(np.float32)), torch.as_tensor(centroids, device=device),
                         config=config, device=device).cpu().numpy()
    np.testing.assert_allclose(masses, [0.1, 0.3, 0.6])


def test_counting_stops_early_only_for_data_in_random_order():
    centroids = np.array([[0.0], [1.0]])
    X = np.random.default_rng(0).random((10_000, 1)).astype(np.float32)
    masses = {"cells": {"masses": {"rel_error": 0.5}}}                               # 4 points a cell
    random_order = load_config(overrides=masses)                                     # prefix sample
    any_order = load_config(overrides={**masses, "sample": {"strategy": "reservoir"}})
    shuffled, ordered = _Counting(X), _Counting(X)
    cell_masses(shuffled, torch.as_tensor(centroids), config=random_order, device="cpu")
    cell_masses(ordered, torch.as_tensor(centroids), config=any_order, device="cpu")
    assert shuffled.handed_out == 1 and ordered.handed_out == 100
