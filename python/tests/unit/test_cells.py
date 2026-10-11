import numpy as np
import pytest
import torch

from lvm.cells import _allocate, assign_cells, cell_masses, fit_cells, refine_cells, split_cells
from lvm.config import load_config


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


def test_farthest_cells_spread_evenly_over_the_data_whatever_its_density(device):
    rng = np.random.default_rng(0)
    dense, sparse = rng.random((9000, 2)), rng.random((1000, 2)) + [2.0, 0.0]       # two unit squares
    X = torch.as_tensor(np.vstack([dense, sparse]).astype(np.float32), device=device)
    config = load_config(overrides={"cells": {"n_cells": 50, "placement": "farthest"}})
    centroids = fit_cells(X, config=config, device=device, seed=0).cpu().numpy()
    assert 0.4 <= np.mean(centroids[:, 0] > 1.5) <= 0.6


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
    centroids, _ = refine_cells(chunks, torch.as_tensor(sample, device=device), torch.as_tensor(start, device=device),
                                config=config, device=device)
    np.testing.assert_allclose(centroids.cpu().numpy(), [X[blob == b].mean(axis=0) for b in range(4)], atol=1e-4)


def test_streamed_lloyd_is_k_means_on_all_the_points(device):
    from sklearn.cluster import KMeans

    X = np.random.default_rng(0).random((2000, 2)).astype(np.float32)
    start = X[:8].astype(np.float64)
    chunks = [X[a:a + 128] for a in range(0, len(X), 128)]
    config = load_config(overrides={"cells": {"refine": {"strategy": "lloyd", "passes": 5, "tolerance": 1e-12}}})
    centroids, passes = refine_cells(chunks, torch.as_tensor(X[:100], device=device),
                                     torch.as_tensor(start, device=device), config=config, device=device)
    expected = KMeans(8, init=start, n_init=1, max_iter=5, tol=0.0, algorithm="lloyd").fit(X.astype(np.float64))
    np.testing.assert_allclose(centroids.cpu().numpy(), expected.cluster_centers_, atol=1e-5)
    assert passes == 5


def test_streamed_lloyd_stops_once_the_centroids_settle(device):
    _, X, blob = _blobs(4, per_blob=500)
    start = np.array([X[blob == b][:5].mean(axis=0) for b in range(4)])
    chunks = [X[a:a + 128] for a in range(0, len(X), 128)]
    config = load_config(overrides={"cells": {"refine": {"strategy": "lloyd", "passes": 30}}})
    centroids, passes = refine_cells(chunks, torch.as_tensor(X[:100], device=device),
                                     torch.as_tensor(start, device=device), config=config, device=device)
    np.testing.assert_allclose(centroids.cpu().numpy(), [X[blob == b].mean(axis=0) for b in range(4)], atol=1e-4)
    assert passes < 30


def test_allocation_is_proportional_sums_to_the_budget_and_gives_every_parent_a_cell():
    allocation = _allocate(np.array([100, 50, 30, 15, 5, 0]), 20)
    assert allocation.sum() == 20 and allocation[-1] == 0 and (allocation[:-1] >= 1).all()
    np.testing.assert_array_equal(allocation, [10, 5, 3, 1, 1, 0])
    assert (_allocate(np.array([3, 1]), 4) == [3, 1]).all()


def test_split_cells_lie_inside_their_parents_in_proportion_to_their_points(device):
    centres, X, blob = _blobs(4, per_blob=500)
    keep = np.concatenate([np.flatnonzero(blob == b)[:n] for b, n in enumerate((500, 300, 150, 50))])
    X, blob = X[keep], blob[keep]
    config = load_config(overrides={"cells": {"n_cells": 40, "min_points_per_cell": 10}})
    parents = torch.as_tensor(centres, dtype=torch.float32, device=device)
    children = split_cells(torch.as_tensor(X, device=device), parents, config=config, device=device, seed=0)
    owner = assign_cells(children.to(torch.float32), parents).cpu().numpy()
    np.testing.assert_array_equal(np.bincount(owner, minlength=4), [20, 12, 6, 2])

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
    every_point = load_config(overrides={"cells": {"masses": {"rel_error": None}}})
    shuffled, ordered, all_counted = _Counting(X), _Counting(X), _Counting(X)
    cell_masses(shuffled, torch.as_tensor(centroids), config=random_order, device="cpu")
    cell_masses(ordered, torch.as_tensor(centroids), config=any_order, device="cpu")
    cell_masses(all_counted, torch.as_tensor(centroids), config=every_point, device="cpu")
    assert shuffled.handed_out == 1 and ordered.handed_out == 100 and all_counted.handed_out == 100
