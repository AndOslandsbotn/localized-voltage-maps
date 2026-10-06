import numpy as np
import pytest
import torch

from lvm_new.config import load_config
from lvm_new.graph import build_graph


def _graph(centroids, device="cpu", **graph):
    config = load_config(overrides={"graph": graph} if graph else None)
    centroids = torch.as_tensor(centroids, device=device)
    return build_graph(centroids, centroids.float(), config=config, device=device, seed=0)[0].cpu().numpy()


def _tight_and_wide(seed=0):
    # One tight cluster and one 10x wider, 60 cells each: one radius suits only one of them.
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0.0, 0.1, (60, 3)), rng.normal(5.0, 1.0, (60, 3))])


def test_a_radius_per_cell_gives_every_cell_k_edges_where_one_radius_does_not(device):
    C = _tight_and_wide()
    K = _graph(C, device, radius={"adaptive_per_cell": {"k": 5}}, connect=False)
    assert np.array_equal(K, K.T) and (K > 0).sum(axis=1).min() >= 5
    one = _graph(C, device, radius={"strategy": "knn", "knn": {"k": 5}}, connect=False)
    assert (one[60:] > 0).sum(axis=1).min() < 5


def test_connect_joins_an_isolated_cell_to_its_nearest_cell():
    # Isolated only under one radius: a radius per cell always reaches a cell's k nearest cells.
    C = np.vstack([np.random.default_rng(0).random((30, 2)), [[10.0, 10.0]]])
    K = _graph(C, radius={"strategy": "knn"})
    nearest = np.linalg.norm(C[:30] - C[30], axis=1).argmin()
    assert np.flatnonzero(K[30]).tolist() == [nearest] and K[30, nearest] == 1.0


@pytest.mark.parametrize("kernel", ["radial", "tapered", "gaussian"])
def test_a_kernel_is_one_at_distance_zero_and_zero_beyond_its_reach(kernel):
    from lvm_new.graph import kernel_weights

    config = load_config(overrides={"graph": {"kernel": {"strategy": kernel}}}).graph.kernel
    sq = torch.tensor([[0.0, 0.25, 4.0]], dtype=torch.float64)                 # distances 0, 0.5, 2 at radius 1
    K = kernel_weights(sq, torch.ones_like(sq), config)[0]
    assert K[0] == 1.0 and K[2] == 0.0 and 0.0 < K[1] <= 1.0
