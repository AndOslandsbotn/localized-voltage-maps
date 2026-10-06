import numpy as np
import pytest
import torch

from lvm_new.config import load_config
from lvm_new.voltage import _mutual_information, _reached, choose_landmarks, threshold, voltage_maps


def _graph(n_cells, seed=0):
    # Cells at random points of the unit square, a Gaussian kernel of width 0.1, random masses.
    rng = np.random.default_rng(seed)
    points = rng.random((n_cells, 2))
    kernel = np.exp(-((points[:, None] - points[None]) ** 2).sum(axis=2) / (2 * 0.1 ** 2))
    np.fill_diagonal(kernel, 0.0)
    masses = rng.random(n_cells) + 0.1
    return kernel, masses / masses.sum()


def test_a_map_is_one_at_its_source_and_a_grounded_average_elsewhere(device):
    # Dividing row i of L v = 0 by p_i: v_i = sum_j K_ij p_j v_j / (rho_g + sum_j K_ij p_j) at every cell but the source.
    kernel, masses = _graph(40)
    rho_g, sources = 0.1, [0, 7, 23]
    maps = voltage_maps(torch.as_tensor(kernel, device=device), torch.as_tensor(masses, device=device), rho_g,
                        torch.tensor(sources, device=device)).cpu().numpy()
    average = maps @ (kernel * masses).T / (rho_g + kernel @ masses)
    for v, a, s in zip(maps, average, sources):
        others = np.arange(len(v)) != s
        assert v[s] == 1.0
        np.testing.assert_allclose(v[others], a[others], rtol=1e-9, atol=1e-14)
        assert np.all((v[others] > 0) & (v[others] < 1))


def test_the_greedy_mutual_information_equals_the_direct_estimate_of_the_chosen_maps(device):
    kernel, masses = _graph(80)
    K, p = torch.as_tensor(kernel, device=device), torch.as_tensor(masses, device=device)
    maps = threshold(voltage_maps(K, p, 0.05, torch.arange(80, device=device)), 1e-3)
    chosen, information = _mutual_information(maps, p, n_landmarks=6, noise_std=0.01)
    V = maps.cpu().numpy()
    for i in range(1, 7):
        X = V[chosen[:i].cpu().numpy()].T                                         # (cells, landmarks)
        sq = ((X[:, None] - X[None]) ** 2).sum(axis=2)
        direct = -(masses * np.log(np.exp(-sq / (2 * 0.01 ** 2)) @ masses)).sum()
        assert information[i - 1].item() == pytest.approx(direct, rel=1e-4)


def test_reach_gives_the_largest_rho_g_at_which_the_chosen_landmarks_reach_the_data(device):
    kernel, masses = _graph(80)
    K, p = torch.as_tensor(kernel, device=device), torch.as_tensor(masses, device=device)
    config = load_config(overrides={"landmarks": {"count": {"strategy": "fixed", "fixed": {"n": 4}}}})
    reach, tau = config.landmarks.reach, config.voltage.threshold
    landmarks, rho_g = choose_landmarks(K, p, dimension=2.0, config=config)
    assert len(landmarks) == 4
    assert _reached(K, p, rho_g, landmarks, tau, reach.share) >= 4
    assert _reached(K, p, rho_g * (1 + reach.rel_tolerance), landmarks, tau, reach.share) < 4
