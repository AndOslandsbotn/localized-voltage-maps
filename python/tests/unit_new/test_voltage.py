import numpy as np
import torch

from lvm_new.voltage import voltage_maps


def test_a_map_is_one_at_its_source_and_a_grounded_average_elsewhere(device):
    # Dividing row i of L v = 0 by p_i: v_i = sum_j K_ij p_j v_j / (rho_g + sum_j K_ij p_j) at every cell but the source.
    rng = np.random.default_rng(0)
    points = rng.random((40, 2))
    kernel = np.exp(-((points[:, None] - points[None]) ** 2).sum(axis=2) / (2 * 0.1 ** 2))
    np.fill_diagonal(kernel, 0.0)
    masses = rng.random(40) + 0.1
    masses /= masses.sum()
    rho_g, sources = 0.1, [0, 7, 23]
    maps = voltage_maps(torch.as_tensor(kernel, device=device), torch.as_tensor(masses, device=device), rho_g,
                        torch.tensor(sources, device=device)).cpu().numpy()
    average = maps @ (kernel * masses).T / (rho_g + kernel @ masses)
    for v, a, s in zip(maps, average, sources):
        others = np.arange(40) != s
        assert v[s] == 1.0
        np.testing.assert_allclose(v[others], a[others], rtol=1e-9, atol=1e-14)
        assert np.all((v[others] > 0) & (v[others] < 1))
