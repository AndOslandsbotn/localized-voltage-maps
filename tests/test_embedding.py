import numpy as np
import pytest

from lvm.config import load_config
from lvm.embedding import fit_embedding

TAU = 1e-3


def _fit(V, p, n_components=2):
    config = load_config(overrides={"embedding": {"strategy": "log_mds",
                                                  "log_mds": {"n_components": n_components}}}).embedding
    return fit_embedding(V, p, config=config, tau=TAU)


def test_log_mds_is_weighted_pca_of_log_features():
    rng = np.random.default_rng(0)
    V = rng.random((6, 50))
    p = rng.dirichlet(np.ones(50))
    emb = _fit(V, p, n_components=3)

    assert np.allclose(emb.components @ emb.components.T, np.eye(3))
    assert np.allclose(emb.transform(V), emb.cell_coords)
    # Weighted mean of the coordinates is zero, and the axes are ordered by variance.
    assert np.allclose(p @ emb.cell_coords, 0.0)
    var = p @ emb.cell_coords**2
    assert np.all(np.diff(var) <= 1e-12)


def test_zeroed_voltages_become_finite_far_features():
    V = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    emb = _fit(V, np.full(3, 1 / 3), n_components=1)
    assert np.all(np.isfinite(emb.cell_coords))
    # The two end cells are mirror images, the middle one sits between them.
    c = emb.cell_coords[:, 0]
    assert c[0] == pytest.approx(-c[2]) and c[1] == pytest.approx(0.0, abs=1e-12)


def test_zero_mass_cells_get_coordinates_without_moving_the_axes():
    rng = np.random.default_rng(1)
    V = rng.random((4, 30))
    p = rng.dirichlet(np.ones(30))
    p_zero = p.copy()
    p_zero[5] = 0.0
    V_moved = V.copy()
    V_moved[:, 5] = rng.random(4)  # changing a zero-mass cell's voltages...
    a, b = _fit(V, p_zero), _fit(V_moved, p_zero)
    assert np.allclose(a.components, b.components)  # ...doesn't change the fit
    assert np.all(np.isfinite(b.cell_coords[5]))


def _lmds_config(missing="clip", n_components=2):
    return load_config(overrides={"embedding": {"strategy": "landmark_mds",
                                                "landmark_mds": {"missing": missing, "n_components": n_components}}}).embedding


def _voltages_from_points(points, landmark_idx):
    # Exact "voltages" v = exp(-d) from true Euclidean distances, so -log v = d.
    from scipy.spatial.distance import cdist

    return np.exp(-cdist(points[landmark_idx], points))


def test_landmark_mds_recovers_euclidean_geometry_exactly():
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(0)
    points = rng.random((200, 2)) * 3
    landmarks = np.array([0, 1, 2, 3, 4])
    V = _voltages_from_points(points, landmarks)
    emb = fit_embedding(V, np.full(200, 1 / 200), config=_lmds_config(), tau=1e-12, landmark_cells=landmarks)
    # Same pairwise distances as the true points: correct up to rotation/reflection/translation.
    np.testing.assert_allclose(pdist(emb.cell_coords), pdist(points), rtol=1e-6, atol=1e-6)


def test_chain_fills_distances_outside_the_support():
    rng = np.random.default_rng(1)
    points = np.column_stack([np.linspace(0, 10, 300), 0.3 * rng.random(300)])
    landmarks = np.array([0, 75, 150, 225, 299])
    V = _voltages_from_points(points, landmarks)
    tau = np.exp(-4.0)                                  # each landmark only reaches distance 4
    V = np.where(V >= tau, V, 0.0)
    p = np.full(300, 1 / 300)
    clip = fit_embedding(V, p, config=_lmds_config("clip", 1), tau=tau, landmark_cells=landmarks)
    chain = fit_embedding(V, p, config=_lmds_config("chain", 1), tau=tau, landmark_cells=landmarks)
    # Chained landmark distances follow the line (2.5 apart each step); clipped ones saturate at 4.
    assert chain.landmark_D[0, 4] == pytest.approx(10.0, rel=0.05)
    assert clip.landmark_D[0, 4] == pytest.approx(4.0)
    # The 1-D chained embedding keeps the order along the line; correlation with position is high.
    assert abs(np.corrcoef(chain.cell_coords[:, 0], points[:, 0])[0, 1]) > 0.99


def test_landmark_mds_transform_on_gpu_matches_numpy():
    import torch

    rng = np.random.default_rng(2)
    points = rng.random((100, 2))
    landmarks = np.arange(6)
    V = _voltages_from_points(points, landmarks)
    emb = fit_embedding(V, np.full(100, 0.01), config=_lmds_config("chain"), tau=1e-3, landmark_cells=landmarks)
    gpu = emb.transform(torch.as_tensor(V, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(gpu, emb.transform(V), rtol=1e-8, atol=1e-8)
