import numpy as np
import pytest
import torch
from scipy.spatial.distance import pdist
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap

from experiments.common.methods import landmark_isomap_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_landmark_isomap_with_every_point_a_landmark_is_isomap():
    # With all points as landmarks, Landmark MDS is classical MDS: the embedding is full Isomap's (up to rotation).
    X = make_swiss_roll(n_samples=400, random_state=0)[0].astype(np.float32)
    ours = landmark_isomap_gpu(X, seed=0, n_neighbors=10, n_landmarks=len(X))
    reference = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    np.testing.assert_allclose(pdist(ours), pdist(reference), rtol=1e-3, atol=1e-3 * pdist(reference).max())
