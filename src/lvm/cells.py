import numpy as np
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus


def random_subset_indices(n_samples: int, subset_size: int, rng: np.random.Generator) -> np.ndarray:
    m = min(subset_size, n_samples)
    return rng.choice(n_samples, size=m, replace=False)

def build_voroni_cells(
    X: np.ndarray,
    n_clusters: int,
    *,
    subset_size: int | None = None,
    batch_size: int = 1024,
    random_state: int | None = 0,
    max_iter: int = 100,
    n_local_trials: int = 2,
) -> np.ndarray:
    """
    1) Random subset of rows.
    2) k-means++ on that subset only -> initial centers (no full Lloyd on subset).
    3) MiniBatchKMeans on the full X, starting from those centers.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    if subset_size is None:
        subset_size = min(10_000, n)
    subset_size = max(n_clusters, min(subset_size, n))

    sub_idx = random_subset_indices(n, subset_size, rng)
    sub = X[sub_idx]

    init_centers, _ = kmeans_plusplus(
        sub,
        n_clusters=n_clusters,
        random_state=random_state,
        n_local_trials=n_local_trials,
    )

    mb = MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        batch_size=min(batch_size, n),
        max_iter=max_iter,
        random_state=random_state,
    )
    mb.fit(X)
    return mb.cluster_centers_
