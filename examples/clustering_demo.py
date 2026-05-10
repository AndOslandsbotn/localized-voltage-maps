import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.mnist import load_mnist
from lvm.clustering import create_centroids


def main() -> None:
    print("Loading MNIST...")
    X, _y = load_mnist()
    n = 64
    print("Subset init + mini-batch k-means on full data...")
    centroids = create_centroids(X, n, subset_size=3000, batch_size=2048)
    print("samples", X.shape[0], "dim", X.shape[1], "n_clusters", n)
    print("centroids", centroids.shape)


if __name__ == "__main__":
    main()
