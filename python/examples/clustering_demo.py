from pathlib import Path

from lvm_old.clustering import create_centroids
from lvm_old.datasets import load_mnist

try:
    from lvm_old.viz import plot_flat_image_grid
except ImportError:
    plot_flat_image_grid = None


def main() -> None:
    print("Loading MNIST...")
    X, _y = load_mnist()
    n = 64
    print("Subset init + mini-batch k-means on full data...")
    centroids = create_centroids(X, n, subset_size=3000, batch_size=2048)
    print("samples", X.shape[0], "dim", X.shape[1], "n_clusters", n)
    print("centroids", centroids.shape)

    if plot_flat_image_grid is not None:
        out = Path(__file__).resolve().parent / "output" / "centroids_mnist.png"
        plot_flat_image_grid(
            centroids,
            (28, 28),
            path=out,
            show=False,
            title="MNIST centroids (subset init + mini-batch k-means)",
        )
        print("wrote", out)
    else:
        print("Skipping centroid plot: install the viz extra with: pip install -e '.[viz]'")


if __name__ == "__main__":
    main()
