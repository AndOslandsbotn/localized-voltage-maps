# MNIST8M

8.1 million deformed MNIST digits (Loosli, Canu & Bottou 2007, "infimnist"), 784 pixels, labels 0–9. Downloaded from the LIBSVM site (`mnist8m.xz`, 2.35 GB), converted to a uint8 file on disk (6.4 GB) and shuffled once with a fixed seed:

    python benchmarks/common/datasets.py --download --only mnist8m

**The data does not fit in this machine's memory** (7.8 GB of RAM; 25 GB as float32), which is the point of the experiment. All methods run on one grid, 50k, 200k, 500k, 1M, 4M and 8.1M points, each run in its own memory-guarded process (5 GB):

| Method | How it gets to 8.1M |
|---|---|
| LVM, LVM + local PCA | streamed from disk, chunk by chunk, for fitting and for placing every point; every size |
| UMAP | fitted on all n while it fits; then fitted on the largest size that fitted, with the rest placed by `transform` |
| Landmark Isomap | the same. Its transform is the standard out-of-sample rule: d(x, l) = min over x's k nearest fitted points x_j of \|x − x_j\| + D(l, x_j), then the same triangulation |
| t-SNE, Laplacian Eigenmaps | no transform in cuML, so fitted on all n as far as memory allows |

A run that exceeds the memory guard is recorded as `memory_limit`, and that method stops climbing the grid. Every run is on the GPU (the CPU comparison is on MNIST at 50k).

Every method uses its MNIST-tuned settings, unchanged; there is no tuning split. Quality is measured on 5,000 random points of each run, as on MNIST.

| Folder | What |
|---|---|
| `scaling/` | `run.py` runs everything; `plot.py` draws three figures: `figure_cost.png` (construction: fit time and memory vs fit size; streaming: time to place n points after a 50k fit, and throughput), `figure_quality.png` (the four measures vs points streamed, and vs fit size), `figure_embeddings.png` (the 8.1M embeddings) |

## Results (2026-10-03, seed 0, RTX 5060, 7.8 GB RAM, memory guard 5 GB per run)

| At n = 8.1M | Time | Peak memory | Trust | Cont | 5-NN | Global |
|---|---|---|---|---|---|---|
| LVM, streamed from disk | 78 s | 3.2 GB | 0.947 | 0.867 | 0.820 | 0.440 |
| LVM + local PCA, streamed | 61 s | 3.6 GB | 0.956 | 0.867 | 0.836 | 0.438 |
| UMAP, fit on 500k + transform | 2,720 s | 4.2 GB | 0.920 | 0.924 | 0.964 | 0.235 |
| Landmark Isomap, fit on 200k + transform | 1,218 s | 4.7 GB | 0.716 | 0.920 | 0.463 | 0.551 |
| t-SNE, Laplacian Eigenmaps | out of memory from 1M on; no transform | | | | | |

- LVM's memory is flat across n (3.2 GB at 50k and at 8.1M, mostly the libraries). Its fit takes about 3 s at every n; only the streaming pass grows.
- Every other method's memory grows with n until the guard stops it: UMAP, t-SNE and Laplacian Eigenmaps at 1M, Landmark Isomap at 500k.
- LVM's quality is stable across n (trust about 0.95, global 0.42–0.44).
- The jump from 4M (17 s) to 8.1M (78 s) is disk speed: 4M points (3.1 GB) fit in the file cache, while 8.1M (6.4 GB) are read from disk on every pass.
- The 8.1M pictures show LVM's weak spot, the global crowding: the 0s are spread out and the other digits squeezed together. LVM + local PCA helps but does not remove it.
