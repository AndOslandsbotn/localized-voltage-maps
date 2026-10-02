# Benchmarks: LVM vs UMAP vs Laplacian Eigenmaps

All experiments use MNIST (70,000 images), split once and for all into a
**tuning split** (20,000 images, used only to choose hyperparameters) and a
disjoint **evaluation split** (50,000 images, used for every reported result).
See `common/data.py`.

## Order

1. `tuning/{lvm,umap,le}/tune.py`: tune each method on the tuning split; writes `best.yaml`.
2. `gpu_comparison/run.py` and `cpu_comparison/run.py`: compare the methods at their tuned
   settings on the evaluation split.
3. `*/plot.py`: one figure per experiment (`figure.png`), drawn from its `results.csv`.

## Fairness rules

- Every method runs on GPU and on CPU, and each comparison is within one kind of hardware:
  LVM (cuML k-means + torch | FAISS + torch), UMAP (cuML | umap-learn),
  Laplacian Eigenmaps (cuML | scikit-learn with the AMG eigensolver).
- Equal tuning effort: ~15 configurations per method over its most influential parameters,
  selected by mean trustworthiness (no labels), 3 seeds each.
- Each comparison run is its own process, with the same thread count (16) for every library.
- GPU memory is sampled from the driver (NVML), so every allocator is counted the same way.
- Quality on 5,000 embedded points: trustworthiness, continuity, 5-NN accuracy (labels: reported,
  never used for selection), and global distance rank correlation on 1,500 points.
- Each result folder has `metadata.json`: hardware, library versions, git commit, settings.

## Layout

- `common/`: shared code (data split, metrics, method registry, runner, results I/O, plotting).
- `tuning/<method>/`: tune.py, plot.py, results.csv, best.yaml, figure.png, metadata.json.
- `gpu_comparison/`, `cpu_comparison/`: run.py, plot.py, results.csv, figure.png, metadata.json.
- `diagnostics/`: exploratory analyses (tuning split only).
- `archive/`: results and scripts from before this structure (2026-10-01), kept for reference.
  They used the full 70k (no tuning/evaluation split) and older solvers (ARPACK for LE).
