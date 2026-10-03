# Benchmarks

LVM against UMAP, t-SNE, Laplacian Eigenmaps and Landmark Isomap, on several datasets. Every dataset folder is laid out the same way, and all code is shared through `common/`.

## Layout

```
common/             shared code only; nothing dataset-specific
  datasets.py       registry: mnist (tune/eval splits, labels), half_sphere (generated)
  methods.py        every method (GPU and CPU versions), method_params() = tuned settings + fixed overrides
  metrics.py        trustworthiness, continuity, 5-NN accuracy (if labels), global distance rank
                    correlation, stacking (how piled up a picture is)
  runner.py         one run in its own memory-guarded process; time, memory, quality
  guard.py          memory guard for any heavy command (this machine has 7.8 GB of RAM)
  tuning.py, comparison.py, gallery.py, plotting.py, results.py
<dataset>/          mnist/, half_sphere/, mnist8m/, ...; each with the subfolders it needs, always named:
  tuning/<family>/  tune.py, plot.py -> results.csv, best.yaml, figure.png
  gpu_comparison/   run.py, plot.py -> results.csv, figure.png (methods on the GPU, by data size)
  cpu_comparison/   the same on the CPU
  tradeoff/         local vs global quality across each method's tuning grid
  gallery/          run.py -> every method's embedding side by side (figure.png)
  diagnostics/      explorations that led to design decisions (tuning data only)
  archive/          superseded results, kept for reference
demonstrations/     one folder per problem -> fix, each with a README and pinned settings
```

A dataset folder holds only small scripts that name the dataset and the methods, plus the results. The logic lives in `common/`.

## Protocol

- **Tuning:** on the tuning split only (MNIST: 20k images), 3 seeds per configuration, about 15 configurations per method, selected by mean trustworthiness (no labels). Settings are tuned on MNIST and used unchanged on the other datasets.
- **Reported results:** on the evaluation split (MNIST: 50k images, disjoint from tuning).
- **Hardware:** every method has a GPU and a CPU version, and each comparison stays on one kind of hardware.
- **Runs:**
  - each run is its own process, with 16 threads for every library and a memory guard (5 GB);
  - time is from raw data to coordinates for every point (`total_s`, fit plus transform), after a warm-up;
  - GPU memory is sampled from the driver (NVML).
- **Quality:**
  - measured on 5,000 embedded points (the global correlation on 1,500);
  - labels are only reported, never used for selection;
  - the rank-based measures cannot see points piled onto one spot, so pictures also report `visible_share` from `metrics.stacking`.
- **Metadata:** every result folder has `metadata.json`: hardware, library versions, git commit, settings.

## Demonstrations

Each `demonstrations/<problem>/` shows a problem we found and how it was fixed, before and after, on the data where it showed up.

- **Pinned settings:** every case stores its complete settings (`settings*.yaml`), so later changes to the defaults can't change it.
- **README:** states the problem, the cause with the evidence, the fix, and the status.

## Data and results in git

- **Datasets are never committed.**
  - `common/datasets.py` downloads MNIST on first use, or up front with `python benchmarks/common/datasets.py --download`.
  - It checks the cache against SHA-256 fingerprints of the arrays every result was made with.
  - Generated datasets need no files.
- **Committed:** results (`results.csv`, `*.json`, `metadata.json`) and figures.
- **Not committed:** embeddings (`*.npz`). The scripts regenerate them.
- **Guard:** `.githooks/pre-commit` refuses staged files over 2 MB. Enable it once per clone with `git config core.hooksPath .githooks`.

## Adding a dataset

1. Register it in `common/datasets.py`: loader, splits, colourings for pictures, and an optional reference view.
2. Create `<dataset>/` with the subfolders above, each holding a small `run.py` that passes the dataset name to the shared code (see `half_sphere/gallery/run.py`).
