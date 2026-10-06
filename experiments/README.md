# Experiments

Every result and figure in the papers, made with `lvm`. Each experiment is a folder with two scripts, run from the
repository root with no arguments:

```
python -m experiments.sfv.mnist.embedding.run      # computes the results and saves them next to the script
python -m experiments.sfv.mnist.embedding.plot     # draws the figure from the saved results
```

`run.py` calls the methods directly and saves its results (e.g. `embeddings.npz`, not in git); `plot.py` only reads
them, so changing how a figure looks never recomputes anything.

## Layout

```
common/       the little that every experiment shares
  datasets.py   loading the datasets (downloaded on first use, checked by fingerprint)
  style.py      figure sizes, palettes and saving (PDF + PNG), from style/
  guard.py      watch_memory(): stops a run that passes a memory limit
style/
  paper.mplstyle  fonts, font sizes, line widths, PDF font embedding (matplotlib's own format)
  figures.yaml    printed text width, figure widths, palettes, output formats and dpi
sfv/          paper 1, "Structure from Voltage": one folder per dataset, one subfolder per experiment
```

## Figures

Figures are drawn at their printed size and included in LaTeX without scaling, so font sizes are the same in every
figure. Fonts, sizes and colours come only from `style/`. To match the paper, set `text_width_in` in
`style/figures.yaml` (the paper's `\the\textwidth` in points / 72.27) and the font in `style/paper.mplstyle`, then
rerun the experiments: saved results are reused, so only the figures are redrawn.

PDFs embed fonts as TrueType (journals reject Type 3) and rasterise the dense scatter layers, so files stay small;
the PNG is a preview.

The placeholder font (Computer Modern, `cmr10`) has no symbols such as ±, ×, ≤ or Greek letters: write them as math
(`$\pm$`, `$\times$`, `$\leq$`, `$\rho_g$`), which uses the matching math font.

## Datasets

`common/datasets.py` gives `load(split, n, seed, dataset)` -> (X, labels or None): n points of a split in a random
order (`seed` picks the subset and the order), float64. Datasets are never committed. To fetch them up front:
`python -m experiments.common.datasets --download`.

- `mnist`: one fixed shuffle of all 70,000 images (`SPLIT_SEED`) gives `tune` (the first 20,000, used only to choose
  settings) and `eval` (the other 50,000, used for every reported result). Downloaded from OpenML on first use, cached
  as uint8 .npy files under `data/` and checked against SHA-256 fingerprints, so every reproduction uses the same
  images in the same order.
- `half_sphere`: points uniform on the upper half of the unit sphere in R^3, generated; `tune` and `eval` use disjoint
  random streams. It can be flattened into a disc, so a good 2-D embedding keeps every neighbourhood.
- `mnist8m`: 8.1 million deformed MNIST digits (Loosli, Canu & Bottou 2007, "infimnist"), downloaded from the LIBSVM
  site, converted to a 6.4 GB uint8 file on disk and shuffled once with a fixed seed. `stream(...)` reads it chunk
  by chunk, never holding it all in memory.

## Memory

Each `run.py` starts with `watch_memory(limit_mb=...)`: a background thread checks the memory of the process (and any
processes it started) every 0.2 s and stops it past the limit, so one run cannot exhaust the machine's memory; the
peak is printed at the end. Heavy ad-hoc commands can run under the same guard:

```
python experiments/common/guard.py --limit-mb 3000 -- <command> [args ...]
```
