# Experiments

Every result and figure in the papers, made with `lvm`. Each experiment runs from the repository root with no
arguments and writes its figure next to its code:

```
python -m experiments.sfv.mnist.embedding.run
python -m experiments.sfv.mnist.embedding.run --recompute     # compute again instead of reusing saved results
```

Results are saved as they are computed and reused, so changing how a figure looks only redraws it.

## Layout

```
common/       shared code
  datasets.py   loading the datasets (downloaded on first use, checked by fingerprint)
  methods.py    every method behind one interface: fit_transform(X, seed, device) -> coordinates
  embed.py      one embedding in its own process (started by runner.embed)
  runner.py     embed(...): computed under the memory guard, saved in the experiment's embeddings/, reused
  guard.py      runs a command, killing it if its memory passes a limit
  plots.py      reusable panels and figures (one embedding, several side by side, ...)
  style.py      figure sizes, palettes and saving (PDF + PNG), from style/
  cli.py        the one command-line option every experiment has (--recompute)
style/
  paper.mplstyle  fonts, font sizes, line widths, PDF font embedding (matplotlib's own format)
  figures.yaml    printed text width, figure widths, palettes, output formats and dpi
sfv/          paper 1, "Structure from Voltage": one folder per dataset, one subfolder per experiment
```

An experiment's `run.py` only says what it computes and which figure it draws; everything else is in `common/`.
Embeddings (`embeddings/*.npz`) are not in git.

## Figures

Figures are drawn at their printed size and included in LaTeX without scaling, so font sizes are the same in every
figure. Fonts, sizes and colours come only from `style/`. To match the paper, set `text_width_in` in
`style/figures.yaml` (the paper's `\the\textwidth` in points / 72.27) and the font in `style/paper.mplstyle`, then
rerun the experiments: saved results are reused, so only the figures are redrawn.

PDFs embed fonts as TrueType (journals reject Type 3) and rasterise the dense scatter layers, so files stay small;
the PNG is a preview.

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

Every fit runs in its own process under `guard.py`, which sums the memory of the process tree every 0.2 s and kills
it past a limit (3 GB by default), so one run cannot exhaust the machine's memory. Heavy ad-hoc commands can go
through it too:

```
python experiments/common/guard.py --limit-mb 3000 -- <command> [args ...]
```
