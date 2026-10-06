# Python library design (draft for review)

Status: **plan only**, nothing implemented. Agreed in discussion on 2026-10-04; open points are marked **Open**.

## Goals

- An installable library with a scikit-learn-style interface: `from lvm import LocalizedVoltageMaps`, then `fit`, `transform`, `fit_transform`.
- Streaming in every step: a fit reads the data in chunks and never needs it all in memory.
- Built for the hierarchy from the start. The unit is a **region**, not a level: one fitted region is a `RegionModel`, a hierarchy is a tree of them, and a level is just the regions at one depth. Level 0 is a single region (the whole data).
- Built so that regions can later be fitted in parallel or on several machines (not implemented now, see "Rules for later distribution").

Stages: (1) single level built from these pieces, (2) several levels fitted one region after another, (3) parallel and distributed fitting as a project of its own.

## Public interface

```python
from lvm import LocalizedVoltageMaps

model = LocalizedVoltageMaps(n_components=2, n_cells=300, config="my_settings.yaml")   # any argument may be left out
model.fit(X)                       # returns the model itself
Z = model.transform(X)             # (n, n_components) array
Z = model.fit_transform(X)
for Z_chunk in model.transform_chunks(X):   # one array per chunk, for data too big to hold
    ...
```

**Constructor arguments.** The most important settings are direct arguments; everything else comes from a YAML file. Every direct argument is exactly a setting, with the same values, and defaults to `None` ("not given"): its value then comes from the `config` YAML if that sets it, else from the package defaults (`lvm/config/config.yaml`). A given argument wins over both. So the YAML and the direct arguments are equivalent, and all rules (types, ranges, allowed values) live in the config models.

| Argument | Meaning | Default |
|---|---|---|
| `n_components` | dimension of the embedding | 2 |
| `n_cells` | number of cells (graph nodes) per region | 300 |
| `device` | `"auto"` (CUDA if available, else CPU), `"cuda"` or `"cpu"` | `"auto"` |
| `random_state` | seed | 0 |
| `local_chart` | the second scale: place points within their cell by the cell's local PCA (config `embedding.local_scale: pca`). `"last"`: charts at the deepest level during `fit`, other levels on demand; `"all"`: charts at every level during `fit` (more time, every level ready); `"none"`: plain coordinates. With one level, `"last"` and `"all"` are the same (see "Local charts") | `"last"` |
| `chunk_size` | points per chunk when an array is read in pieces. A re-iterable's own pieces are used as they come: re-cutting them would copy every chunk (measured: as costly as LVM's own computation per chunk), so for a loader its batch size is the chunk size. Streaming k-means updates once per chunk, so results depend on it; that's also why it is fixed rather than estimated from free memory | 10,000 |
| `config` | path to a YAML file with any other settings, merged over the package's `config.yaml` | `None` = the package defaults |

A direct argument overrides the same setting in the YAML. Following scikit-learn, `__init__` only stores its arguments, unchanged; all checking and work happen in `fit`. That gives `get_params()`, `set_params()` and `clone()` for free (inherit `sklearn.base.BaseEstimator` and `TransformerMixin`).

**Inputs.** `fit`, `transform` and `transform_chunks` accept:
- an array: anything with a shape that can be sliced (NumPy, memory-mapped, HDF5, Zarr, torch, CuPy), read in chunks internally;
- a re-iterable of chunks: an object that starts a fresh pass each time it is looped over (for example a torch `DataLoader`, or the readers in `lvm.io`).

A one-shot generator or iterator is refused with an explanation (it can only be read once, and `fit` reads the data several times). Data must be in shuffled order, so that its first points form a fair sample; for sorted data set `data.shuffled: false` in the config and LVM samples over a full pass instead.

**Fitted attributes** (scikit-learn style, ending in `_`, available after `fit`):

| Attribute | Meaning |
|---|---|
| `root_` | the `RegionModel` of level 0 |
| `config_` | the complete settings used (YAML + direct arguments) |
| `device_` | the device actually used |
| `n_features_in_` | number of input features (scikit-learn convention) |

A region's own results (centroids, masses, landmarks, ρ_g, dimension, ...) are attributes of its `RegionModel`: `model.root_` for level 0, `model.regions(level)` for any level. No shortcuts on the estimator: one way to reach any region's results, at every level alike.

**Other methods.**
- `regions(level=0)`: the fitted regions at one depth (a list of `RegionModel`).
- `save(path)` and `LocalizedVoltageMaps.load(path)`: store and reload a fitted model, as a folder: `model.yaml` (library and format version, n_features, levels, the complete settings; readable without Python) and `regions/<id>.npz` (the arrays of each region, loaded without pickle). One file per region suits fitting regions on different machines; a single zipped file can be offered on top later.
- `fit_charts(X, level=k)`: fit the local charts of a level in advance (see "Local charts").
- Later, with several levels: `transform(X, level=k)`. **Open (for stage 2):** whether a deep level returns each point's region and its coordinates within that region's map, or one combined map (needs the chained distances of paper 2).

## Internal structure

**`RegionModel`** (today's `LevelModel`): the fitted model of one region. It holds the region's id (a path such as `(0, 3, 1)`, `()` for the root), its cells and masses, graph, ρ_g, landmarks and voltage maps, the embedding, the local chart, and its children (empty at level 0). Methods: `transform_chunk(X)` (coordinates of one chunk), `voltages(X)`, and conversion to and from plain arrays for saving.

**`fit_region(data, config, *, region_id=(), seed)` → `RegionModel`**: the fit of one region, a pure function (no shared state; same inputs give the same model), so it can later run in another process or on another machine. Its steps, in order, each in its own module and configured by its own config section:

| Step | Module | Config section |
|---|---|---|
| 1. sample the region (shuffled prefix, or reservoir over a full pass) | `data` | `sample` |
| 2. intrinsic dimension d̂ | `dimension` | `dimension` |
| 3. cells: k-means on the sample, then streaming refinement | `cells` | `cells.kmeans`, `cells.refine` |
| 4. cell masses (a streamed pass) | `cells` | `cells.masses` |
| 5. graph: radius per cell, kernel, connecting pieces | `graph` | `graph` |
| 6. landmarks and their reach ρ_g, chosen together (the reach alternation) | `voltage` | `voltage`, `landmarks` |
| 7. voltage maps of the landmarks (support and distance thresholds) | `voltage` | `voltage`, `embedding.distance_floor` |
| 8. embedding: Landmark MDS with chaining | `embedding` | `embedding` |
| 9. local chart (optional second scale) | `embedding` | `embedding.local_scale` |

**Local charts.** A chart is fitted from a region's raw points (per cell, at most 256 of them), so it is cheapest while the region's sample is in memory, during `fit`; samples are not kept afterwards (with many regions they would cost too much). Hence:
- with `local_chart="last"` (the default), `fit` fits the charts of the deepest level only (with one level: the root; nothing else changes); with `"all"` it fits them at every level, so every level's embedding is ready without another pass;
- `transform(X, level=k)` for a level without charts first takes one pass over X to fit them (at most 256 points per cell), then places the points, and keeps the charts in the model (reused by later calls and by `save`); `fit_charts(X, level=k)` does the first part explicitly;
- an on-demand chart is only as good as the data it was fitted from: a cell needs at least k + 1 points for k directions, ideally a few dozen. Charts never move the cells, only how points spread around them.

At one level a chart costs about 0.3–0.4 s on top of a 1.4 s fit (MNIST, 50k points), so fitting charts only where they are used matters once there are many regions.

**`LocalizedVoltageMaps`** turns the input into a re-iterable source, resolves the device and settings, calls `fit_region` for the root (stage 1), and for `transform` sends each chunk to the right region model (at level 0: the root).

## Modules

```
lvm/
  __init__.py      exports LocalizedVoltageMaps, RegionModel, fit_region, load_config
  estimator.py     LocalizedVoltageMaps: the scikit-learn estimator; resolves inputs, device and settings; runs the levels
  region.py        RegionModel + fit_region (today's pipeline.py)
  data.py          turning inputs into re-iterable chunk sources; refusing one-shot iterators; the region's sample
  io.py            readers for files: CsvChunks, NpyChunks (re-iterable); plain reads, one chunk at a time, not memory
                   maps, so files bigger than RAM stream without filling memory (pages of a memory map count as the
                   process's memory)
  cli.py           command line: `lvm fit data.csv --out model.npz`, `lvm transform model.npz data.csv --out coords.npy`
  __main__.py      `python -m lvm` → cli
  config/          config.py (pydantic models) + config.yaml (the defaults)
  compute.py       precisions (data path, solve) and the memory per block
  dimension.py, cells.py, graph.py, voltage.py, embedding.py      the steps
```

**Removed:** `lvm.py` (the stale sketch; replaced by `estimator.py`), `pipeline.py` (becomes `region.py`), `stream.py` (split into `data.py` and `io.py`), `datasets/mnist.py` (dataset-specific: belongs in `experiments/common/datasets.py`).

**Device by step.** With `device="auto"` each step picks its implementation by the device, unless the config names one explicitly. For example k-means: cuML on the GPU, FAISS on the CPU (today it is always cuML, which needs a GPU even when `compute.device` is `cpu`). That becomes a `strategy: auto` in the config sections concerned.

## Rules for later distribution

Kept from the start, so that stage 3 needs no redesign:
1. `fit_region` is pure: it gets its data, its settings and its seed, and returns a model; no shared or global state.
2. A `RegionModel` can be saved and loaded (arrays + settings), so results can move between processes and machines.
3. Routing (which points go to which region) is separate from fitting (a region sees only its own points).
4. Each region's seed comes from the global seed and the region's id, so results don't depend on the order in which regions finish.
5. `transform` works chunk by chunk, each chunk independently, routed down the tree of regions.

## Order of work

The library was built from scratch as `lvm_new` and renamed to `lvm` on 2026-10-06; the previous library is kept as `python/src/lvm_old/` (its tests in `python/tests_old/`, its experiments in `experiments_old/`), for reference only. The two never import from each other. The paper's experiments are rebuilt in `experiments/` with the new `lvm`.

Each step keeps all tests passing, including the regression test (pinned results), and is committed on its own:
1. `estimator.py`, `data.py` (inputs as chunk sources, the region's sample), `compute.py` (precisions, block memory), `config.py` (pydantic models + `config.yaml`, sections added as the steps that use them arrive). `fit` works once the step modules exist; nothing wraps the old `fit_level`.
2. `LevelModel` → `RegionModel`, `pipeline.py` → `region.py` with `fit_region`; region ids and per-region seeds.
3. `device="auto"` and device-based strategies (k-means first).
4. `io.py` and `cli.py`; delete the removed modules.
5. scikit-learn compatibility checks (`get_params`, `clone`, scikit-learn's estimator checks where they apply), `save`/`load`.
6. `pyproject.toml`: correct dependencies and extras (GPU: torch with CUDA, cuML; CPU: FAISS); README with usage.
7. Move the experiments' method wrappers (`experiments/common/methods.py`) from `fit_level` to `LocalizedVoltageMaps`.

## Decided

- `n_cells` default 300, the value tuned on MNIST and used in the paper's benchmarks (changed from 1000 on 2026-10-06: at 1000 cells the landmark selection alone took 6 s on MNIST 50k).
- `chunk_size` is a direct argument with a fixed default of 10,000 (an opt-in `"auto"` from free memory could come later).
- `local_chart="last"` (default): charts at the deepest level during `fit`, at other levels on demand; `"all"` fits them at every level during `fit` (the user's choice to spend the time); `"none"`: no charts (see "Local charts").
- Save format: a folder with a readable `model.yaml` and one `.npz` per region.

## Open questions

- None for stage 1. For stage 2: what `transform(X, level=k)` returns at a deep level (each point's region and coordinates within it, or one combined map via the chained distances of paper 2).
