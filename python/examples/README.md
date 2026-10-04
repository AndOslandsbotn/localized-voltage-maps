# Examples

Scripts that demonstrate **`lvm`**. They live in the repository only (not shipped on the wheel); run them from a clone after installing the library.

## Setup

From the **repository root**:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Editable install (`-e .`) registers **`lvm`** on your environment’s import path so examples can use normal imports (`from lvm...`) from any working directory.

For **plots** (centroid grids, `lvm.viz`), add the optional extra:

```bash
pip install -e ".[viz]"
```

## Run

With the virtualenv activated and the repo as current directory (or anywhere, after `pip install -e .`):

```bash
python examples/clustering_demo.py
```

Or as a module from the **repository root** (so the `examples` package resolves):

```bash
python -m examples.clustering_demo
```

**MNIST** files are cached under `~/.cache/localized-voltage-maps/mnist` (or `$XDG_CACHE_HOME/...`). The loader lives in **`lvm.datasets.mnist`** (`load_mnist`).

## Contents

| Script | What it does |
|--------|----------------|
| `clustering_demo.py` | Downloads MNIST via `lvm.datasets`, builds centroids with `create_centroids`. With `.[viz]` installed, writes `examples/output/centroids_mnist.png`. |
