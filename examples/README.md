# Examples

Small scripts that show how to use **`lvm`** (the library in the parent directory). They are **not** part of the installed package; keep them here for learning and experimentation.

## Setup

From the **repository root** (parent of this folder):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Installing with `-e .` puts **`lvm`** on your environment’s path. The example scripts still add the repo root to `sys.path` so you can also run them **without** `-e .` as long as dependencies are installed.

## Run

From the **repository root**:

```bash
python examples/clustering_demo.py
```

Or as a module:

```bash
python -m examples.clustering_demo
```

**MNIST** files are cached under `~/.cache/localized-voltage-maps/mnist` (or `$XDG_CACHE_HOME/...`).

## Contents

| Script | What it does |
|--------|----------------|
| `clustering_demo.py` | Downloads MNIST, builds centroids with `create_centroids`. |
| `mnist.py` | IDX download + `load_mnist()` helper used by the demo. |
