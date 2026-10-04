# MNIST

70,000 images, split once (fixed shuffle `SPLIT_SEED`, see `common/datasets.py`):
- **tune:** 20,000 images, used only to choose settings;
- **eval:** 50,000 images, used for every reported result.

| Folder | What | Run |
|---|---|---|
| `tuning/{lvm,umap,tsne,le,lisomap}/` | each method's grid on the tuning split → `best.yaml` | `tune.py`, then `plot.py` |
| `gpu_comparison/`, `cpu_comparison/` | tuned methods on 5k to 50k evaluation images, 3 seeds | `run.py`, then `plot.py` |
| `tradeoff/` | trustworthiness, continuity and 5-NN against global correlation, every tuned configuration, per-method Pareto fronts | `plot.py` |
| `gallery/` | every method's embedding of the 50k evaluation images | `run.py` |
| `diagnostics/` | explorations behind design decisions (see its README) | |
| `archive/` | superseded results: each folder is named after what changed since | |
