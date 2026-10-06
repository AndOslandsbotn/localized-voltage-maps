# t-SNE tuning

The grid is in `tune.py`: perplexity {5, 15, 30, 50, 100} × late exaggeration {1, 2, 4}. That is 15 configurations, the same budget as UMAP.

Protocol:
- Run on the tuning split (20k images), 3 seeds per configuration, on the GPU (cuML).
- The configuration with the highest mean trustworthiness goes to `best.yaml`, which the comparisons read.
- The CPU version (openTSNE) uses the same settings.

The optimiser settings are not tuned. They were chosen once by t-SNE's own objective (KL divergence); see `diagnostics/tsne_optimizer/README.md`.

    python experiments_old/sfv/tuning/tsne/tune.py   # results.csv, best.yaml, metadata.json
    python experiments_old/sfv/tuning/tsne/plot.py   # figure.png
