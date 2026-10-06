# GPU comparison

LVM, UMAP and Laplacian Eigenmaps, all on the GPU, at the settings in `../tuning/*/best.yaml`,
on random subsets (5k-50k) of the evaluation split, 3 seeds each, each run in its own process.


    python experiments_old/sfv/gpu_comparison/run.py    # results.csv, metadata.json (rerunning skips finished runs)
    python experiments_old/sfv/gpu_comparison/plot.py   # figure.png
