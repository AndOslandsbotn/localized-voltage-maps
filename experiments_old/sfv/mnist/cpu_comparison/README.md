# CPU comparison

LVM, UMAP and Laplacian Eigenmaps, all on the CPU, at the settings in `../tuning/*/best.yaml`,
on random subsets (5k-50k) of the evaluation split, 3 seeds each, each run in its own process.
Laplacian Eigenmaps is capped at 40k (`MAX_N` in run.py): at 70k the old ARPACK solver exhausted the machine's 7 GB of RAM.

    python experiments_old/sfv/cpu_comparison/run.py    # results.csv, metadata.json (rerunning skips finished runs)
    python experiments_old/sfv/cpu_comparison/plot.py   # figure.png
