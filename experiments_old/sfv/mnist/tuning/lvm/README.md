# LVM tuning

Grid in `tune.py`, run on the tuning split (20k images), 3 seeds per configuration, on the GPU.
The configuration with the highest mean trustworthiness goes to `best.yaml`, which the
comparisons read. The CPU version uses the same settings.

    python experiments_old/sfv/tuning/lvm/tune.py   # results.csv, best.yaml, metadata.json
    python experiments_old/sfv/tuning/lvm/plot.py   # figure.png
