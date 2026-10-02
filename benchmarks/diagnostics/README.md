# Diagnostics

Exploratory analyses, run on the tuning split only so they never touch the evaluation data.

- `feature_space/run.py`: how much quality LVM's full L-dimensional voltage features hold
  compared with its 2-D embedding (is the 2-D projection the bottleneck?). Writes results.txt.
