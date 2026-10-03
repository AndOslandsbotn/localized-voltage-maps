# MNIST diagnostics

Explorations behind design decisions. They run on the tuning split only, so they never touch the evaluation data. Each folder has its scripts, results and figures.

| Folder | Question | Outcome |
|---|---|---|
| `feature_space/` | Is the 2-D projection the bottleneck of LVM's quality? | Compares the full voltage features with the 2-D embedding |
| `landmark_mds/` | Embedding (log-MDS vs Landmark MDS), missing distances (clip vs chain), landmark count and selection | Landmark MDS with chained distances, d̂ + 1 landmarks chosen by mutual information |
| `reach/` | Choose ρ_g by `reach` (a typical point reached by all landmarks) instead of `coverage` overlap | `reach` matches or beats `coverage`, with no overlap to tune |
| `reach_alternation/` | Choose landmarks once per round instead of at every bisection probe | Same ρ_g and quality, 2–4× faster search |
| `kmeans_sample/` | How many points per cell k-means needs | Still improving at 133 per cell; a test of 256 is open |
| `kernels/` | Points piling up: kernel shape, point-step variants, sharpness, local scale and local PCA | k-nearest-cells point step (k 3, sharpness 16); optional local PCA chart |
| `tsne_optimizer/` | cuML t-SNE optimiser settings | Adaptive mode off, learning rate n/3 (lowest KL divergence) |
| `distance_floor/` | Fix A (distances read below τ, down to 10⁻¹⁰) on MNIST | Local quality unchanged, global 0.502 → 0.514; no pairs left to chain (1.3% before) |
| `point_extension/` | Point voltages: grounded (zero-mass node with its own ground) vs average (paper's Def. 10, no ground) | average: trust 0.967 → 0.972, cont 0.919 → 0.927, 5-NN 0.890 → 0.897, global unchanged |
