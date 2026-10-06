# Point stacking

**Problem.** LVM placed MNIST's points on only about 1,300 distinct spots for 20,000 points. Most digits looked like a few dozen dots. Rank-based metrics did not show it, because a pile of identical points still counts as "neighbours together".

**Cause.**
- With the hard kernel 1{d ≤ r}, a point's voltages depend only on *which* cells lie within r. Every point near the same cells gets exactly the same voltages, and so the same position.
- Worse, 43% of points have no cell within r at all, because in 784 dimensions points sit on a shell farther from every centroid than the centroids are from each other. Those points fall back to their nearest cell's voltages exactly.
- Evidence: `../../mnist/diagnostics/kernels/`. Smooth kernels and a larger r did not help.

**Fix.**
1. **after:** each point weights its 3 nearest cells, wᵢ = exp(−16 (dᵢ² − d₁²) / 2s²), where s² is the point's mean excess. Subtracting d₁² removes the shell distance that every point shares with all nearby cells. Points are now correctly ordered within each cell, and every metric improves, but they still sit very close together: visible share 7%.
2. **after + local PCA:** each cell keeps LVM's global position, and its points are placed by the cell's own two principal directions. These are rotated to match LVM's offsets within the cell and scaled to a share of the distance to the nearest cell. Visible share 47%, best metrics. It's an optional add-on, timed separately (`lvm_pca`).

| Variant (MNIST tune 20k, seed 0, 150 cells) | Trust | Cont | 5-NN | Global | Visible share |
|---|---|---|---|---|---|
| before | 0.933 | 0.911 | 0.827 | 0.556 | 6% |
| after | 0.964 | 0.923 | 0.857 | 0.571 | 7% |
| after + local PCA | 0.967 | 0.924 | 0.869 | 0.572 | 47% |

**Files.**
- `settings_{before,after,after_local_pca}.yaml`: the pinned settings;
- `one.py`: one variant;
- `run.py`: runs all three variants, each in its own guarded process, then draws `figure.png`.
