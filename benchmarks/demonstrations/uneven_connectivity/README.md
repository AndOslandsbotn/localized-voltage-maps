# Uneven connectivity

**Status: fixed** (`graph.radius: adaptive_per_cell`, the default since 2026-10-03). Both cases stay pinned for comparison.

**Problem.** In LVM's MNIST pictures one or two digits take most of the frame and the rest are pressed together. On MNIST8M the 0s fill the frame (room 1.69× their pixel-space share) while the 1s get 0.11×. Removing the 0s from MNIST doesn't help: the 2s and 6s take over (`mnist/diagnostics/without_zeros`).

**Cause.**
- The cell graph used one radius r for every cell: the median distance to a cell's 10th-nearest cell.
- Digits differ in spread. Cells of compact digits (the 1s) are close together, so they get many edges; cells of varied digits (the 0s, MNIST8M's deformed 0s most of all) are farther apart, so they get few. On MNIST8M the 0s' cells have 2 edges each, the 1s' 12 (`mnist8m/diagnostics/connectivity`).
- Voltage follows the graph. A digit with few edges forms thin chains and even separate pieces, so voltage has to detour: on MNIST8M the route between two 0-cells is 3.7× their straight distance (others 1.2–1.6), and their voltage distances come out 2.2× typical. A well-connected digit is the reverse: squeezed.
- Landmark MDS then gives the frame to the largest distances, so the most under-connected digit takes it.
- **Why the half sphere doesn't show it:** in both datasets the cells' spacing varies about as much (90th/10th percentile 1.37 on MNIST, 1.33 on the sphere). But the number of cells within a radius grows like (radius / spacing)^d. With MNIST's intrinsic dimension d ≈ 12, a 1.37× difference in spacing is a ~40× difference in neighbours; on the 2-dimensional sphere it is ~1.8×. Uneven connectivity needs uneven density **and** a high intrinsic dimension.

**Fix.** `graph.radius: adaptive_per_cell` gives each cell its own radius, the distance to its 10th-nearest cell, and joins two cells when either is within the other's radius (r_ij = max(r_i, r_j), the symmetric kNN graph). Every cell then has at least 10 edges wherever it lies, the same principle as UMAP's per-point scale.

| Seed 0, LVM + local PCA chart | Edges per cell (p10/p50/p90) | Room 0s / 1s | Trust | Cont | 5-NN | Global | Visible share |
|---|---|---|---|---|---|---|---|
| MNIST, before (one radius) | 4 / 10 / 19 | 0.60 / 0.13 | 0.919 | 0.928 | 0.845 | 0.478 | 33% |
| MNIST, after (radius per cell) | 10 / 12 / 15 | 0.39 / 0.39 | 0.918 | 0.940 | 0.831 | 0.328 | 73% |
| half sphere, before | 6 / 10 / 13 | – | 0.996 | 0.998 | – | 0.959 | 94% |
| half sphere, after | 10 / 11 / 12 | – | 0.996 | 0.998 | – | 0.943 | 91% |

On MNIST every digit gets its share of the picture and continuity rises. The price is global distance correlation (0.48 → 0.33): evening out the scale also removes real differences in spread between digits. UMAP does the same and has 0.30 here. On the half sphere, whose density is uniform, little changes.

**Files.**
- `settings_before.yaml` / `settings_after.yaml`: pinned settings, differing only in `graph.radius.strategy`;
- `one.py`: one case (dataset × variant);
- `run.py`: runs all four in guarded processes and draws `figure.png`.
