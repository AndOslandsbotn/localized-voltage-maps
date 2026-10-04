# Isolated landmark

**Status: fixed** (`graph.connect: true`, the default). The "before" case stays pinned for comparison.

**Problem.** On MNIST8M, LVM collapsed almost all points onto a tiny central pile; only the 0s spread out, in rays. The same happens at 50k and at 8.1M points, because LVM fits the same model at every size (on the first 50k points of the shuffled stream).

**Cause.**
- The cell graph had 5 pieces: the main one (296 cells) and 4 isolated cells, each with no cell within r. One of the 15 landmarks was an isolated cell.
- Mutual information favours such a cell, because its map (1 at itself, 0 elsewhere) identifies its cell perfectly. But its map reaches nothing.
- `reach` with k = all asks that a typical point be reached by all 15 landmarks, which is then impossible: at best 14. So the ρ_g search ran to its lower bound, 10⁻⁶.
- With practically no ground every map is nearly flat (v ≈ 1 everywhere), so −log v ≈ 0 for nearly every point, and the embedding collapses.
- MNIST also has 2 isolated cells at 300 cells; by chance none was chosen as a landmark.

**Fix.** `graph.connect` joins every disconnected piece of the cell graph to the main piece by its shortest link between centroids (weight 1, like an ordinary edge of the hard kernel). This is the standard remedy for neighbour graphs.

| MNIST8M, first 50k, seed 0 | Graph pieces | Landmarks in a size-1 piece | ρ_g | Reach | Trust | Cont | 5-NN | Global | Visible share |
|---|---|---|---|---|---|---|---|---|---|
| before | 5 | 1 | 1.0e-06 (the bound) | 14 / 15 | 0.953 | 0.822 | 0.822 | 0.358 | 1% |
| after | 1 | 0 | 6.4e-04 | 15 / 15 | 0.960 | 0.877 | 0.832 | 0.428 | 4% |

What is left in the after picture (the 0s and 6s spread out, the other digits pushed together) is the global-scaling crowding that plain LVM shows on MNIST too.

**Files.**
- `settings_before.yaml` / `settings_after.yaml`: pinned settings, differing only in `graph.connect`;
- `one.py`: one variant;
- `run.py`: runs both variants in guarded processes and draws `figure.png`.
