# Demonstrations

One folder per problem we found in LVM. Each shows the problem, its cause and its fix, before and after.

| Folder | Problem | Status |
|---|---|---|
| `point_stacking/` | Points pile onto a few spots (MNIST) | Fixed (k-nearest-cells point step), plus an optional local PCA chart |
| `sphere_tears/` | Seams tear the half sphere apart when there are few landmarks | Fixed (fix A: distances read below τ, down to 10⁻¹⁰) |
| `isolated_landmark/` | An isolated cell chosen as a landmark makes `reach` impossible; ρ_g runs to its bound and the embedding collapses (MNIST8M) | Fixed (`graph.connect`: the cell graph is made connected) |
| `uneven_connectivity/` | One global kernel radius under-connects spread-out digits and over-connects compact ones; voltage detours along thin chains, so one digit takes the frame (MNIST, MNIST8M); the half sphere is the control | Fixed (`graph.radius: adaptive_per_cell`) |

**Rules:**
- Every case loads its complete, pinned settings (`settings*.yaml`, resolved from the defaults of the time), never "the current defaults". So it reproduces after the defaults change, and a fix can be shown next to the original problem.
- GPU runs are not bit-for-bit deterministic: floating-point sums in landmark selection can break near-ties differently. Reruns agree within about ±0.5 percentage points on the tear and stacking shares.
