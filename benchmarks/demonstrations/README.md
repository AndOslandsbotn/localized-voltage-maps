# Demonstrations

One folder per problem we found in LVM. Each shows the problem, its cause and its fix, before and after.

| Folder | Problem | Status |
|---|---|---|
| `point_stacking/` | Points pile onto a few spots (MNIST) | Fixed (k-nearest-cells point step), plus an optional local PCA chart |
| `sphere_tears/` | Seams tear the half sphere apart when there are few landmarks | Fixed (fix A: distances read below τ, down to 10⁻¹⁰) |

**Rules:**
- Every case loads its complete, pinned settings (`settings*.yaml`, resolved from the defaults of the time), never "the current defaults". So it reproduces after the defaults change, and a fix can be shown next to the original problem.
- GPU runs are not bit-for-bit deterministic: floating-point sums in landmark selection can break near-ties differently. Reruns agree within about ±0.5 percentage points on the tear and stacking shares.
