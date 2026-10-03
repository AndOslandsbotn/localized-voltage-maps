# Half sphere

Points uniform on the upper half of the unit sphere in 3-D (z ≥ 0), generated (`common/datasets.py`).
- It can be flattened into a disc, so a good 2-D embedding keeps every neighbourhood intact and changes colour smoothly: by height (rings) and by angle around the pole (a colour wheel).
- The global distance correlation already measures distance along the sphere: straight-line distance grows monotonically with it, so the two give the same rank order.
- Every method uses its MNIST-tuned settings unchanged.

| Folder | What |
|---|---|
| `gallery/` | LVM, LVM + local PCA and t-SNE on 20k points, next to the truth (`run.py`) |
| `diagnostics/landmarks/` | Landmark count and selection (mutual information vs max-min) on the half sphere. Its results were made with the pre-reorganisation scripts; see git history. |
| `archive/` | superseded results |

The tears seen with few landmarks are a demonstration: `../demonstrations/sphere_tears/`.
