# MNIST archive

Superseded results, kept for reference. Each folder is named after what changed since.

| Folder | Superseded by |
|---|---|
| `results/`, `legacy_scripts/` | the tuning/evaluation split structure (2026-10-01). These results used all 70k images and ARPACK for Laplacian Eigenmaps |
| `tuning_lvm_log_mds/`, `comparisons_log_mds_lvm/` | Landmark MDS with chained distances (previously log-MDS) |
| `tuning_lvm_coverage/`, `comparisons_coverage_lvm/` | `reach` scaling (previously `coverage`, overlap 8) |
| `comparisons_reach_reselect/` | `reach` by alternation (previously re-selecting landmarks at every probe) |
| `tuning_lvm_reach_reselect/`, `comparisons_dimension_cpu/` | GPU dimension estimate and the new `reach` search |
| `tuning_umap_k5-50/`, `comparisons_umap_k5-50/` | UMAP grid widened to n_neighbors 200 |
| `tuning_lvm_radial_points/`, `comparisons_radial_points/` | k-nearest-cells point step (previously the hard kernel) |
| `comparisons_lvm_pca_numpy/` | local PCA on the GPU (previously NumPy) |
| `gallery_before_reorg/` | the shared gallery tool (`common/gallery.py`) |
| `tuning_lvm_fixed_radius/` | LVM tuning with one global kernel radius (`graph.radius: knn`), the default until 2026-10-03; superseded by `graph.radius: adaptive_per_cell` (see `demonstrations/uneven_connectivity/`). Only LVM rows moved; the other methods don't use the radius and were kept (also the trade-off figure of that time) |
| `comparisons_fixed_radius_lvm/` | LVM rows of the GPU and CPU comparisons with one global kernel radius (`graph.radius: knn`), the default until 2026-10-03; superseded by `graph.radius: adaptive_per_cell` (see `demonstrations/uneven_connectivity/`). Only LVM rows moved; the other methods don't use the radius and were kept |
| `gallery_fixed_radius/` | LVM panels of the gallery with one global kernel radius (`graph.radius: knn`), the default until 2026-10-03; superseded by `graph.radius: adaptive_per_cell` (see `demonstrations/uneven_connectivity/`). Only LVM rows moved; the other methods don't use the radius and were kept |
