# MNIST8M archive

| Folder | Superseded by |
|---|---|
| `scaling_unconnected_graph/` | `graph.connect` (default since 2026-10-03). These LVM rows were made without it: an isolated cell became a landmark, so ρ_g ran to its bound and the embedding collapsed (see `../../demonstrations/isolated_landmark/`). The other methods' rows are unaffected and were kept in `scaling/` |
| `scaling_float64_sample/` | LVM's k-means sample in float32 without the extra copy (2026-10-03). The LVM rows here kept it in float64 and copied it once more, so fitting LVM on all n (part A) ran out of memory already at 200k. The other methods' rows are unaffected and were kept in `scaling/` |
| `scaling_sample_kmeans/` | `cells.refine: stream` (2026-10-03). These LVM part-A rows fitted on all n by making k-means' in-memory sample n points, so memory grew with n and 500k ran out. Part A now keeps the 50k sample and refines its centroids by streaming mini-batch k-means over all n points (memory independent of n). The other rows were kept in `scaling/` |
| `scaling_float64_stream/` | float32 chunks from the MNIST8M stream (2026-10-03). These LVM rows read the stream as float64 and LVM converted every chunk back to float32: that conversion, not the disk, took most of the streaming time (8.1M points: 26–31 s of 37–44 s; with float32 chunks, 7 s of 14 s). The other methods already got float32 chunks; their rows were kept in `scaling/` |
| `scaling_fixed_radius/` | LVM rows of the scaling runs with one global kernel radius (`graph.radius: knn`), the default until 2026-10-03; superseded by `graph.radius: adaptive_per_cell` (see `demonstrations/uneven_connectivity/`). Only LVM rows moved; the other methods don't use the radius and were kept |
