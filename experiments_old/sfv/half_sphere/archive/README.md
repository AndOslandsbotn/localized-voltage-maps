# Half-sphere archive

| Folder | Superseded by |
|---|---|
| `gallery_before_reorg/` | the shared gallery tool (`common/gallery.py`). Its points came from a different random stream, generator seed 0 rather than the eval stream |
| `gallery_fixed_radius/` | LVM panels of the 2-D and 3-D galleries with one global kernel radius (`graph.radius: knn`), the default until 2026-10-03; superseded by `graph.radius: adaptive_per_cell` (see `demonstrations/uneven_connectivity/`). Only LVM rows moved; the other methods don't use the radius and were kept |
