# Sphere tears

**Problem.** On the half sphere, LVM tears the surface along seams:
- with 3 landmarks (d̂ + 1 for d̂ ≈ 2), the rim is thrown outward into separate arms, and 6% of points have a sphere neighbour placed more than 10% of the embedding's width away;
- more landmarks reduce the tearing, but only by diluting it (see the table).

**Cause.**
- Every torn point sits on a reach boundary, where the set of landmarks whose map reaches a point (v ≥ τ) changes from one neighbour to the next. With 3 and 6 landmarks, 100% of torn points are next to such a boundary (`reach_boundary_check.py`).
- On the reached side, the distance to a landmark is measured (−log v ≈ −log τ near the edge of its map). On the other side, it is chained through another landmark, which gives a different value. The distance jumps across a line on the sphere, and Landmark MDS turns that jump into a gap.
- More landmarks only dilute the jump, since each distance is one of L inputs to the triangulation.

**Fix.** Pending: treat the cause, a smooth hand-over between the measured and the chained distance near the edge of each map, so distances are continuous across the sphere. When it exists, its results go next to these.

| Landmarks | Torn points | Continuity | Global (= sphere-distance) corr. |
|---|---|---|---|
| 3 | 6.3% | 0.992 | 0.883 |
| 6 | 3.9% | 0.995 | 0.949 |
| 9 | 1.3% | 0.996 | 0.972 |
| 12 | 0.6% | 0.996 | 0.975 |
| 20 | 0.0% | 0.996 | 0.978 |

**Files.**
- `settings.yaml`: the pinned "before" settings (MNIST-tuned LVM of 2026-10-03: 300 cells, 3-nearest-cells point step, chained distances, reach scaling). Only the landmark count varies.
- `sweep_one.py --landmarks L`: one run on the pinned 20k points (seed 0).
- `plot_sweep.py`: `figure.png`, showing the embeddings in 2-D and the torn points and landmarks on the sphere in 3-D.
- `reach_boundary_check.py`: the evidence for the cause.
- `before_reorg/`: the same sweep from before the reorganisation; it agrees within GPU run-to-run variation.
