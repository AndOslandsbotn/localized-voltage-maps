# Sphere tears

**Status: fixed (fix A).** The "before" case stays pinned for comparison.

**Problem.** On the half sphere, LVM tears the surface along seams:
- with 3 landmarks (d̂ + 1 for d̂ ≈ 2), the rim is thrown outward into separate arms, and 6% of points have a sphere neighbour placed more than 10% of the embedding's width away;
- more landmarks reduce the tearing, but only by diluting it (see the table).

**Cause.**
- Every torn point sits on a reach boundary, where the set of landmarks whose map reaches a point (v ≥ τ) changes from one neighbour to the next. With 3 and 6 landmarks, 100% of torn points are next to such a boundary (`reach_boundary_check.py`).
- On the reached side, the distance to a landmark is measured (−log v ≈ −log τ near the edge of its map). On the other side, it is chained through another landmark, which gives a different value. The distance jumps across a line on the sphere, and Landmark MDS turns that jump into a gap.
- More landmarks only dilute the jump, since each distance is one of L inputs to the triangulation.

**Characterisation** (`characterise.py --landmarks L [--missing clip]`, results in `characterise_L*_*.json`):

| | 3 landmarks | 6 landmarks |
|---|---|---|
| torn points | 6.1% | 3.7% |
| jump in d(., l) across the edge of l's map (median / 90%) | 1.67 / 2.04 | 1.37 / 1.71 |
| usual difference between sphere neighbours (median / 90%) | 0.0002 / 0.18 | 0.0002 / 0.16 |
| chained distance / the line -log v = a·geodesic + b fitted where measured (median) | 1.15 | 1.11 |
| torn among points next to a map edge / away from every edge | 99.8% / 0% | 62% / 0% |
| torn among points reached by every landmark | 3.6% | 1.8% |
| control, missing = clip (continuous at the edge) | 0.5% torn | 0.1% torn |

- The trigger is "next to the edge of some landmark's map", not "reached by too few landmarks". Points reached by every landmark tear too when a neighbour falls outside one map.
- The jump is the chained route's detour bias. x → j → l is never shorter than the direct distance, and with few landmarks j is rarely on the way. Measured distances follow the true sphere distance closely (R² 0.97).
- Landmark MDS works on squared distances, so the jump of one coordinate, from 6.9 to about 8.6, is enough to pull a point away from its neighbours.
- Clip removes the tears but destroys global structure. It is the confirming control, not a fix.

**Fix A: read distances below τ** (`settings_fix_A.yaml`: `settings.yaml` plus `embedding.distance_floor: 1e-10`).
- Distances d = −log v are read from the voltage down to a numerical floor ε = 10⁻¹⁰ instead of τ = 10⁻³. Only below ε is a distance chained.
- τ keeps its other roles: support and locality, `reach` choosing ρ_g, and landmark selection.
- The theory supports this: Theorem 12 bounds the voltage's decay between two exponentials everywhere, not only inside the support. τ is where a map is no longer computed or stored for efficiency, not where −log v stops measuring distance.

| Landmarks | Torn, before → fix A | Global (= sphere-distance) corr., before → fix A | Pairs chained with fix A |
|---|---|---|---|
| 3 | 6.3% → 0.2% | 0.883 → 0.962 | 0 |
| 6 | 3.9% → 0.03% | 0.949 → 0.972 | 0 |
| 9 | 1.3% → 0.03% | 0.972 → 0.976 | 0 |
| 12 | 0.6% → 0.03% | 0.975 → 0.975 | 0 |
| 20 | 0.0% → 0.03% | 0.978 → 0.977 | 0 |

- The rim stays attached even with 3 landmarks, and nothing needs chaining: every voltage on the half sphere is above 10⁻¹⁰.
- The remaining few tears (0.2% with 3 landmarks, 0–0.03% with more) are a different, smaller effect: the sharp point step.
  - With sharpness 16, a point takes essentially its nearest cell's voltages, so crossing the boundary between two cells switches almost abruptly from one cell's voltages to the other's.
  - That is invisible between typical neighbouring cells. But where two neighbouring centroids lie far apart along a landmark's gradient (e.g. cells 105 and 162: 0.21 apart, against a typical 0.13), it is a visible jump (about 0.9 in distance, against a typical 0.3 between neighbouring cells).
  - The torn points are not in landmark cells and not at the pole (z ≈ 0.7). All cell pairs holding neighbouring points are joined by an edge, so the graph is not the cause.
  - Lower sharpness removes them. With 3 landmarks: sharpness 1 gives 0% torn, continuity 0.9985, global 0.970; sharpness 16 gives 0.2%, 0.9953, 0.963.
  - The default of 16 was chosen on MNIST, where flatter weights mix clusters. Which sharpness generalises is an open question.
- On MNIST (`../../mnist/diagnostics/distance_floor/`), fix A is neutral locally and slightly better globally (0.502 → 0.514). Only 1.3% of point-landmark pairs were chained there before.
- Limitation: in the hierarchy (paper 2), maps are computed only on their support, so values below τ won't exist and chaining returns between regions. There it needs a correction of the detour bias.

`sweep_one.py --variant fix_A`, `plot_sweep.py fix_A` → `figure_fix_A.png`.

**Before (pinned):**

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
