"""One global radius (knn) vs a radius per cell (adaptive_per_cell), on MNIST and MNIST8M's 50k sample, 3 seeds;
each run its own memory-guarded process. Prints metrics (charted and plain) and per-digit edges and room; draws
figure.png (seed-0 pictures, LVM + local PCA chart).

    python experiments/sfv/mnist8m/diagnostics/adaptive_radius/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.runner import thread_env  # noqa: E402

DATASETS = {"mnist": "MNIST evaluation split (50k)", "mnist8m": "MNIST8M sample (50k)"}
RADII = {"knn": "one global radius (knn)", "adaptive_per_cell": "radius per cell (adaptive_per_cell)"}
SEEDS = range(3)
METRICS = ("trustworthiness", "continuity", "knn_accuracy", "distance_correlation")


def plot(rows: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(len(DATASETS), len(RADII), figsize=(9 * len(RADII), 8 * len(DATASETS)))
    for i, (dataset, dname) in enumerate(DATASETS.items()):
        for j, (radius, rname) in enumerate(RADII.items()):
            ax, f = axes[i, j], HERE / "embeddings" / f"{dataset}_{radius}.npz"
            r = next((r for r in rows if r["dataset"] == dataset and r["radius"] == radius and r["seed"] == 0), None)
            if r is None or not f.exists():
                ax.axis("off")
                continue
            e = np.load(f)
            order = np.random.default_rng(0).permutation(len(e["y"]))
            ax.scatter(e["Z"][order, 0], e["Z"][order, 1], c=e["y"][order], cmap="tab10", vmin=-0.5, vmax=9.5,
                       s=0.3, rasterized=True)
            c = r["charted"]
            ax.set_title(f"{dname}, {rname}\ntrust {c['trustworthiness']:.3f}  cont {c['continuity']:.3f}  "
                         f"5-NN {c['knn_accuracy']:.3f}  global {c['distance_correlation']:.3f}  "
                         f"visible {100 * c['visible_share']:.0f}%\nroom of the 0s {r['room']['0']:.2f}, "
                         f"of the 1s {r['room']['1']:.2f} (1 = their pixel-space share)", fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
    cmap = plt.get_cmap("tab10")
    fig.legend([Line2D([], [], ls="", marker="o", ms=8, color=cmap((d + 0.5) / 10)) for d in range(10)],
               [str(d) for d in range(10)], title="digit", loc="center right", frameon=False)
    fig.suptitle("LVM + local PCA chart (seed 0): one global kernel radius vs a radius per cell", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.95, 0.97))
    fig.savefig(HERE / "figure.png", dpi=90)
    plt.close(fig)
    print(HERE / "figure.png")


def main() -> None:
    path = HERE / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    for dataset in DATASETS:
        for radius in RADII:
            for seed in SEEDS:
                if any(r["dataset"] == dataset and r["radius"] == radius and r["seed"] == seed for r in rows):
                    continue
                cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
                       str(HERE / "one.py"), "--dataset", dataset, "--radius", radius, "--seed", str(seed)]
                out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
                lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
                if not lines:
                    print(f"{dataset} {radius} seed {seed}: failed\n{out.stderr[-800:]}", flush=True)
                    continue
                rows.append(json.loads(lines[-1][7:]))
                with open(path, "a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
    for dataset in DATASETS:
        print(f"\n== {DATASETS[dataset]} (mean over seeds)")
        for radius in RADII:
            rs = [r for r in rows if r["dataset"] == dataset and r["radius"] == radius]
            if not rs:
                continue
            for kind in ("plain", "charted"):
                print(f"  {radius:18s} {kind:8s} " + "  ".join(
                    f"{m[:5]} {np.mean([r[kind][m] for r in rs]):.3f}" for m in METRICS))
            print(f"  {'':18s} edges    " + " ".join(f"{d}:{np.mean([r['edges'][str(d)] for r in rs]):5.1f}"
                                                   for d in range(10)))
            print(f"  {'':18s} room     " + " ".join(f"{d}:{np.mean([r['room'][str(d)] for r in rs]):5.2f}"
                                                   for d in range(10)))
    plot(rows)


if __name__ == "__main__":
    main()
