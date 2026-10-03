"""Local vs global quality: every tuned configuration of every method (tuning split).

Each point is one configuration from tuning/<family>/results.csv (mean over
seeds, bars +-1 std); the line joins a method's Pareto front (configurations no
other of its configurations beats on both axes); the star is the configuration
the protocol selected (highest trustworthiness, tuning/<family>/best.yaml).
No new runs: it only reads the tuning results.

    python benchmarks/mnist/tradeoff/plot.py      # figure.png
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from common.plotting import COLORS  # noqa: E402
from common.results import read_rows  # noqa: E402

FAMILIES = {"lvm": "LVM", "umap": "UMAP", "tsne": "t-SNE", "le": "Laplacian Eigenmaps", "lisomap": "Landmark Isomap"}
LOCAL = [("trustworthiness", "trustworthiness (k=10)"), ("continuity", "continuity (k=10)"),
         ("knn_accuracy", "5-NN label accuracy")]
GLOBAL = ("distance_correlation", "global: distance rank correlation")


def label(family: str, p: dict) -> str:
    if family == "lvm":
        return f"{p['cells']['n_cells']}"
    if family == "umap":
        return f"k{p['n_neighbors']} d{p['min_dist']:g}"
    if family == "tsne":
        return f"p{p['perplexity']:g} x{p['late_exaggeration']:g}"
    if family == "le":
        return f"k{p['n_neighbors']}"
    return f"L{p['n_landmarks']} k{p['n_neighbors']}"


def configurations(family: str) -> list[dict]:
    rows = [r for r in read_rows(BENCH / "mnist" / "tuning" / family / "results.csv") if r["status"] == "ok"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["params"]].append(r)
    out = []
    for params, rs in groups.items():
        stats = {k: (statistics.mean(float(r[k]) for r in rs), statistics.pstdev(float(r[k]) for r in rs))
                 for k in [key for key, _ in LOCAL] + [GLOBAL[0]]}
        out.append({"params": json.loads(params), "stats": stats})
    return out


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points not beaten on both coordinates by another point, sorted by x."""
    front = [p for p in points if not any(q[0] >= p[0] and q[1] >= p[1] and q != p for q in points)]
    return sorted(front)


def main() -> None:
    fig, axes = plt.subplots(1, len(LOCAL), figsize=(6.0 * len(LOCAL), 5.2))
    for family, name in FAMILIES.items():
        confs = configurations(family)
        best = yaml.safe_load((BENCH / "mnist" / "tuning" / family / "best.yaml").read_text())["params"]
        color = COLORS[family]
        for ax, (key, _) in zip(axes, LOCAL):
            xs = [c["stats"][key] for c in confs]
            ys = [c["stats"][GLOBAL[0]] for c in confs]
            ax.errorbar([x[0] for x in xs], [y[0] for y in ys], xerr=[x[1] for x in xs], yerr=[y[1] for y in ys],
                        fmt="o", ms=3.5, color=color, alpha=0.55, elinewidth=0.6, label=name)
            front = pareto([(x[0], y[0]) for x, y in zip(xs, ys)])
            ax.plot([p[0] for p in front], [p[1] for p in front], "-", color=color, lw=1.4)
            for c, x, y in zip(confs, xs, ys):
                ax.annotate(label(family, c["params"]), (x[0], y[0]), fontsize=5, color=color,
                            xytext=(3, 2), textcoords="offset points")
                if c["params"] == best:
                    ax.plot(x[0], y[0], "*", ms=13, color=color, mec="black", mew=0.6, zorder=5)
    for ax, (_, xlabel) in zip(axes, LOCAL):
        ax.set_xlabel(xlabel + "  (local, better →)")
        ax.set_ylabel(GLOBAL[1] + "  (better ↑)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Local vs global structure across each method's tuning grid  (MNIST tuning split, 20k, "
                 "mean ± 1 std over 3 seeds)\nline: each method's Pareto front;  ★: configuration selected by "
                 "trustworthiness;  labels: LVM n_cells, UMAP k/min_dist, t-SNE perplexity/late exaggeration, "
                 "LE k, Landmark Isomap L/k", fontsize=9)
    fig.tight_layout()
    out = HERE / "figure.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
