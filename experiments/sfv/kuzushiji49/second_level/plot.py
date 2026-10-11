import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
METHODS = {"level1": ("LVM level 1", "lvm", 0.45),
           "same": ("LVM level 2, same data", "lvm", 0.75), "same_chart": ("  + local chart", "lvm_chart", 0.75),
           "new": ("LVM level 2, new data", "lvm", 1.0), "new_chart": ("  + local chart", "lvm_chart", 1.0),
           "umap_same": ("UMAP per region, same data", "umap", 0.75), "umap_new": ("UMAP per region, new data", "umap", 1.0),
           "tsne_same": ("t-SNE per region, same data", "tsne", 0.75), "tsne_new": ("t-SNE per region, new data", "tsne", 1.0),
           "umap_global": ("UMAP global (50k)", "umap", 0.45), "tsne_global": ("t-SNE global (50k)", "tsne", 0.45)}
PANELS = {"knn_accuracy": "kNN accuracy", "trustworthiness": "trustworthiness", "continuity": "continuity",
          "density_preservation": "density preservation", "seconds": "total time (s)"}


def weighted(rows, value) -> list[float]:
    """Per seed: the mean over regions, weighted by the region's points."""
    per_seed = {}
    for r in rows:
        if "quality" in r and value(r) is not None:
            per_seed.setdefault(r["seed"], []).append((r["n_fit"], value(r)))
    return [sum(n * v for n, v in pairs) / sum(n for n, _ in pairs) for pairs in per_seed.values()]


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    rows, timings = results["rows"], results["timings"]
    fig, axes = style.figure("full", aspect=0.5, ncols=len(PANELS), sharey=True)
    for letter, ax, (metric, label) in zip("abcde", axes, PANELS.items()):
        for y, (method, (name, colour, alpha)) in enumerate(METHODS.items()):
            if metric == "seconds":
                values = [t["seconds"][method] for t in timings]
            else:
                values = weighted(rows, lambda r: r["quality"][method][metric])
            if values:
                ax.barh(y, np.mean(values), xerr=np.std(values), color=style.method_colour(colour), alpha=alpha,
                        error_kw={"linewidth": 0.5, "capsize": 1.0})
        ax.set_xlabel(label)
        ax.set_title(f"({letter})", loc="left")
        if metric == "seconds":
            ax.set_xscale("log")
    axes[0].set_yticks(range(len(METHODS)), [name for name, _, _ in METHODS.values()])
    axes[0].invert_yaxis()
    style.save_figure(fig, HERE / "second_level")


if __name__ == "__main__":
    main()
