import numpy as np

from experiments.common import style

SIZES = (5_000, 10_000, 20_000, 50_000, 100_000, 150_000, 200_000, 270_912)
SIZE_LABELS = {5_000: "5k", 10_000: "10k", 20_000: "20k", 50_000: "50k", 100_000: "100k", 270_912: "271k"}
FEW_SIZE_LABELS = {5_000: "5k", 50_000: "50k", 270_912: "271k"}
METHODS = {"umap": ("UMAP", "umap", "-"), "tsne": ("t-SNE", "tsne", "-"),
           "lvm_level1": ("LVM level 1", "lvm_chart", "-"), "lvm_level1_chart": ("LVM level 1 + chart", "lvm_chart", ":"),
           "lvm_level2": ("LVM level 2", "lvm", "-"), "lvm_level2_chart": ("LVM level 2 + chart", "lvm", ":"),
           "lvm_level2_chart_parallel": ("LVM level 2 + chart, parallel", "lvm", "--")}
QUALITY = {"knn_accuracy": "kNN accuracy", "trustworthiness": "trustworthiness", "continuity": "continuity",
           "density_preservation": "density preservation", "density_slope": "density slope"}


def draw(ax, rows, methods, metric, labels=SIZE_LABELS) -> None:
    for method in methods:
        name, colour, linestyle = METHODS[method]
        by_n = {}
        for r in rows:
            if r["method"] == method and metric in r and np.isfinite(r[metric]):
                by_n.setdefault(r["n"], []).append(r[metric])
        n = sorted(by_n)
        if n:
            mean = np.array([np.mean(by_n[k]) for k in n])
            spread = np.array([np.std(by_n[k]) for k in n])
            colour = style.method_colour(colour)
            ax.fill_between(n, mean - spread, mean + spread, color=colour, alpha=0.2, linewidth=0)
            ax.plot(n, mean, color=colour, linestyle=linestyle, linewidth=0.8, marker="o", markersize=1.6, label=name)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [labels.get(n, "") for n in SIZES])
    ax.minorticks_off()
    ax.set_xlabel("n")
