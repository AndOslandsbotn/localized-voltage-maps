import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.sfv.kuzushiji49.streaming.run_lvm import SIZES

HERE = Path(__file__).resolve().parent
SIZE_LABELS = ("5k", "20k", "50k", "100k", "", "", "271k")
METHODS = {"lvm": ("LVM, k-means on all points in memory", "lvm", "-"),
           "lvm_minibatch": ("LVM streamed, one mini-batch pass", "lvm", ":"),
           "lvm_lloyd": ("LVM streamed, Lloyd passes", "lvm", "--"),
           "umap": ("UMAP", "umap", "-"), "tsne": ("t-SNE", "tsne", "-")}
PANELS = {"knn_accuracy": "kNN accuracy", "trustworthiness": "trustworthiness", "seconds": "time (s)"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS))
    for letter, ax, (metric, label) in zip("abc", axes, PANELS.items()):
        for method, (name, colour, linestyle) in METHODS.items():
            by_n = {}
            for r in rows:
                if r["method"] == method:
                    by_n.setdefault(r["n"], []).append(r[metric])
            n = sorted(by_n)
            mean = np.array([np.mean(by_n[k]) for k in n])
            spread = np.array([np.std(by_n[k]) for k in n])
            colour = style.method_colour(colour)
            ax.fill_between(n, mean - spread, mean + spread, color=colour, alpha=0.2, linewidth=0)
            ax.plot(n, mean, color=colour, linestyle=linestyle, linewidth=0.8, marker="o", markersize=1.6, label=name)
        ax.set_xscale("log")
        ax.set_xticks(SIZES, SIZE_LABELS, rotation=45)
        ax.minorticks_off()
        ax.set_xlabel("n (Kuzushiji-49)")
        ax.set_ylabel(label)
        ax.set_title(f"({letter})", loc="left")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=3)
    style.save_figure(fig, HERE / "streaming")


if __name__ == "__main__":
    main()
