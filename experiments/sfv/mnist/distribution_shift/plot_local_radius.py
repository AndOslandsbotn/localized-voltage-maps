from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

from experiments.common import style
from experiments.sfv.mnist.distribution_shift.run import SEEDS, shifted

HERE = Path(__file__).resolve().parent
SHOWN_SEED, K = 0, 30
EMBEDDINGS = {"uniform": "uniform masses", "shifted": "masses counted on the data"}


def relative_radius(points: np.ndarray) -> np.ndarray:
    radius = NearestNeighbors(n_neighbors=K).fit(points).kneighbors()[0].mean(axis=1)
    return radius / np.median(radius)


def per_digit(radius: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.array([np.median(radius[labels == d]) for d in range(10)])


def main() -> None:
    digits = {"data": [], **{name: [] for name in EMBEDDINGS}}
    for seed in SEEDS:
        data, labels = shifted(seed)
        digits["data"].append(per_digit(relative_radius(data), labels))
        for name in EMBEDDINGS:
            Z = np.load(HERE / "embeddings" / f"lvm_chart_{name}_seed{seed}.npz")["Z"]
            digits[name].append(per_digit(relative_radius(Z), labels))
        if seed == SHOWN_SEED:
            shown_labels = labels

    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.32, ncols=3)
    for letter, ax, (name, title) in zip("ab", axes, EMBEDDINGS.items()):
        Z = np.load(HERE / "embeddings" / f"lvm_chart_{name}_seed{SHOWN_SEED}.npz")["Z"]
        ax.scatter(Z[:, 0], Z[:, 1], c=shown_labels, cmap=colours, vmin=-0.5, vmax=9.5, s=1.35, linewidths=0,
                   rasterized=True)
        ax.set_title(f"({letter}) {title}", loc="left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()

    ax = axes[2]
    markers = {"data": ("k", "s", "data"), "uniform": (style.method_colour("lvm_chart"), "o", "uniform masses"),
               "shifted": (style.method_colour("lvm"), "D", "masses counted on the data")}
    for i, (name, (colour, marker, label)) in enumerate(markers.items()):
        values = np.array(digits[name])
        ax.errorbar(np.arange(10) + 0.2 * (i - 1), values.mean(axis=0), yerr=values.std(axis=0), color=colour,
                    marker=marker, linestyle="", markersize=2.5, elinewidth=0.5, capsize=1.0, label=label)
    ax.set_yscale("log")
    ax.set_xticks(range(10), [str(d) for d in range(10)])
    ax.set_xlabel("digit (1 and 7 are 10x more common)")
    ax.set_ylabel("local radius / median")
    ax.set_title("(c) spread of each digit", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    digits_legend = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits_legend, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "local_radius")


if __name__ == "__main__":
    main()
