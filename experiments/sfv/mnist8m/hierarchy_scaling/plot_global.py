import json
from pathlib import Path

from experiments.common import style
from experiments.sfv.mnist8m.hierarchy_scaling.plot_style import FEW_SIZE_LABELS, QUALITY, draw

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["global"]
    fig, axes = style.figure("full", aspect=0.3, ncols=len(QUALITY))
    for letter, ax, (metric, label) in zip("abcde", axes, QUALITY.items()):
        draw(ax, rows, ("umap", "tsne", "lvm_level1", "lvm_level1_chart"), metric, FEW_SIZE_LABELS)
        ax.set_ylabel(label)
        ax.set_title(f"({letter})", loc="left")
    axes[-1].axhline(1.0, color="0.6", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4)
    style.save_figure(fig, HERE / "global")


if __name__ == "__main__":
    main()
