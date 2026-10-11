import json
from pathlib import Path

from experiments.common import style
from experiments.sfv.kuzushiji49.hierarchy_scaling.plot_style import QUALITY, draw

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["time"]
    fig, ax = style.figure("half", aspect=0.8)
    draw(ax, rows, ("umap", "tsne", "lvm_level1", "lvm_level1_chart", "lvm_level2_chart", "lvm_level2_chart_parallel"),
         "seconds")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("total time from disk (s)")
    ax.legend(loc="upper left", fontsize="x-small")
    style.save_figure(fig, HERE / "time")


if __name__ == "__main__":
    main()
