import json
from pathlib import Path

from experiments.common import style
from experiments.sfv.mnist8m.stability.plot import LABELS

HERE = Path(__file__).resolve().parent
PLACING = ("lvm", "lvm_chart", "umap", "landmark_isomap")


def main() -> None:
    sizes = json.loads((HERE / "model_size.json").read_text())["bytes"]
    fig, ax = style.figure("half", aspect=0.6)
    for y, method in enumerate(PLACING):
        ax.barh(y, sizes[method] / 2**20, color=style.method_colour(method))
    ax.axvline(sizes["data"] / 2**20, color="0.4", linestyle="--", linewidth=0.7, label="the 50k fitted points")
    ax.set_xscale("log")
    ax.set_yticks(range(len(PLACING)), [LABELS[m] for m in PLACING])
    ax.invert_yaxis()
    ax.set_xlabel("model size (MB)")
    ax.legend(loc="upper center", fontsize="small")
    style.save_figure(fig, HERE / "model_size")


if __name__ == "__main__":
    main()
