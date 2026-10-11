import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
LABELS = {"lvm": "LVM", "lvm_chart": "LVM + local chart", "umap": "UMAP", "landmark_isomap": "Landmark Isomap"}
QUALITY = {"trustworthiness": "trustworthiness", "continuity": "continuity", "knn_accuracy": "kNN accuracy",
          "density_preservation": "density preservation"}


def mean_and_spread(rows, method, key):
    values = [row[key] for row in rows if row["method"] == method]
    return np.mean(values), np.std(values)


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    methods = list(LABELS)
    colours = [style.method_colour(method) for method in methods]
    fig, axes = style.figure("full", aspect=0.3, ncols=1 + len(QUALITY), sharey=True)
    transform = [mean_and_spread(rows, method, "transform_seconds") for method in methods]
    axes[0].barh(list(LABELS.values()), [t for t, _ in transform], xerr=[s for _, s in transform], color=colours,
                 error_kw={"linewidth": 0.5, "capsize": 1.0})
    for y, (t, _) in enumerate(transform):
        axes[0].text(t + 25, y, f"{t:.0f} s", va="center")
    axes[0].set_xlim(0, 1.35 * max(t for t, _ in transform))
    axes[0].set_xlabel("time to place (s)")
    axes[0].set_title("(a)", loc="left")
    for letter, ax, (key, label) in zip("bcdef", axes[1:], QUALITY.items()):
        values = [mean_and_spread(rows, method, key) for method in methods]
        ax.barh(list(LABELS.values()), [v for v, _ in values], xerr=[s for _, s in values], color=colours,
                error_kw={"linewidth": 0.5, "capsize": 1.0})
        ax.set_xlim(-0.5 if key == "density_preservation" else 0.0, 1.0)
        ax.set_xlabel(label)
        ax.set_title(f"({letter})", loc="left")
    axes[0].invert_yaxis()
    style.save_figure(fig, HERE / "transform")


if __name__ == "__main__":
    main()
