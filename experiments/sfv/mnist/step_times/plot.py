import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent


def main() -> None:
    results = json.loads((HERE / "step_times.json").read_text())
    steps = list(results["seconds"])
    milliseconds = np.array([results["seconds"][step] for step in steps]) * 1000      # (steps, seeds)
    labels = [f"{step} (off)" if step in results["off"] else step for step in steps]

    fig, ax = style.figure("half", aspect=0.8)
    ax.barh(labels, milliseconds.mean(axis=1), xerr=milliseconds.std(axis=1), color=style.method_colour("lvm"),
            error_kw={"linewidth": 0.6, "capsize": 1.5})
    ax.invert_yaxis()                                                              # pipeline order, top to bottom
    ax.set_xlabel("time (ms)")
    total = milliseconds.sum(axis=0)
    ax.text(0.98, 0.98, f"total {total.mean():.0f} $\\pm$ {total.std():.0f} ms", transform=ax.transAxes, ha="right",
            va="top")
    style.save_figure(fig, HERE / "step_times")


if __name__ == "__main__":
    main()
