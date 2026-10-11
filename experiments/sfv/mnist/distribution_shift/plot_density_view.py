from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from experiments.common import style
from experiments.sfv.mnist.distribution_shift.run import COMMON, shifted

HERE = Path(__file__).resolve().parent
SEED, BINS = 0, 60
CONDITIONS = {"uniform": "uniform masses", "balanced": "masses from\nbalanced data",
              "shifted": "masses from\nshifted data", "refit": "refit on\nshifted data"}


def histograms(Z: np.ndarray, common: np.ndarray):
    edges = [np.linspace(*np.quantile(Z[:, axis], [0.005, 0.995]), BINS + 1) for axis in (0, 1)]
    everything = np.histogram2d(Z[:, 0], Z[:, 1], bins=edges)[0]
    ones_and_sevens = np.histogram2d(Z[common, 0], Z[common, 1], bins=edges)[0]
    return edges, everything, ones_and_sevens


def main() -> None:
    labels = shifted(SEED)[1]
    common = np.isin(labels, COMMON)
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.85, nrows=3, ncols=len(CONDITIONS))
    for column, (condition, title) in enumerate(CONDITIONS.items()):
        Z = np.load(HERE / "embeddings" / f"lvm_chart_{condition}_seed{SEED}.npz")["Z"]
        edges, everything, ones_and_sevens = histograms(Z, common)
        occupied = everything > 0
        area = np.mean(ones_and_sevens[occupied] > everything[occupied] / 2)

        ax = axes[0, column]
        ax.scatter(Z[~common, 0], Z[~common, 1], color="0.6", s=0.4, alpha=0.08, linewidths=0, rasterized=True)
        for digit in COMMON:
            chosen = labels == digit
            ax.scatter(Z[chosen, 0], Z[chosen, 1], color=colours(digit), s=0.4, alpha=0.08, linewidths=0,
                       rasterized=True)
        ax.set_title(f"{title}\n1s, 7s: {100 * area:.0f}% of area", fontsize="small")

        ax = axes[1, column]
        ax.pcolormesh(edges[0], edges[1], everything.T, norm=LogNorm(vmin=1), cmap=style.continuous_colours(),
                      rasterized=True)

        axes[2, column].remove()
        ax3 = fig.add_subplot(3, len(CONDITIONS), 2 * len(CONDITIONS) + column + 1, projection="3d")
        centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
        x, y = np.meshgrid(*centres, indexing="ij")
        height = gaussian_filter(np.log1p(everything), 1.0)
        share = np.divide(ones_and_sevens, everything, out=np.zeros_like(everything), where=everything > 0)
        ax3.plot_surface(x, y, height, facecolors=style.continuous_colours()(gaussian_filter(share, 1.0)),
                         rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        ax3.set_axis_off()
        ax3.view_init(elev=35, azim=-60)
        for ax in axes[:2, column]:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
    axes[0, 0].text(-0.08, 0.5, "points (1s, 7s coloured)", transform=axes[0, 0].transAxes, rotation=90,
                    ha="right", va="center", fontsize="small")
    axes[1, 0].text(-0.08, 0.5, "density (log)", transform=axes[1, 0].transAxes, rotation=90, ha="right",
                    va="center", fontsize="small")
    fig.text(0.5, 0.01, "1s and 7s are 71% of the points. Bottom row: height = log density; colour = share of 1s and 7s "
             "(dark: none, light: all).", fontsize="small", ha="center")
    style.save_figure(fig, HERE / "density_view")


if __name__ == "__main__":
    main()
