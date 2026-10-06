from __future__ import annotations

from string import ascii_lowercase

import numpy as np
from matplotlib.lines import Line2D

from experiments_new.common import style


def embedding(ax, Z: np.ndarray, categories: np.ndarray, colours, *, glyphs: int | None = None) -> None:
    """One embedding as a scatter of its points, coloured by category; rasterised so a PDF stays small. With
    ``glyphs``, all points are faint dots and a random ``glyphs`` of them are drawn as their category's number."""
    dots = dict(s=0.15, linewidths=0, rasterized=True)
    if glyphs is None:
        ax.scatter(Z[:, 0], Z[:, 1], c=categories, cmap=colours, vmin=-0.5, vmax=colours.N - 0.5, **dots)
    else:
        ax.scatter(Z[:, 0], Z[:, 1], color="0.85", **dots)
        shown = np.random.default_rng(0).choice(len(Z), min(glyphs, len(Z)), replace=False)
        for category in np.unique(categories[shown]):
            rows = shown[categories[shown] == category]
            ax.scatter(Z[rows, 0], Z[rows, 1], marker=f"${category}$", s=14, linewidths=0, color=colours(category),
                       rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)


def embeddings(panels: dict[str, dict[str, np.ndarray]], *, legend_title: str = "digit", glyphs: int | None = None):
    """One panel per embedding, labelled (a), (b), ..., with one shared legend for the categories (none with
    ``glyphs``: the points are drawn as their categories)."""
    n_categories = int(max(panel["labels"].max() for panel in panels.values())) + 1
    colours = style.category_colours(n_categories)
    fig, axes = style.figure("full", aspect=0.55, ncols=len(panels))
    for letter, ax, (title, panel) in zip(ascii_lowercase, np.atleast_1d(axes), panels.items()):
        embedding(ax, panel["Z"], panel["labels"], colours, glyphs=glyphs)
        ax.set_title(f"({letter}) {title}", loc="left")
    if glyphs is None:
        handles = [Line2D([], [], marker="o", linestyle="", color=colours(i), label=str(i))
                   for i in range(n_categories)]
        fig.legend(handles=handles, title=legend_title, loc="outside lower center", ncols=n_categories)
    return fig
