"""Galleries: every method's embedding of one dataset, side by side, coloured the dataset's way.

A gallery folder holds a small ``run.py`` that names its dataset and panels and
calls ``run_gallery``. Each panel embeds the dataset in its own memory-guarded
process (``python common/gallery.py ...``), saving ``embeddings/<panel>.npz`` (the
embedding, colours and reference view) and ``<panel>.json`` (settings, time,
scores, how piled up it looks); finished panels are skipped. Then
``plot_gallery`` draws ``figure.png``: one column per panel (after the truth,
for datasets that have a reference view), one row per colouring.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402


@dataclass(frozen=True)
class Panel:
    name: str                                   # file name and column
    method: str                                 # common.methods.METHODS key
    override: dict = field(default_factory=dict)   # settings on top of the method's tuned ones
    title: str | None = None                    # default: the method's label


def run_gallery(folder: Path, dataset: str, panels: list[Panel], *, split: str = "eval", n: int | None = None,
                seed: int = 0, threads: int = 16, memory_limit_mb: int = 5000) -> Path:
    from common.runner import thread_env

    for p in panels:
        if (folder / "embeddings" / f"{p.name}.npz").exists():
            continue
        cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", str(memory_limit_mb), "--",
               sys.executable, str(Path(__file__).resolve()), "--folder", str(folder), "--dataset", dataset,
               "--split", split, "--seed", str(seed), "--name", p.name, "--method", p.method,
               "--override", json.dumps(p.override)] + (["--n", str(n)] if n else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(threads))
        print([l for l in proc.stdout.splitlines() if l.startswith(p.name)] or proc.stderr[-1500:], flush=True)
    return plot_gallery(folder, dataset, panels)


def _embed(folder: Path, dataset: str, split: str, n: int | None, seed: int, name: str, method: str,
           override: dict) -> None:
    from common.datasets import DATASETS, load
    from common.methods import METHODS, _merge, method_params
    from common.metrics import quality, stacking

    ds, m = DATASETS[dataset], METHODS[method]
    X, labels = load(split, n=n, seed=seed, dataset=dataset)
    params = _merge(method_params(method), override)
    m.embed(X[:2000].copy(), params, seed)                         # warm-up: one-off start-up costs
    t0 = time.perf_counter()
    Z, fit_s, info = m.embed(X, params, seed)
    total_s = time.perf_counter() - t0
    scores = {**quality(X, Z, labels, seed=seed, extra=ds.extra_metrics), **stacking(Z)}
    colours = ds.colourings(X, labels) if ds.colourings else {}
    (folder / "embeddings").mkdir(exist_ok=True)
    np.savez(folder / "embeddings" / f"{name}.npz", Z=Z, **{f"colour_{k}": c.values for k, c in colours.items()},
             **({"reference": ds.reference(X)} if ds.reference else {}))
    (folder / f"{name}.json").write_text(json.dumps(
        {"dataset": dataset, "split": split, "n": len(X), "seed": seed, "method": method, "label": m.label,
         "params": params, "fit_s": fit_s, "total_s": total_s, "info": info,
         "colourings": {k: {"cmap": c.cmap, "categorical": c.categorical} for k, c in colours.items()},
         **scores}, indent=1, default=str))
    print(name, {k: round(v, 3) for k, v in scores.items()}, f"{total_s:.1f}s", flush=True)


def _settings(info: dict) -> str:
    p = info["params"]
    if info["method"].startswith("lvm"):
        return f"{p['cells']['n_cells']} cells" + (", local PCA per cell" if info["method"].startswith("lvm_pca") else "")
    return ", ".join(f"{k}={v}" for k, v in p.items())


def plot_gallery(folder: Path, dataset: str, panels: list[Panel]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common.datasets import DATASETS

    ds = DATASETS[dataset]
    panels = [p for p in panels if (folder / "embeddings" / f"{p.name}.npz").exists()]
    data = {p.name: (np.load(folder / "embeddings" / f"{p.name}.npz"), json.loads((folder / f"{p.name}.json").read_text()))
            for p in panels}
    first = data[panels[0].name][0]
    styles = data[panels[0].name][1]["colourings"]
    colourings = list(styles)
    columns = ([("truth (seen from above)", first["reference"], None)] if "reference" in first.files else []) + [
        (p.title or data[p.name][1]["label"], data[p.name][0]["Z"], data[p.name][1]) for p in panels]
    # One colouring: wrap the panels 4 per row. Several: one row per colouring.
    if len(colourings) == 1:
        cols = min(4, len(columns))
        grid = [(i // cols, i % cols, colourings[0]) for i in range(len(columns))]
        rows = -(-len(columns) // cols)
    else:
        cols, rows = len(columns), len(colourings)
        grid = [(r, c, colourings[r]) for r in range(rows) for c in range(cols)]
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    cmaps = {k: v["cmap"] for k, v in styles.items()}
    categorical = {k: v["categorical"] for k, v in styles.items()}
    for r, c, colouring in grid:
        title, Z, info = columns[c if len(colourings) > 1 else r * cols + c]
        ax = axes[r, c]
        values = first[f"colour_{colouring}"]
        order = np.random.default_rng(0).permutation(len(Z))       # no category drawn on top of all others
        kw = dict(vmin=-0.5, vmax=9.5) if categorical[colouring] else {}
        ax.scatter(Z[order, 0], Z[order, 1], c=values[order], cmap=cmaps[colouring], s=0.4, rasterized=True, **kw)
        if info is not None and (len(colourings) == 1 or r == 0):
            title = (f"{title}\n{_settings(info)}\ntrust {info['trustworthiness']:.3f}  "
                     f"cont {info['continuity']:.3f}  global {info['distance_correlation']:.3f}"
                     + ("" if info["knn_accuracy"] != info["knn_accuracy"] else f"  5-NN {info['knn_accuracy']:.3f}")
                     + f"\n{info['total_s']:.1f} s;  visible share {info['visible_share']:.0%} "
                       f"(occupied squares of a 1000x1000 grid per point)")
        if len(colourings) == 1 or r == 0:
            ax.set_title(title, fontsize=8.5)
        if c == 0 and len(colourings) > 1:
            ax.set_ylabel(f"coloured by {colouring}")
        ax.set_xticks([]), ax.set_yticks([])
        if not categorical[colouring]:
            ax.set_aspect("equal", adjustable="datalim")
    for ax in axes.flat[len(grid):]:
        ax.axis("off")
    for colouring in colourings:            # categories: one colour legend for the figure, nothing drawn on the data
        if categorical[colouring]:
            from matplotlib.lines import Line2D

            cats = np.unique(first[f"colour_{colouring}"])
            cmap = plt.get_cmap(cmaps[colouring])
            handles = [Line2D([], [], ls="", marker="o", ms=8, color=cmap((c + 0.5) / 10)) for c in cats]
            fig.legend(handles, [str(c) for c in cats], title=colouring, loc="center right", fontsize=11,
                       title_fontsize=11, frameon=False)
            fig.subplots_adjust(right=0.94)
    info0 = data[panels[0].name][1]
    fig.suptitle(f"{ds.title}: {info0['split']} split, n = {info0['n']}, seed {info0['seed']}; "
                 "each method at its MNIST-tuned settings", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.95, 1) if any(categorical.values()) else (0, 0, 1, 1))
    out = folder / "figure.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--override", default="{}")
    args = parser.parse_args()
    _embed(args.folder, args.dataset, args.split, args.n, args.seed, args.name, args.method, json.loads(args.override))


if __name__ == "__main__":
    main()
