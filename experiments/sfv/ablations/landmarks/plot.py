"""Figures of the landmark ablation (results.jsonl, embeddings/*.npz, best.json):

* figure_strategy.png: metrics per strategy and dataset (mean and spread over seeds, plain LVM), and the seed-0
  embeddings (strategies x datasets) with the landmarks marked: with LVM's local PCA chart where run.py --pictures
  made them (results_pca.jsonl), else plain LVM.
* figure_count.png: the best strategy's metrics against the number of landmarks, and its embeddings
  (datasets x count multipliers).

    python experiments/sfv/ablations/landmarks/plot.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGIES = {"mi": "MI", "maxmin": "maxmin", "maxmin+mi": "maxmin → MI", "mi+maxmin": "MI → maxmin"}
COLOURS = {"mi": "tab:blue", "maxmin": "tab:orange", "maxmin+mi": "tab:green", "mi+maxmin": "tab:red"}
DATASETS = {"mnist_tune": "MNIST tuning split (20k)", "mnist": "MNIST evaluation split (50k)",
            "mnist8m": "MNIST8M sample (50k)", "half_sphere": "half sphere (20k)"}
DATASET_COLOURS = {"mnist": "tab:blue", "mnist8m": "tab:purple", "half_sphere": "tab:green"}
METRICS = {"trustworthiness": "trustworthiness", "continuity": "continuity", "knn_accuracy": "5-NN accuracy",
           "distance_correlation": "global distance rank correlation"}
MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
SHOWN = 15_000      # points drawn per embedding: looks the same at this panel size, keeps each figure under 2 MB


def save(fig, path: Path, dpi: int) -> None:
    """PNG with a 256-colour palette: scatter plots need no more, and the file is several times smaller."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, dpi=dpi)
    Image.open(buffer).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def stats(rs, key):
    v = np.array([r[key] for r in rs], dtype=float)
    return (float(np.mean(v)), float(np.std(v))) if len(v) and np.all(np.isfinite(v)) else (np.nan, np.nan)


def embedding(ax, dataset: str, tag: str, title: str) -> None:
    f = HERE / "embeddings" / f"{tag}_pca.npz"
    f = f if f.exists() else HERE / "embeddings" / f"{tag}.npz"
    if not f.exists():
        ax.axis("off")
        return
    e = np.load(f)
    Z, c, L = e["Z"], e["colour"], e["landmarks"]
    order = np.random.default_rng(0).permutation(len(Z))[:SHOWN]           # the scores in the titles use all points
    if dataset == "half_sphere":
        ax.scatter(Z[order, 0], Z[order, 1], c=c[order], cmap="twilight", s=0.5, rasterized=True)
    else:
        ax.scatter(Z[order, 0], Z[order, 1], c=c[order], cmap="tab10", vmin=-0.5, vmax=9.5, s=0.3, rasterized=True)
    ax.scatter(L[:, 0], L[:, 1], marker="X", s=60, c="black", edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])


def metrics_row(axes, rows, groups, series, x_of, label_of, colour_of, xlabel, log=False):
    """One panel per metric: for each series a line/points over the groups, mean +- std over seeds."""
    for ax, (metric, name) in zip(axes, METRICS.items()):
        for s in series:
            xs, ms, sd = [], [], []
            for g in groups:
                rs = select(rows, **x_of(s, g))
                if not rs:
                    continue
                m, d = stats(rs, metric)
                if np.isfinite(m):
                    xs.append(g if not log else np.mean([r["n_landmarks"] for r in rs])), ms.append(m), sd.append(d)
            if xs:
                ax.errorbar(xs, ms, yerr=sd, fmt="o-" if log else "o", color=colour_of(s), label=label_of(s),
                            capsize=3, ms=5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        if log:
            ax.set_xscale("log")
            ax.set_xticks([3, 5, 10, 20, 40, 60], ["3", "5", "10", "20", "40", "60"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].legend(fontsize=8)


def picture_row(rows, pca_rows, **match):
    """The seed-0 run shown in a picture: the charted one if it exists."""
    r = select(pca_rows, seed=0, **match) or select(rows, seed=0, **match)
    return r[0] if r else None


def figure_strategy(rows, pca_rows) -> None:
    fig = plt.figure(figsize=(20, 21), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 3.4])
    axes = top.subplots(1, 4)
    datasets = list(DATASETS)
    for ax, (metric, name) in zip(axes, METRICS.items()):
        for j, s in enumerate(STRATEGIES):
            for i, d in enumerate(datasets):
                m, sd = stats(select(rows, dataset=d, strategy=s, multiplier=1.0), metric)
                if np.isfinite(m):
                    ax.errorbar(i + (j - 1.5) * 0.17, m, yerr=sd, fmt="o", color=COLOURS[s], capsize=3, ms=5,
                                label=STRATEGIES[s] if i == 0 else None)
        ax.set_xticks(range(len(datasets)), [DATASETS[d].split(" (")[0] for d in datasets], fontsize=8, rotation=15)
        ax.set_title(name, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    best = json.loads((HERE / "best.json").read_text())["best"] if (HERE / "best.json").exists() else None
    top.suptitle("Landmark selection strategy, at the default count d̂ + 1 (mean ± std over 5 seeds)"
                 + (f"; chosen: {STRATEGIES[best]} (the metrics tie; chosen on the pictures)" if best else ""),
                 fontsize=12)
    shown = ("mnist", "mnist8m", "half_sphere")
    grid = bottom.subplots(len(STRATEGIES), len(shown))
    for i, s in enumerate(STRATEGIES):
        for j, d in enumerate(shown):
            r = picture_row(rows, pca_rows, dataset=d, strategy=s, multiplier=1.0)
            text = "" if not r else (f"trust {r['trustworthiness']:.3f}  cont {r['continuity']:.3f}  "
                                     + ("" if d == "half_sphere" else f"5-NN {r['knn_accuracy']:.3f}  ")
                                     + f"global {r['distance_correlation']:.3f}")
            embedding(grid[i, j], d, f"{d}_{s}_m1", f"{STRATEGIES[s]} — {DATASETS[d]}\n{text}")
    bottom.suptitle("Embeddings, seed 0" + (", LVM + local PCA chart (fill 0.6)" if pca_rows else ", plain LVM")
                    + " (X: landmarks; MNIST coloured by digit, half sphere by angle)", fontsize=12)
    save(fig, HERE / "figure_strategy.png", dpi=100)
    plt.close(fig)


def figure_count(rows, pca_rows) -> None:
    best_file = HERE / "best.json"
    if not best_file.exists():
        return
    best = json.loads(best_file.read_text())["best"]
    shown = ("mnist", "mnist8m", "half_sphere")
    fig = plt.figure(figsize=(26, 17), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.4])
    metrics_row(top.subplots(1, 4), rows, MULTIPLIERS, shown,
                lambda d, m: {"dataset": d, "strategy": best, "multiplier": m}, lambda d: DATASETS[d],
                lambda d: DATASET_COLOURS[d], "number of landmarks (log scale)", log=True)
    top.suptitle(f"Number of landmarks ({STRATEGIES[best]}): ceil(m (d̂ + 1)) for m = "
                 f"{', '.join(f'{m:g}' for m in MULTIPLIERS)} (mean ± std over 5 seeds)", fontsize=12)
    grid = bottom.subplots(len(shown), len(MULTIPLIERS))
    for i, d in enumerate(shown):
        for j, m in enumerate(MULTIPLIERS):
            r = picture_row(rows, pca_rows, dataset=d, strategy=best, multiplier=m)
            text = "" if not r else (f"L = {r['n_landmarks']}: trust {r['trustworthiness']:.3f}  "
                                     f"global {r['distance_correlation']:.3f}")
            embedding(grid[i, j], d, f"{d}_{best}_m{m:g}", f"{DATASETS[d]}, m = {m:g}\n{text}")
    bottom.suptitle("Embeddings, seed 0" + (", LVM + local PCA chart (fill 0.6)" if pca_rows else ", plain LVM")
                    + " (X: landmarks)", fontsize=12)
    save(fig, HERE / "figure_count.png", dpi=90)
    plt.close(fig)


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines()]
    pca_file = HERE / "results_pca.jsonl"
    pca_rows = [json.loads(l) for l in pca_file.read_text().splitlines()] if pca_file.exists() else []
    figure_strategy(rows, pca_rows)
    figure_count(rows, pca_rows)
    for name in ("figure_strategy.png", "figure_count.png"):
        if (HERE / name).exists():
            print(HERE / name)


if __name__ == "__main__":
    main()
