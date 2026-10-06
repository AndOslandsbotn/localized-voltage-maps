"""MNIST with and without the 0s: LVM, LVM + local PCA and UMAP (reference), each run in its own memory-guarded
process; figure.png puts the two versions of each method above each other.

    python experiments_old/sfv/mnist/diagnostics/without_zeros/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.runner import thread_env  # noqa: E402

METHODS = {"lvm_gpu": "LVM", "lvm_pca_gpu": "LVM + local PCA", "umap_gpu": "UMAP"}


def plot(rows: list) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, len(METHODS), figsize=(7 * len(METHODS), 12.5))
    for col, (method, title) in enumerate(METHODS.items()):
        for row, drop in enumerate((False, True)):
            ax = axes[row, col]
            r = next((r for r in rows if r["method"] == method and r["drop_zeros"] == drop), None)
            if r is None:
                ax.axis("off")
                continue
            e = np.load(HERE / "embeddings" / f"{r['name']}.npz")
            order = np.random.default_rng(0).permutation(len(e["y"]))
            ax.scatter(e["Z"][order, 0], e["Z"][order, 1], c=e["y"][order], cmap="tab10", vmin=-0.5, vmax=9.5,
                       s=0.3, rasterized=True)
            ax.set_title(f"{title}, {'without' if drop else 'with'} the 0s (n = {r['n']:,})\n"
                         f"trust {r['trustworthiness']:.3f}  cont {r['continuity']:.3f}  5-NN {r['knn_accuracy']:.3f}  "
                         f"global {r['distance_correlation']:.3f}  visible {100 * r['visible_share']:.0f}%", fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
    cmap = plt.get_cmap("tab10")
    fig.legend([Line2D([], [], ls="", marker="o", ms=8, color=cmap((d + 0.5) / 10)) for d in range(10)],
               [str(d) for d in range(10)], title="digit", loc="center right", frameon=False)
    fig.suptitle("MNIST evaluation split (seed 0): with all digits (top) and with the 0s removed (bottom)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    fig.savefig(HERE / "figure.png", dpi=100)
    plt.close(fig)
    print(HERE / "figure.png")


def main() -> None:
    path = HERE / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    for method in METHODS:
        for drop in (False, True):
            if any(r["method"] == method and r["drop_zeros"] == drop for r in rows):
                continue
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
                   str(HERE / "one.py"), "--method", method] + (["--drop-zeros"] if drop else [])
            out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
            lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
            if not lines:
                print(f"{method} drop={drop}: failed\n{out.stderr[-800:]}", flush=True)
                continue
            rows.append(json.loads(lines[-1][7:]))
            with open(path, "a") as f:
                f.write(json.dumps(rows[-1]) + "\n")
            r = rows[-1]
            print(f"{r['name']:28s} trust {r['trustworthiness']:.3f}  cont {r['continuity']:.3f}  "
                  f"5-NN {r['knn_accuracy']:.3f}  global {r['distance_correlation']:.3f}", flush=True)
    plot(rows)


if __name__ == "__main__":
    main()
