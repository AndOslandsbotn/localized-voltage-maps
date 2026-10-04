"""3-D embeddings of the half sphere: can a method bring back the bowl, not just flatten it into a disc?

In 2-D every method must flatten the half sphere. In 3-D, a method that captures
distances along the surface (as LVM's -log v approximately does) needs the third
direction to fit them, so classical-MDS-type methods should return a bowl.
Panels: the truth; LVM with the default landmark count (at least 4 for 3-D), the
same with its local PCA chart (each cell's 3 main directions), and LVM with 12; UMAP; t-SNE on the CPU (openTSNE,
Barnes-Hut: cuML t-SNE is 2-D only; only the picture matters here, not the time). Each in its own memory-guarded
process; then figure.png.

    python benchmarks/half_sphere/gallery_3d/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

from common.runner import thread_env  # noqa: E402

LVM_3D = {"embedding": {"landmark_mds": {"n_components": 3}}}
PANELS = [
    ("lvm", "lvm_gpu", LVM_3D, "LVM, default landmark count"),
    ("lvm_pca", "lvm_pca_gpu", LVM_3D, "LVM + local PCA chart (3 directions per cell), default landmark count"),
    ("lvm_L12", "lvm_gpu", {**LVM_3D, "landmarks": {"n_landmarks": 12, "count": {"strategy": "fixed"}}}, "LVM, 12 landmarks"),
    ("umap", "umap_gpu", {"n_components": 3}, "UMAP"),
    ("tsne", "tsne_cpu", {"n_components": 3}, "t-SNE (CPU, openTSNE Barnes-Hut: cuML t-SNE is 2-D only)"),
]


def main() -> None:
    for name, method, override, _ in PANELS:
        if (HERE / "embeddings" / f"{name}.npz").exists():
            continue
        cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "4500", "--", sys.executable,
               str(HERE / "one.py"), "--name", name, "--method", method, "--override", json.dumps(override)]
        out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
        print([l for l in out.stdout.splitlines() if l.startswith(name)] or out.stderr[-1500:], flush=True)
    subprocess.run([sys.executable, str(HERE / "plot.py")], check=True)


if __name__ == "__main__":
    main()
