"""Results files: append-as-you-go CSVs, plus a metadata file describing how they were made.

Every experiment folder holds ``results.csv`` (one row per run, rewritten
after each run so an interrupted experiment keeps what it has) and
``metadata.json`` (hardware, library versions, git commit, settings).
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ("torch", "cuml-cu13", "umap-learn", "scikit-learn", "numpy", "scipy", "faiss-cpu", "pyamg", "numba",
            "array-api-compat")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu_model() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception:
        return "unknown"


def write_metadata(folder: Path, settings: dict) -> None:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    meta = {
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "gpu": _gpu_model(),
        "versions": versions,
        "settings": settings,
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
