from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import psutil

DEFAULT_LIMIT_MB = 3000


def tree_rss_mb(proc: psutil.Process) -> float:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 2**20


def kill_tree(proc: psutil.Process) -> None:
    for p in [*proc.children(recursive=True), proc]:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def run_guarded(cmd: list[str], limit_mb: float = DEFAULT_LIMIT_MB, cwd: Path | None = None) -> int:
    child = subprocess.Popen(cmd, cwd=cwd)
    proc = psutil.Process(child.pid)
    peak = 0.0
    while child.poll() is None:
        try:
            rss = tree_rss_mb(proc)
        except psutil.NoSuchProcess:
            break
        peak = max(peak, rss)
        if rss > limit_mb:
            kill_tree(proc)
            child.wait()
            print(f"[guard] killed: memory {rss:.0f} MB > limit {limit_mb:.0f} MB", file=sys.stderr, flush=True)
            return 137
        time.sleep(0.2)
    print(f"[guard] peak memory {peak:.0f} MB (limit {limit_mb:.0f} MB)", file=sys.stderr, flush=True)
    return child.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a command, killing it if its memory use passes a limit.")
    parser.add_argument("--limit-mb", type=float, default=DEFAULT_LIMIT_MB)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("no command given")
    sys.exit(run_guarded(cmd, args.limit_mb))


if __name__ == "__main__":
    main()
