"""Run a command, killing it if its memory use passes a limit.

This machine has 7.8 GB of RAM (WSL), ~2 GB of which the OS, WSL and the
editor need; a process that grows past what is left takes the whole machine
down. Every heavy command should run under this guard (or, for benchmark
runs, through ``runner.run_isolated``, which applies the same guard per run).

    python experiments/common/guard.py [--limit-mb 4500] -- <command> [args ...]

The guard sums the resident memory of the command's whole process tree every
0.2 s, kills the tree when it exceeds the limit, and exits with code 137.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import psutil

DEFAULT_LIMIT_MB = 4500


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


def run_guarded(cmd: list[str], limit_mb: float = DEFAULT_LIMIT_MB) -> int:
    child = subprocess.Popen(cmd)
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit-mb", type=float, default=DEFAULT_LIMIT_MB)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("no command given")
    sys.exit(run_guarded(cmd, args.limit_mb))


if __name__ == "__main__":
    main()
