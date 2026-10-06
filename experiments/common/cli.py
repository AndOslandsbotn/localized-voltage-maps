from __future__ import annotations

import argparse


def arguments(description: str) -> argparse.Namespace:
    """Every experiment runs with no arguments; --recompute ignores what is already computed."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--recompute", action="store_true", help="compute again instead of reusing saved results")
    return parser.parse_args()
