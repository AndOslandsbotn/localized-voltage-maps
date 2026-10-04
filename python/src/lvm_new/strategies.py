from __future__ import annotations

from typing import Any, Callable, Mapping


def resolve(registry: Mapping[str, Callable[..., Any]], config: Any) -> tuple[Callable[..., Any], Any]:
    """The strategy function named by ``config.strategy``, and its options section (None if it has none)."""
    return registry[config.strategy], getattr(config, config.strategy, None)
