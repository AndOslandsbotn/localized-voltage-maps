"""Shared dispatch for the pipeline's configurable choice points.

Every choice point follows the same convention:

* Config: a section with ``strategy: <name>`` and, if the method has
  options, a ``<name>: {...}`` subsection (see ``config.yaml``).
* Module: one public entry point ``<verb>_<thing>(inputs, *, config, ...)``
  (``choose_radius``, ``fit_embedding``, ``extend_voltages``, ...)
  returning a frozen result dataclass, and a registry ``_STRATEGIES`` mapping
  each name to a private function ``_<name>(inputs, *, options, ...)``. The
  options dataclass is passed whole (``None`` for a method without options);
  settings shared by all methods of a section are keyword arguments.

Adding a method: write ``_<name>``, register it, add ``<name>`` to the
section's ``Literal`` in ``config.py`` and its options to ``config.yaml``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping


def resolve(registry: Mapping[str, Callable[..., Any]], config: Any) -> tuple[Callable[..., Any], Any]:
    """The strategy function named by ``config.strategy`` and its options (or ``None``)."""
    return registry[config.strategy], getattr(config, config.strategy, None)
