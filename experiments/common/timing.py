from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class StepTimer:
    """Wall-clock seconds per step, for ``fit_region(..., timer=StepTimer(device))``. On a GPU it waits for the queued
    work before starting and before stopping the clock: CUDA runs asynchronously, so otherwise a step's GPU work would
    be counted in whichever later step first waits for a result."""

    def __init__(self, device: str):
        self.device = device
        self.seconds: dict[str, float] = {}

    @contextmanager
    def __call__(self, step: str):
        self._synchronize()
        start = time.perf_counter()
        yield
        self._synchronize()
        self.seconds[step] = self.seconds.get(step, 0.0) + time.perf_counter() - start

    def _synchronize(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
