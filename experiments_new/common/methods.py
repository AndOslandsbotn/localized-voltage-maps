from __future__ import annotations

import numpy as np

from lvm_new import LocalizedVoltageMaps


def lvm(X: np.ndarray, *, seed: int, device: str) -> np.ndarray:
    return LocalizedVoltageMaps(local_chart="none", random_state=seed, device=device).fit(X).transform(X)


def lvm_chart(X: np.ndarray, *, seed: int, device: str) -> np.ndarray:
    return LocalizedVoltageMaps(local_chart="last", random_state=seed, device=device).fit(X).transform(X)


METHODS = {"lvm": lvm, "lvm_chart": lvm_chart}
TITLES = {"lvm": "LVM", "lvm_chart": "LVM + local chart"}
