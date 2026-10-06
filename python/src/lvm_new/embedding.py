from __future__ import annotations

import torch

from lvm_new.config import Config


def fit_embedding(maps: torch.Tensor, *, config: Config):
    raise NotImplementedError


def fit_chart(region, sample: torch.Tensor, *, config: Config, device: str):
    raise NotImplementedError
