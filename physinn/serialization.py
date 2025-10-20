"""Helpers for turning tensors and numpy objects into serialisable structures."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import yaml


def _to_serializable(x: Any):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.ndim else x.item()
    return x


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):  # pragma: no cover - yaml hook
        return True


def save_config(run_dir_path: str, **cfg) -> str:
    path = os.path.join(run_dir_path, "config.yaml")
    clean = _to_serializable(cfg)
    with open(path, "w") as handle:
        yaml.dump(
            clean,
            handle,
            Dumper=_NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    return path

