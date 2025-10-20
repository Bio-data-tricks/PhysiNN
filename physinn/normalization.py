"""Utility functions for parameter normalisation."""

from __future__ import annotations

import math
from typing import Dict

import torch

from .config import LOG_FLOOR, LOG_SCALE_PARAMS, NORMALIZATION


def norm_param_value(name: str, value: float) -> float:
    vmin, vmax = NORMALIZATION.get(name)
    if name in LOG_SCALE_PARAMS:
        vmin = max(vmin, LOG_FLOOR)
        vmax = max(vmax, vmin * (1 + 1e-12))
        value = max(value, LOG_FLOOR)
        log_value = math.log10(value)
        log_min = math.log10(vmin)
        log_max = math.log10(vmax)
        return (log_value - log_min) / (log_max - log_min)
    return (value - vmin) / (vmax - vmin)


def norm_param_torch(name: str, tensor: torch.Tensor) -> torch.Tensor:
    vmin, vmax = NORMALIZATION.get(name)
    vmin_t = torch.as_tensor(vmin, dtype=tensor.dtype, device=tensor.device)
    vmax_t = torch.as_tensor(vmax, dtype=tensor.dtype, device=tensor.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        eps = torch.finfo(tensor.dtype).eps
        vmax_t = torch.maximum(vmax_t, vmin_t * (1 + eps))
        tensor = torch.clamp(tensor, min=LOG_FLOOR)
        log_tensor = torch.log10(tensor)
        log_min = torch.log10(vmin_t)
        log_max = torch.log10(vmax_t)
        return (log_tensor - log_min) / (log_max - log_min)
    return (tensor - vmin_t) / (vmax_t - vmin_t)


def unnorm_param_torch(name: str, tensor: torch.Tensor) -> torch.Tensor:
    vmin, vmax = NORMALIZATION.get(name)
    vmin_t = torch.as_tensor(vmin, dtype=tensor.dtype, device=tensor.device)
    vmax_t = torch.as_tensor(vmax, dtype=tensor.dtype, device=tensor.device)
    if name in LOG_SCALE_PARAMS:
        vmin_t = torch.clamp(vmin_t, min=LOG_FLOOR)
        eps = torch.finfo(tensor.dtype).eps
        vmax_t = torch.maximum(vmax_t, vmin_t * (1 + eps))
        log_min = torch.log10(vmin_t)
        log_max = torch.log10(vmax_t)
        log_tensor = tensor * (log_max - log_min) + log_min
        return torch.pow(10.0, log_tensor)
    return tensor * (vmax_t - vmin_t) + vmin_t


def as_normalised_vector(params: Dict[str, float]) -> torch.Tensor:
    ordered = [norm_param_value(name, params[name]) for name in NORMALIZATION.as_dict()]
    return torch.tensor(ordered, dtype=torch.float32)

