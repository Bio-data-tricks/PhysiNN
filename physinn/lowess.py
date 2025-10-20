"""Torch-compatible LOWESS smoothing utilities."""

from __future__ import annotations

import math
from typing import Callable

import torch


def _k_from_length(length: int, frac: float | None, window: int | None) -> int:
    if window is not None:
        k = int(window)
    elif frac is not None:
        k = int(max(5, frac * length))
    else:
        k = max(5, int(0.08 * length))
    return max(5, min(k, length))


def _solve_wls(design: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted = design * weights.unsqueeze(1)
    xtwx = design.T @ weighted
    xtwy = weighted.T @ targets
    return torch.linalg.pinv(xtwx) @ xtwy


def _lowess_at_1d(values: torch.Tensor, x0: float, k: int, iterations: int = 2) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("LOWESS expects a 1D tensor.")

    length = values.numel()
    device = values.device
    dtype = torch.float64
    indices = torch.arange(length, device=device, dtype=dtype)
    samples = values.to(dtype)

    distances = (indices - x0).abs()
    neighbours = torch.topk(distances, k, largest=False).indices
    xs = indices[neighbours]
    ys = samples[neighbours]
    dmax = distances[neighbours].max().clamp_min(1e-12)

    u = (distances[neighbours] / dmax).clamp(max=1)
    weights = (1 - u.pow(3)).clamp(min=0).pow(3)

    design = torch.stack([torch.ones_like(xs), (xs - x0)], dim=1)
    beta = _solve_wls(design, ys, weights)

    for _ in range(max(0, iterations - 1)):
        residuals = ys - (design @ beta)
        scale = torch.median(torch.abs(residuals)) + 1e-12
        uu = (residuals / (6 * scale)).clamp(min=-1, max=1)
        robust = (1 - uu.pow(2)).clamp(min=0).pow(2)
        beta = _solve_wls(design, ys, weights * robust + 1e-12)

    return beta[0].to(values.dtype)


def lowess_value(
    values: torch.Tensor,
    kind: str = "start",
    *,
    frac: float | None = 0.08,
    window: int | None = None,
    x0: float | None = None,
    n_eval: int = 64,
    iterations: int = 2,
    clamp_min_value: float = 1e-6,
) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(0)
        squeeze = True
    elif values.ndim == 2:
        squeeze = False
    else:
        raise ValueError("LOWESS expects a 1D or 2D tensor.")

    batch, length = values.shape
    k = _k_from_length(length, frac, window)

    def _evaluate(sample: torch.Tensor) -> torch.Tensor:
        match kind:
            case "start":
                return _lowess_at_1d(sample, x0=0.0, k=k, iterations=iterations)
            case "at":
                if x0 is None:
                    raise ValueError("Parameter 'x0' must be provided when kind='at'.")
                return _lowess_at_1d(sample, float(x0), k=k, iterations=iterations)
            case "max":
                xs = torch.linspace(0, max(0, length - 1), n_eval, device=sample.device, dtype=torch.float32)
                vals = torch.stack([
                    _lowess_at_1d(sample, float(pos), k=k, iterations=iterations) for pos in xs
                ])
                return vals.max()
            case _:
                raise ValueError("kind must be 'start', 'at' or 'max'.")

    output = torch.stack([_evaluate(values[i]) for i in range(batch)]).to(values.dtype)
    floor = torch.tensor(clamp_min_value, dtype=output.dtype, device=output.device)
    output = output.clamp_min(floor)
    return output[0] if squeeze else output

