"""Noise augmentation utilities."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def add_noise_variety(spectra, *, generator=None, **cfg):
    std_add_range = cfg.get("std_add_range", (0.001, 0.01))
    std_mult_range = cfg.get("std_mult_range", (0.002, 0.02))
    p_drift = cfg.get("p_drift", 0.7)
    drift_sigma_range = cfg.get("drift_sigma_range", (8.0, 90.0))
    drift_amp_range = cfg.get("drift_amp_range", (0.002, 0.03))
    p_fringes = cfg.get("p_fringes", 0.6)
    n_fringes_range = cfg.get("n_fringes_range", (1, 3))
    fringe_freq_range = cfg.get("fringe_freq_range", (0.2, 12.0))
    fringe_amp_range = cfg.get("fringe_amp_range", (0.001, 0.01))
    p_spikes = cfg.get("p_spikes", 0.4)
    spikes_count_range = cfg.get("spikes_count_range", (1, 4))
    spike_amp_range = cfg.get("spike_amp_range", (0.002, 0.03))
    spike_width_range = cfg.get("spike_width_range", (1.0, 4.0))
    clip = cfg.get("clip", (0.0, 1.3))

    y = spectra
    batch, length = y.shape[-2], y.shape[-1]
    device, dtype = y.device, y.dtype
    g = generator

    def r(a, b):
        return (torch.rand((), device=device, generator=g) * (b - a) + a).item()

    def ri(a, b):
        return int(torch.randint(a, b + 1, (), device=device, generator=g).item())

    def rbool(p):
        return bool(torch.rand((), device=device, generator=g) < p)

    std_add = r(*std_add_range)
    std_mult = r(*std_mult_range)
    add = torch.randn(y.shape, device=device, dtype=dtype, generator=g) * std_add
    add = add - add.mean(dim=-1, keepdim=True)
    mult = 1.0 + torch.randn(y.shape, device=device, dtype=dtype, generator=g) * std_mult
    mult = mult / mult.mean(dim=-1, keepdim=True)
    out = y * mult + add

    if rbool(p_drift):
        sigma = r(*drift_sigma_range)
        radius = max(1, int(3 * sigma))
        kernel_x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (kernel_x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        drift = torch.randn((batch, length), device=device, dtype=dtype, generator=g)
        drift = drift - drift.mean(dim=-1, keepdim=True)
        drift = F.pad(drift.unsqueeze(1), (radius, radius), mode="reflect")
        drift = F.conv1d(drift, kernel.view(1, 1, -1)).squeeze(1)
        drift = drift / (drift.std(dim=-1, keepdim=True) + 1e-8) * r(*drift_amp_range)
        out = out + drift

    if rbool(p_fringes):
        t = torch.linspace(0, 1, length, device=device, dtype=dtype)
        fringes = torch.zeros((batch, length), device=device, dtype=dtype)
        for _ in range(ri(*n_fringes_range)):
            freq = r(*fringe_freq_range)
            phase = r(0.0, 2 * math.pi)
            amp = r(*fringe_amp_range)
            fringes = fringes + amp * torch.sin(2 * math.pi * freq * t + phase)
        out = out + fringes

    if rbool(p_spikes):
        grid = torch.arange(length, device=device, dtype=dtype)
        spikes = torch.zeros((batch, length), device=device, dtype=dtype)
        for _ in range(ri(*spikes_count_range)):
            mu = r(0.0, length - 1.0)
            width = r(*spike_width_range)
            amp = r(*spike_amp_range) * (1.0 if rbool(0.5) else -1.0)
            spikes = spikes + amp * torch.exp(-0.5 * ((grid - mu) / width) ** 2)
        out = out + spikes

    if clip is not None:
        out = torch.clamp(out, *clip)
    return out

