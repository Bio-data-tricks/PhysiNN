"""Spectral line profile utilities."""

from __future__ import annotations

import math

import torch

from .constants import INV_SQRT_PI, SQRT_LN2


_B = torch.tensor(
    [-0.0173 - 0.0463j, -0.7399 + 0.8395j, 5.8406 + 0.9536j, -5.5834 - 11.2086j],
    dtype=torch.cdouble,
)
_B = torch.cat((_B, _B.conj()))
_C = torch.tensor(
    [2.2377 - 1.626j, 1.4652 - 1.7896j, 0.8393 - 1.892j, 0.2739 - 1.9418j],
    dtype=torch.cdouble,
)
_C = torch.cat((_C, -_C.conj()))


def wofz_torch(z: torch.Tensor) -> torch.Tensor:
    b_loc = _B.to(device=z.device, dtype=z.dtype)
    c_loc = _C.to(device=z.device, dtype=z.dtype)
    w_pos = (b_loc / (z.unsqueeze(-1) - c_loc)).sum(dim=-1) * (1j * INV_SQRT_PI)
    w_neg = (b_loc / ((-z).unsqueeze(-1) - c_loc)).sum(dim=-1) * (1j * INV_SQRT_PI)
    return torch.where(z.imag < 0, 2.0 * torch.exp(-(z**2)) - w_neg, w_pos)


def pine_profile_torch_complex(x, sigma_hwhm, gamma, g_dicke):
    xh = SQRT_LN2 * x / sigma_hwhm
    yh = SQRT_LN2 * gamma / sigma_hwhm
    z_d = SQRT_LN2 * g_dicke / sigma_hwhm
    z = xh + 1j * (yh + z_d)
    k = -wofz_torch(z)
    real, imag = k.real, k.imag
    pi_sqrt = math.sqrt(math.pi)
    denom = (1 - z_d * pi_sqrt * real) ** 2 + (z_d * pi_sqrt * imag) ** 2
    real_out = (real - z_d * pi_sqrt * (real**2 + imag**2)) / denom
    imag_out = imag / denom
    factor = math.sqrt(math.log(2.0) / math.pi) / sigma_hwhm
    return real_out * factor, imag_out * factor


def apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T, P, *, pref=1013.25, tref=296.0):
    flm = lmf * ((tref / T) ** nlmf) * (P / pref)
    return -(real_prof + imag_prof * flm)

