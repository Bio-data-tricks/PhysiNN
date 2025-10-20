"""Spectroscopy core routines."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch

from .constants import C, L0, MOLECULE_PARAMS, P0, R, T0, TREF
from .profiles import apply_line_mixing_complex, pine_profile_torch_complex
from .qtpy import Tips2021QTpy


def parse_csv_transitions(csv_str: str) -> List[dict]:
    transitions = []
    for line in csv_str.strip().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = [tok.strip() for tok in line.split(";")]
        while len(tokens) < 14:
            tokens.append("0")
        transitions.append(
            {
                "mid": int(tokens[0]),
                "lid": int(float(tokens[1])),
                "center": float(tokens[2]),
                "amplitude": float(tokens[3]),
                "gamma_air": float(tokens[4]),
                "gamma_self": float(tokens[5]),
                "e0": float(tokens[6]),
                "n_air": float(tokens[7]),
                "shift_air": float(tokens[8]),
                "abundance": float(tokens[9]),
                "gDicke": float(tokens[10]),
                "nDicke": float(tokens[11]),
                "lmf": float(tokens[12]),
                "nlmf": float(tokens[13]),
            }
        )
    return transitions


def transitions_to_tensors(transitions: Sequence[dict], device: torch.device | str):
    keys = [
        "amplitude",
        "center",
        "gamma_air",
        "gamma_self",
        "n_air",
        "shift_air",
        "gDicke",
        "nDicke",
        "lmf",
        "nlmf",
    ]
    return [
        torch.tensor([transition[key] for transition in transitions], dtype=torch.float32, device=device)
        for key in keys
    ]


def polyval_torch(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    powers = torch.arange(coeffs.shape[1], device=coeffs.device, dtype=coeffs.dtype)
    return torch.sum(coeffs.unsqueeze(2) * x.unsqueeze(0).pow(powers.view(1, -1, 1)), dim=1)


def ST_hitran_with_qtpy(
    Sref,
    nu0,
    e0,
    abundance,
    def_abundance,
    mid_arr,
    iso_arr,
    T_exp,
    mf=None,
    *,
    tipspy: Tips2021QTpy,
    Tref: float = 296.0,
    device=None,
):
    if tipspy is None:
        raise RuntimeError("tipspy (QTpy/TIPS) is required for ST_hitran_with_qtpy().")

    if device is None:
        if torch.is_tensor(Sref):
            device = Sref.device
        elif torch.is_tensor(T_exp):
            device = T_exp.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = Sref.dtype if torch.is_tensor(Sref) else torch.float32

    def to_tensor(value, dtype_=dtype):
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype_, non_blocking=True)
        return torch.as_tensor(value, device=device, dtype=dtype_)

    Sref = to_tensor(Sref)
    nu0 = to_tensor(nu0)
    e0 = to_tensor(e0)
    abundance = to_tensor(abundance)
    def_abundance = to_tensor(def_abundance)
    mid_arr = to_tensor(mid_arr, dtype_=torch.long)
    iso_arr = to_tensor(iso_arr, dtype_=torch.long)
    T_exp = to_tensor(T_exp)

    if T_exp.ndim == 1:
        T_exp = T_exp.view(-1, 1, 1)
    elif T_exp.ndim == 2:
        T_exp = T_exp.view(T_exp.shape[0], 1, 1)
    batch = T_exp.shape[0]

    if Sref.ndim == 3:
        lines = Sref.shape[1]
    else:
        lines = mid_arr.view(-1).numel()

    c2 = torch.tensor(1.438776877, device=device, dtype=dtype)

    mid_lines = mid_arr.view(-1).to(torch.long)[:lines]
    iso_lines = iso_arr.view(-1).to(torch.long)[:lines]
    key = mid_lines * 100 + iso_lines
    unique = torch.unique(key)

    Q_T = torch.empty((batch, lines, 1), device=device, dtype=dtype)
    Q_refT = torch.empty((batch, lines, 1), device=device, dtype=dtype)
    temps = T_exp.view(batch)

    for entry in unique:
        key_value = int(entry.item())
        mid_i = key_value // 100
        iso_i = key_value % 100
        cols = (key == key_value).nonzero(as_tuple=True)[0]
        qT = tipspy.q_torch(mid_i, iso_i, temps).to(device=device, dtype=dtype)
        qRef = tipspy.q_torch(mid_i, iso_i, torch.full_like(temps, float(Tref))).to(device=device, dtype=dtype)
        Q_T[:, cols, 0] = qT.view(batch, 1).expand(batch, cols.numel())
        Q_refT[:, cols, 0] = qRef.view(batch, 1).expand(batch, cols.numel())

    Tref_t = torch.tensor(float(Tref), device=device, dtype=dtype)
    invT = 1.0 / T_exp
    invTref = 1.0 / Tref_t

    expo_fac = torch.exp(-c2 * e0 * (invT - invTref))
    num_fac = 1.0 - torch.exp(-c2 * nu0 * invT)
    den_fac = 1.0 - torch.exp(-c2 * nu0 * invTref)

    eps = torch.tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
    den_fac = torch.where(den_fac == 0, eps, den_fac)
    Q_T = torch.where(Q_T == 0, eps, Q_T)

    return Sref * (Q_refT / Q_T) * expo_fac * (num_fac / den_fac)


def batch_physics_forward_multimol_vgrid(
    sig0,
    dsig,
    poly_freq,
    v_grid_idx,
    baseline_coeffs,
    transitions_dict,
    P,
    T,
    mf_dict,
    *,
    tipspy: Tips2021QTpy,
    device: str = "cpu",
    use_line_mixing: bool = True,
):
    batch, num_points = sig0.shape[0], v_grid_idx.shape[0]
    v_grid_idx = v_grid_idx.to(device=device, dtype=torch.float64)
    sig0 = sig0.to(dtype=torch.float64, device=device).unsqueeze(1)
    dsig = dsig.to(dtype=torch.float64, device=device).unsqueeze(1)
    pressure = P.to(dtype=torch.float64, device=device).unsqueeze(1)
    temperature = T.to(dtype=torch.float64, device=device).unsqueeze(1)

    if baseline_coeffs.dim() == 1:
        baseline_coeffs = baseline_coeffs.unsqueeze(0)
    baseline_coeffs = baseline_coeffs.to(dtype=torch.float64, device=device)

    poly_freq_torch = torch.tensor(poly_freq, dtype=torch.float64, device=device).unsqueeze(0).expand(batch, -1)
    coeffs = torch.cat([sig0, dsig, poly_freq_torch], dim=1)
    v_grid_batch = polyval_torch(coeffs, v_grid_idx)

    total_profile = torch.zeros((batch, num_points), device=device, dtype=torch.float64)

    P0_t = torch.tensor(P0, dtype=torch.float64, device=device)
    T0_t = torch.tensor(T0, dtype=torch.float64, device=device)
    TREF_t = torch.tensor(TREF, dtype=torch.float64, device=device)
    L0_t = torch.tensor(L0, dtype=torch.float64, device=device)
    C_t = torch.tensor(C, dtype=torch.float64, device=device)
    R_t = torch.tensor(R, dtype=torch.float64, device=device)

    for molecule, transitions in transitions_dict.items():
        (amp, center, gamma_air, gamma_self, n_air, shift_air, g_dicke, n_dicke, lmf, nlmf) = [
            tensor.to(dtype=torch.float64, device=device).view(1, -1, 1)
            for tensor in transitions_to_tensors(transitions, device)
        ]
        e0 = torch.tensor([transition["e0"] for transition in transitions], dtype=torch.float64, device=device).view(1, -1, 1)
        abundance = torch.tensor(
            [transition["abundance"] for transition in transitions], dtype=torch.float64, device=device
        ).view(1, -1, 1)
        def_abundance = torch.ones_like(abundance)
        mid_arr = torch.tensor([transition["mid"] for transition in transitions], dtype=torch.int64, device=device).view(
            1, -1, 1
        )
        iso_arr = torch.tensor([transition["lid"] for transition in transitions], dtype=torch.int64, device=device).view(
            1, -1, 1
        )

        mf = mf_dict[molecule].to(dtype=torch.float64, device=device).view(batch, 1, 1)
        mol_params = MOLECULE_PARAMS[molecule]
        mol_mass = torch.tensor(mol_params["M"], dtype=torch.float64, device=device)
        path_length = torch.tensor(mol_params["PL"], dtype=torch.float64, device=device)

        temp_exp = temperature.view(batch, 1, 1)
        press_exp = pressure.view(batch, 1, 1)
        v_exp = v_grid_batch.view(batch, 1, num_points)

        x = v_exp - (center + shift_air * (press_exp / P0_t))
        sigma_hwhm = (center / C_t) * torch.sqrt(2.0 * R_t * temp_exp * math.log(2.0) / mol_mass)
        gamma = (press_exp / P0_t) * (TREF_t / temp_exp) ** n_air * (gamma_air * (1 - mf) + gamma_self * mf)
        gN_eff = g_dicke * (press_exp / P0_t) * (TREF_t / temp_exp) ** n_dicke

        real_prof, imag_prof = pine_profile_torch_complex(x, sigma_hwhm, gamma, gN_eff)
        if use_line_mixing:
            profile = apply_line_mixing_complex(real_prof, imag_prof, lmf, nlmf, T=temp_exp, P=press_exp)
        else:
            profile = -real_prof

        S_T = ST_hitran_with_qtpy(
            Sref=amp,
            nu0=center,
            e0=e0,
            abundance=abundance,
            def_abundance=def_abundance,
            mid_arr=mid_arr,
            iso_arr=iso_arr,
            T_exp=temp_exp,
            mf=mf,
            tipspy=tipspy,
            device=device,
        )

        column = (press_exp / P0_t) * (T0_t / temp_exp) * L0_t * path_length * 100.0 * mf
        band = profile * S_T * column
        total_profile += band.sum(dim=1)

    transmission = torch.exp(total_profile)

    x_baseline = torch.arange(num_points, device=device, dtype=torch.float64)
    powers = torch.arange(baseline_coeffs.shape[1], device=device, dtype=torch.float64)
    baseline = torch.sum(
        baseline_coeffs.unsqueeze(2) * x_baseline.unsqueeze(0).pow(powers.view(1, -1, 1)), dim=1
    )
    return transmission * baseline, v_grid_batch

