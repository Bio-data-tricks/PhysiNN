"""Dataset and data-loading utilities."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from .config import PARAMS, NORMALIZATION
from .lowess import lowess_value
from .noise import add_noise_variety
from .normalization import norm_param_value
from .physics.spectra import batch_physics_forward_multimol_vgrid
from .physics.qtpy import Tips2021QTpy


class SpectraDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        num_points: int,
        poly_freq_CH4,
        transitions_dict,
        sample_ranges: Optional[Dict[str, tuple]] = None,
        strict_check: bool = True,
        with_noise: bool = True,
        noise_profile: Optional[dict] = None,
        freeze_noise: bool = False,
        tipspy: Tips2021QTpy | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.num_points = num_points
        self.poly_freq_CH4 = poly_freq_CH4
        self.transitions_dict = transitions_dict
        self.sample_ranges = sample_ranges if sample_ranges is not None else NORMALIZATION.as_dict()
        self.with_noise = bool(with_noise)
        self.noise_profile = dict(noise_profile or {})
        self.freeze_noise = bool(freeze_noise)
        self.tipspy = tipspy
        self.epoch = 0

        if strict_check:
            for name in PARAMS:
                sample_min, sample_max = self.sample_ranges[name]
                norm_min, norm_max = NORMALIZATION.get(name)
                if sample_min < norm_min or sample_max > norm_max:
                    raise ValueError(
                        f"sample_ranges['{name}']={self.sample_ranges[name]} exceeds registered normalisation"
                        f" range {NORMALIZATION.get(name)}."
                    )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_generator(self, index: int) -> torch.Generator:
        base = torch.initial_seed()
        generator = torch.Generator(device="cpu")
        if self.freeze_noise:
            seed = (123456789 + 97 * index) % (2**63 - 1)
        else:
            seed = (base + 1_000_003 * self.epoch + 97 * index) % (2**63 - 1)
        generator.manual_seed(seed)
        return generator

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        device, dtype = "cpu", torch.float32
        sampled = {name: torch.empty(1, dtype=dtype).uniform_(*self.sample_ranges[name]) for name in PARAMS}
        sig0 = sampled["sig0"]
        dsig = sampled["dsig"]
        baseline0 = sampled["baseline0"]
        baseline1 = sampled["baseline1"]
        baseline2 = sampled["baseline2"]
        pressure = sampled["P"]
        temperature = sampled["T"]

        baseline_coeffs = torch.cat([baseline0, baseline1, baseline2]).unsqueeze(0)
        v_grid_idx = torch.arange(self.num_points, dtype=dtype, device=device)

        mf_dict = {}
        if "CH4" in self.transitions_dict:
            mf_dict["CH4"] = sampled["mf_CH4"]
        if "H2O" in self.transitions_dict:
            mf_dict["H2O"] = sampled["mf_H2O"]

        params_norm = torch.tensor(
            [norm_param_value(name, sampled[name].item()) for name in PARAMS], dtype=torch.float32
        )

        spectra_clean, _ = batch_physics_forward_multimol_vgrid(
            sig0,
            dsig,
            self.poly_freq_CH4,
            v_grid_idx,
            baseline_coeffs,
            self.transitions_dict,
            pressure,
            temperature,
            mf_dict,
            tipspy=self.tipspy,
            device=device,
        )
        spectra_clean = spectra_clean.to(torch.float32)

        if self.with_noise:
            generator = self._make_generator(idx)
            spectra_noisy = add_noise_variety(spectra_clean, generator=generator, **self.noise_profile)
        else:
            spectra_noisy = spectra_clean

        scale_noisy = lowess_value(spectra_noisy, kind="start", window=30).unsqueeze(1).clamp_min(1e-8)
        noisy_spectra = spectra_noisy / scale_noisy

        max_clean = lowess_value(spectra_clean, kind="start", window=30).unsqueeze(1).clamp_min(1e-8)
        clean_spectra = spectra_clean / max_clean

        return {
            "noisy_spectra": noisy_spectra[0].detach(),
            "clean_spectra": clean_spectra[0].detach(),
            "params": params_norm,
            "scale": scale_noisy.squeeze(1).to(torch.float32)[0],
        }

