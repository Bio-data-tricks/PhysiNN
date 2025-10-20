#!/usr/bin/env python3
"""Generate synthetic spectroscopy datasets with configurable normalisation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch

from physinn.config import LOG_FLOOR, NORMALIZATION, PARAMS
from physinn.datasets import SpectraDataset
from physinn.normalization import unnorm_param_torch
from physinn.physics.spectra import parse_csv_transitions
from physinn.physics.qtpy import Tips2021QTpy
from physinn.ranges import expand_interval, map_ranges


POLY_FREQ_CH4 = [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]
TRANSITIONS_CH4 = """6;1;3085.861015;1.013E-19;0.06;0.078;219.9411;0.73;-0.00712;0.0;0.0221;0.96;0.584;1.12
6;1;3085.832038;1.693E-19;0.0597;0.078;219.9451;0.73;-0.00712;0.0;0.0222;0.91;0.173;1.11
6;1;3085.893769;1.011E-19;0.0602;0.078;219.9366;0.73;-0.00711;0.0;0.0184;1.14;-0.516;1.37
6;1;3086.030985;1.659E-19;0.0595;0.078;219.9197;0.73;-0.00711;0.0;0.0193;1.17;-0.204;0.97
6;1;3086.071879;1.000E-19;0.0585;0.078;219.9149;0.73;-0.00703;0.0;0.0232;1.09;-0.0689;0.82
6;1;3086.085994;6.671E-20;0.055;0.078;219.9133;0.70;-0.00610;0.0;0.0300;0.54;0.00;0.0"""
TRANSITIONS_H2O = """1;2;3083.831748;2.874e-24;0.0971;0.460;78.9886;0.87;-0.00653
1;1;3085.357520;9.562e-25;0.0452;0.282;2254.2838;0.51;0.001433
1;1;3085.506609;1.396e-25;0.0662;0.344;2927.9412;0.63;0.00324
1;1;3085.558839;3.186e-25;0.0491;0.293;2254.2844;0.82;-0.00464
1;1;3085.689600;3.912e-25;0.0508;0.333;2612.7999;0.64;-0.00649
1;1;3086.133208;2.369e-25;0.0457;0.272;2414.7234;0.44;-0.00591
1;1;3087.192118;2.070e-22;0.0768;0.413;648.9787;0.60;-0.00803"""

_BASE_NORMALISATION = {
    "sig0": (3085.43, 3085.46),
    "dsig": (0.001521, 0.00154),
    "mf_CH4": (2e-6, 20e-6),
    "mf_H2O": (0.0059, 0.006),
    "baseline0": (0.99, 1.01),
    "baseline1": (-0.0004, -0.0003),
    "baseline2": (-4.0565e-08, -3.07117e-08),
    "P": (450, 550),
    "T": (273.15 + 32, 273.15 + 37),
}
_EXPAND_FACTORS = {
    "_default": 1.0,
    "sig0": 5.0,
    "dsig": 7.0,
    "mf_CH4": 1.3,
    "mf_H2O": 1.2,
    "baseline0": 1.0,
    "baseline1": 2.0,
    "baseline2": 8.0,
    "P": 1.3,
    "T": 1.3,
}
_DEFAULT_TRAIN = map_ranges(_BASE_NORMALISATION, expand_interval, per_param=_EXPAND_FACTORS)
_DEFAULT_VAL = dict(_BASE_NORMALISATION)
for key in ("mf_CH4", "mf_H2O"):
    lo, hi = _DEFAULT_TRAIN[key]
    _DEFAULT_TRAIN[key] = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR * 10))
    lo_v, hi_v = _DEFAULT_VAL[key]
    _DEFAULT_VAL[key] = (max(lo_v, LOG_FLOOR), max(hi_v, LOG_FLOOR * 10))
_WIDE_TRAIN = map_ranges(_DEFAULT_TRAIN, expand_interval, per_param={"_default": 1.4, "mf_CH4": 1.8, "mf_H2O": 1.5})

NORMALISATION_PRESETS: Dict[str, Mapping[str, tuple[float, float]]] = {
    "train_default": _DEFAULT_TRAIN,
    "train_wide": _WIDE_TRAIN,
    "validation": _DEFAULT_VAL,
}


class ConstantTips:
    """Fallback QTpy/TIPS provider returning constant partition functions."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)

    def q_torch(self, mol_id: int, iso: int, temperature: torch.Tensor) -> torch.Tensor:
        temps = torch.as_tensor(temperature, dtype=torch.float64)
        return torch.full_like(temps, self.value, dtype=torch.float64)


def _load_normalisation(preset: str, custom_file: Path | None) -> Dict[str, tuple[float, float]]:
    if custom_file is not None:
        with open(custom_file, "r") as handle:
            data = json.load(handle)
        normalisation = {k: (float(v[0]), float(v[1])) for k, v in data.items()}
    else:
        key = preset.lower()
        if key not in NORMALISATION_PRESETS:
            raise ValueError(f"Unknown normalisation preset '{preset}'. Available: {sorted(NORMALISATION_PRESETS)}")
        preset_dict = NORMALISATION_PRESETS[key]
        normalisation = {name: (float(lo), float(hi)) for name, (lo, hi) in preset_dict.items()}

    missing = [name for name in PARAMS if name not in normalisation]
    if missing:
        raise ValueError(f"Normalisation ranges missing parameters: {missing}")
    return normalisation


def _load_tipspy(path: Path | None) -> Tips2021QTpy | ConstantTips:
    if path is None:
        return ConstantTips()
    try:
        return Tips2021QTpy(path, device="cpu")
    except FileNotFoundError:
        print(f"[info] QTpy directory '{path}' introuvable, utilisation d'une approximation constante.")
        return ConstantTips()


def _build_transitions() -> Dict[str, Iterable]:
    return {
        "CH4": parse_csv_transitions(TRANSITIONS_CH4),
        "H2O": parse_csv_transitions(TRANSITIONS_H2O),
    }


def _sample_dataset(dataset: SpectraDataset, n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    count = min(n_samples, len(dataset))
    noisy_rows, clean_rows = [], []
    params_phys_rows, params_norm_rows = [], []
    for idx in range(count):
        sample = dataset[idx]
        noisy = sample["noisy_spectra"].cpu().numpy()
        clean = sample["clean_spectra"].cpu().numpy()
        params_norm = sample["params"]
        params_phys = {
            name: float(unnorm_param_torch(name, params_norm[i])) for i, name in enumerate(PARAMS)
        }
        params_phys_rows.append(params_phys)
        params_norm_rows.append({name: float(params_norm[i].item()) for i, name in enumerate(PARAMS)})
        noisy_rows.append(noisy)
        clean_rows.append(clean)

    noisy_df = pd.DataFrame(noisy_rows, columns=[f"v_{i:04d}" for i in range(dataset.num_points)])
    clean_df = pd.DataFrame(clean_rows, columns=[f"v_{i:04d}" for i in range(dataset.num_points)])
    spectra_df = pd.concat({"noisy": noisy_df, "clean": clean_df}, axis=1)
    spectra_df.index.name = "sample"

    phys_df = pd.DataFrame(params_phys_rows)
    norm_df = pd.DataFrame(params_norm_rows)
    params_df = pd.concat({"phys": phys_df, "norm": norm_df}, axis=1)
    params_df.index.name = "sample"
    return spectra_df, params_df


def _plot_spectra_example(spectra_df: pd.DataFrame, output: Path) -> plt.Figure:
    noisy = spectra_df["noisy"].iloc[0]
    clean = spectra_df["clean"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(noisy))
    ax.plot(x, noisy.values, label="Spectre bruité", linewidth=1.2)
    ax.plot(x, clean.values, label="Spectre propre", linewidth=1.2)
    ax.set_xlabel("Indice de point spectral")
    ax.set_ylabel("Transmission")
    ax.set_title("Exemple de spectre simulé")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    return fig


def _plot_param_hist(params_df: pd.DataFrame, output: Path) -> plt.Figure:
    phys_df = params_df["phys"]
    n_params = len(phys_df.columns)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if hasattr(axes, "flatten"):
        flat_axes = axes.flatten()
    else:
        flat_axes = [axes]
    for ax, name in zip(flat_axes, phys_df.columns):
        ax.hist(phys_df[name], bins=20, color="#1f77b4", alpha=0.75)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    for ax in flat_axes[n_params:]:
        ax.axis("off")
    fig.suptitle("Distribution des paramètres physiques", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génération de données spectroscopiques PhysiNN")
    parser.add_argument("--samples", type=int, default=64, help="Nombre d'échantillons à générer pour les DataFrames")
    parser.add_argument("--dataset-size", type=int, default=512, help="Taille du dataset synthétique interne")
    parser.add_argument("--num-points", type=int, default=800, help="Nombre de points spectraux par exemple")
    parser.add_argument(
        "--normalization",
        default="train_default",
        help=f"Preset de normalisation à utiliser ({', '.join(sorted(NORMALISATION_PRESETS))})",
    )
    parser.add_argument(
        "--normalization-file",
        type=Path,
        help="Chemin vers un fichier JSON personnalisé contenant les bornes de normalisation",
    )
    parser.add_argument("--qtpy-dir", type=Path, default=Path("./QTpy"), help="Répertoire QTpy/TIPS 2021")
    parser.add_argument("--output-dir", type=Path, default=Path("./data_examples"), help="Répertoire de sortie")
    parser.add_argument("--seed", type=int, default=123, help="Graine aléatoire PyTorch")
    parser.add_argument("--no-noise", action="store_true", help="Désactive l'ajout de bruit dans les spectres")
    parser.add_argument("--show", action="store_true", help="Affiche les figures générées")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        normalisation = _load_normalisation(args.normalization, args.normalization_file)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))

    NORMALIZATION.clear()
    NORMALIZATION.update(normalisation)

    tipspy = _load_tipspy(args.qtpy_dir)
    transitions = _build_transitions()

    noise_profile = None
    if not args.no_noise:
        noise_profile = dict(
            std_add_range=(0, 1e-3),
            std_mult_range=(0, 1e-3),
            p_drift=0.15,
            drift_sigma_range=(10.0, 80.0),
            drift_amp_range=(0.002, 0.02),
            p_fringes=0.1,
            n_fringes_range=(1, 2),
            fringe_freq_range=(0.3, 25.0),
            fringe_amp_range=(0.001, 0.01),
            p_spikes=0.05,
            spikes_count_range=(1, 4),
            spike_amp_range=(0.001, 0.2),
            spike_width_range=(1.0, 120.0),
            clip=(0.0, 1.1),
        )

    dataset = SpectraDataset(
        n_samples=max(args.dataset_size, args.samples),
        num_points=args.num_points,
        poly_freq_CH4=POLY_FREQ_CH4,
        transitions_dict=transitions,
        sample_ranges=normalisation,
        strict_check=True,
        with_noise=not args.no_noise,
        noise_profile=noise_profile,
        freeze_noise=False,
        tipspy=tipspy,
    )

    spectra_df, params_df = _sample_dataset(dataset, args.samples)

    spectra_path = output_dir / "spectra.csv"
    params_path = output_dir / "parameters.csv"
    spectra_df.to_csv(spectra_path)
    params_df.to_csv(params_path)

    figs: list[plt.Figure] = []
    figs.append(_plot_spectra_example(spectra_df, output_dir / "spectra_example.png"))
    figs.append(_plot_param_hist(params_df, output_dir / "parameters_hist.png"))

    print(f"✓ Données enregistrées : {spectra_path}")
    print(f"✓ Paramètres enregistrés : {params_path}")

    if args.show:
        plt.show()

    for fig in figs:
        plt.close(fig)


if __name__ == "__main__":
    main()
