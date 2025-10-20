#!/usr/bin/env python3
"""Generate synthetic spectroscopy datasets with configurable normalisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import torch

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None

from physinn.config import LOG_FLOOR, NORMALIZATION, PARAMS
from physinn.datasets import SpectraDataset
from physinn.normalization import unnorm_param_torch
from physinn.physics.spectra import parse_csv_transitions
from physinn.physics.qtpy import Tips2021QTpy
from physinn.ranges import expand_interval, map_ranges

try:  # pragma: no cover - import flexibility for script execution
    from .visualization import plot_param_hist, plot_spectra_example
except ImportError:  # pragma: no cover
    from visualization import plot_param_hist, plot_spectra_example


DEFAULT_CONFIG = Path(__file__).with_name("config").joinpath("spectra_config.yaml")


def _load_spectra_config(config_file: Path | None) -> dict:
    path = config_file or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration YAML introuvable : '{path}'."
        )
    with open(path, "r", encoding="utf8") as handle:
        text = handle.read()
    config = _safe_load_yaml(text)
    if not isinstance(config, dict):
        raise ValueError("Le fichier de configuration doit contenir un dictionnaire YAML.")
    return config


def _extract_poly_coeff(config: Mapping[str, Mapping[str, Iterable[float]]], key: str) -> Iterable[float]:
    poly_cfg = config.get("poly_freq", {})
    if key not in poly_cfg:
        raise KeyError(f"Coefficient de polynôme '{key}' introuvable dans la configuration YAML.")
    coeffs = poly_cfg[key]
    if isinstance(coeffs, str) or not isinstance(coeffs, Iterable):
        raise TypeError(f"Les coefficients '{key}' doivent être une liste de nombres.")
    return [float(value) for value in coeffs]

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
        with open(custom_file, "r", encoding="utf8") as handle:
            text = handle.read()
        if yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
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


def _build_transitions(config: Mapping[str, Mapping[str, str]]) -> Dict[str, Iterable]:
    transitions_cfg = config.get("transitions", {})
    if not transitions_cfg:
        raise ValueError("Aucune transition définie dans la configuration YAML.")
    transitions: Dict[str, Iterable] = {}
    for species, csv_text in transitions_cfg.items():
        if not isinstance(csv_text, str):
            raise TypeError(f"Les transitions pour '{species}' doivent être une chaîne CSV.")
        transitions[species] = parse_csv_transitions(csv_text)
    return transitions


def _safe_load_yaml(text: str) -> dict:
    if yaml is not None:
        return yaml.safe_load(text)

    # Fallback minimal parser supporting the subset of YAML used in this project.
    result: Dict[str, dict] = {}
    current_section: str | None = None
    current_key: str | None = None
    block_key: str | None = None
    block_lines: list[str] | None = None

    def flush_block() -> None:
        nonlocal block_lines, block_key
        if block_lines is not None and block_key is not None and current_section is not None:
            result.setdefault(current_section, {})[block_key] = "\n".join(block_lines)
        block_lines = None
        block_key = None

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        content = raw_line.strip()

        if indent == 0:
            flush_block()
            if not content.endswith(":"):
                raise ValueError(f"Ligne YAML invalide : '{raw_line}'.")
            current_section = content[:-1]
            result.setdefault(current_section, {})
            current_key = None
        elif current_section == "poly_freq":
            if indent == 2 and content.endswith(":"):
                flush_block()
                current_key = content[:-1]
                result.setdefault(current_section, {})[current_key] = []
            elif indent >= 4 and content.startswith("- "):
                if current_key is None:
                    raise ValueError("Valeur de liste sans clé de molécule dans 'poly_freq'.")
                value = float(content[2:])
                result[current_section][current_key].append(value)
            else:
                raise ValueError(f"Structure YAML non prise en charge : '{raw_line}'.")
        elif current_section == "transitions":
            if indent == 2 and content.endswith(": |"):
                flush_block()
                block_key = content[:-3]
                block_lines = []
                result.setdefault(current_section, {})
            elif indent >= 4:
                if block_lines is None or block_key is None:
                    raise ValueError("Bloc de texte inattendu dans 'transitions'.")
                block_lines.append(content)
            else:
                raise ValueError(f"Structure YAML non prise en charge : '{raw_line}'.")
        else:
            raise ValueError(f"Section YAML '{current_section}' non prise en charge sans PyYAML.")

    flush_block()
    return result


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génération de données spectroscopiques PhysiNN")
    parser.add_argument("--samples", type=int, default=64, help="Nombre d'échantillons à générer pour les DataFrames")
    parser.add_argument("--dataset-size", type=int, default=512, help="Taille du dataset synthétique interne")
    parser.add_argument("--num-points", type=int, default=800, help="Nombre de points spectraux par exemple")
    parser.add_argument(
        "--config",
        type=Path,
        help="Chemin vers un fichier YAML contenant les transitions et paramètres spectraux",
    )
    parser.add_argument(
        "--normalization",
        default="train_default",
        help=f"Preset de normalisation à utiliser ({', '.join(sorted(NORMALISATION_PRESETS))})",
    )
    parser.add_argument(
        "--normalization-file",
        type=Path,
        help="Chemin vers un fichier YAML ou JSON personnalisé contenant les bornes de normalisation",
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

    try:
        spectra_config = _load_spectra_config(args.config)
        transitions = _build_transitions(spectra_config)
        poly_freq_ch4 = _extract_poly_coeff(spectra_config, "CH4")
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(str(exc))

    tipspy = _load_tipspy(args.qtpy_dir)

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
        poly_freq_CH4=poly_freq_ch4,
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
    figs.append(plot_spectra_example(spectra_df, output_dir / "spectra_example.png"))
    figs.append(plot_param_hist(params_df, output_dir / "parameters_hist.png"))

    print(f"✓ Données enregistrées : {spectra_path}")
    print(f"✓ Paramètres enregistrés : {params_path}")

    if args.show:
        plt.show()

    for fig in figs:
        plt.close(fig)


if __name__ == "__main__":
    main()
