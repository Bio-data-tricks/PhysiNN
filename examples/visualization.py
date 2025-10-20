"""Utility functions to visualise simulated spectra and parameter distributions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


__all__ = ["plot_spectra_example", "plot_param_hist"]


def plot_spectra_example(spectra_df: pd.DataFrame, output: Path | str | None = None) -> plt.Figure:
    """Plot an example noisy vs. clean spectrum.

    Parameters
    ----------
    spectra_df:
        DataFrame containing columns ``("noisy", "clean")`` with spectra values.
    output:
        Optional path where the figure should be saved.
    """

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

    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_param_hist(params_df: pd.DataFrame, output: Path | str | None = None, *, columns: Iterable[str] | None = None) -> plt.Figure:
    """Plot histograms for physical parameters.

    Parameters
    ----------
    params_df:
        DataFrame with a top-level column ``"phys"`` containing physical parameters.
    output:
        Optional path where the figure should be saved.
    columns:
        Optional iterable of column names to plot. Defaults to all physical parameters.
    """

    phys_df = params_df["phys"]
    if columns is not None:
        phys_df = phys_df[list(columns)]

    n_params = len(phys_df.columns)
    n_cols = 3
    n_rows = max(1, -(-n_params // n_cols))
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

    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
