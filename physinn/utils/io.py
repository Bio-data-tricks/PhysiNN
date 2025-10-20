"""I/O related helpers for PhysiNN."""

from __future__ import annotations

import os

import matplotlib as mpl

mpl.use("Agg")
try:  # Configure fonts once.
    mpl.rcParams["font.family"] = ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
except Exception:  # pragma: no cover - best effort configuration.
    pass

import matplotlib.pyplot as plt


def save_fig(fig, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_run_dir(base: str = "runs") -> str:
    job = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = os.path.join(base, job)
    os.makedirs(run_dir, exist_ok=True)
    for child in ("checkpoints", "figs", "eval", "logs"):
        os.makedirs(os.path.join(run_dir, child), exist_ok=True)
    return run_dir

