"""Evaluation utilities for PhysiNN models."""

from __future__ import annotations

import os
import random
from typing import Dict, List

import torch

from .serialization import _to_serializable
from .utils.io import save_fig


def evaluate_and_plot(
    model,
    loader,
    n_show: int = 5,
    refine: bool = True,
    robust_smape: bool = False,
    eps: float = 1e-12,
    seed: int = 123,
    baseline_correction: dict | None = None,
    save_dir: str | None = None,
    tag: str | None = None,
):
    model.eval()
    device = model.device
    rng = random.Random(seed)

    pred_names = list(getattr(model, "predict_params", []))
    if not pred_names:
        raise RuntimeError("No parameters to evaluate.")

    per_param_err: Dict[str, List[torch.Tensor]] = {p: [] for p in pred_names}
    show_examples = []

    for batch in loader:
        noisy = batch["noisy_spectra"].to(device)
        clean = batch["clean_spectra"].to(device)
        params_norm = batch["params"].to(device)
        batch_size = noisy.size(0)

        provided_phys = {}
        if len(model.provided_params) > 0:
            cols = [params_norm[:, model.name_to_idx[n]] for n in model.provided_params]
            provided_norm_tensor = torch.stack(cols, dim=1)
            provided_phys_tensor = model._denorm_params_subset(provided_norm_tensor, model.provided_params)
            for j, name in enumerate(model.provided_params):
                provided_phys[name] = provided_phys_tensor[:, j]

        outputs = model.infer(noisy, provided_phys=provided_phys, refine=refine, resid_target="input")
        recon = outputs["spectra_recon"]
        y_full_pred = outputs["y_phys_full"].clone()

        true_cols = [params_norm[:, model.name_to_idx[n]] for n in pred_names]
        true_norm_tensor = torch.stack(true_cols, dim=1)
        true_phys = model._denorm_params_subset(true_norm_tensor, pred_names)
        pred_phys = torch.stack([y_full_pred[:, model.name_to_idx[n]] for n in pred_names], dim=1)

        if robust_smape:
            denom = pred_phys.abs() + true_phys.abs() + eps
            err_pct = 100.0 * 2.0 * (pred_phys - true_phys).abs() / denom
        else:
            denom = torch.clamp(true_phys.abs(), min=eps)
            err_pct = 100.0 * (pred_phys - true_phys).abs() / denom

        for idx, name in enumerate(pred_names):
            per_param_err[name].append(err_pct[:, idx].detach().cpu())

        for i in range(batch_size):
            if len(show_examples) < n_show:
                show_examples.append(
                    {
                        "noisy": noisy[i].detach().cpu(),
                        "clean": clean[i].detach().cpu(),
                        "recon": recon[i].detach().cpu(),
                    }
                )

    summary = {
        name: torch.cat(err_list).mean().item() if err_list else float("nan") for name, err_list in per_param_err.items()
    }

    save_dir = save_dir or "./eval"
    os.makedirs(save_dir, exist_ok=True)
    tag = tag or "evaluation"
    out_path = os.path.join(save_dir, f"{tag}_metrics.yaml")
    with open(out_path, "w") as handle:
        import yaml

        yaml.dump(_to_serializable(summary), handle)

    for idx, example in enumerate(show_examples):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        x = torch.arange(example["noisy"].shape[-1])
        ax.plot(x, example["noisy"], label="Noisy")
        ax.plot(x, example["clean"], label="Clean")
        ax.plot(x, example["recon"], label="Recon")
        ax.legend()
        ax.set_title(f"Example {idx}")
        save_fig(fig, os.path.join(save_dir, f"{tag}_example_{idx:02d}.png"))

    return summary

