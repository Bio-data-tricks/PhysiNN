"""Lightning callbacks used across training stages."""

from __future__ import annotations

import os
from typing import List

import torch
import pytorch_lightning as pl

from .lowess import lowess_value
from .utils.distributed import is_rank0
from .utils.io import save_fig


class PlotAndMetricsCallback(pl.Callback):
    def __init__(
        self,
        val_loader,
        param_names,
        num_examples: int = 1,
        save_dir: str | None = None,
        stage_tag: str | None = None,
    ) -> None:
        super().__init__()
        self.val_loader = val_loader
        self.param_names = list(param_names)
        self.num_examples = int(num_examples)
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        default_dir = f"./figs_{job_id}"
        self.stage_tag = stage_tag or "stage"
        self.save_dir = os.path.join(save_dir or default_dir, self.stage_tag)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not is_rank0():
            return
        pl_module.eval()
        device = pl_module.device
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            return

        noisy = batch["noisy_spectra"][: self.num_examples].to(device)
        clean = batch["clean_spectra"][: self.num_examples].to(device)
        params_true_norm = batch["params"][: self.num_examples].to(device)

        provided_phys = {}
        for name in getattr(pl_module, "provided_params", []):
            idx = pl_module.name_to_idx[name]
            v_norm = params_true_norm[:, idx]
            v_phys = pl_module._denorm_params_subset(v_norm.unsqueeze(1), [name])[:, 0]
            provided_phys[name] = v_phys

        outputs = pl_module.infer(noisy, provided_phys=provided_phys, refine=True, resid_target="input")
        spectra_recon = outputs["spectra_recon"].detach().cpu()
        noisy_cpu = noisy.detach().cpu()
        clean_cpu = clean.detach().cpu()

        train_loss = trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item()
        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(0.0)).item()
        val_phys_huber = trainer.callback_metrics.get("val_loss_phys_huber", torch.tensor(0.0)).item()
        val_phys_corr = trainer.callback_metrics.get("val_loss_phys_corr", torch.tensor(0.0)).item()
        val_param_group = trainer.callback_metrics.get("val_loss_param_group", torch.tensor(0.0)).item()

        per_param = []
        for name in pl_module.predict_params:
            metric_name = f"val_loss_param_{name}"
            per_param.append(f"{name:>10s} : {trainer.callback_metrics.get(metric_name, torch.tensor(0.0)).item():.4e}")
        per_param_text = "\n".join(per_param)

        for i in range(min(self.num_examples, noisy_cpu.size(0))):
            import matplotlib.pyplot as plt

            fig, (ax_spec, ax_res, ax_tbl) = plt.subplots(3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 1.5, 1]})
            x = torch.arange(noisy_cpu.shape[-1])
            ax_spec.plot(x, noisy_cpu[i], label="Noisy", lw=1.0)
            ax_spec.plot(x, clean_cpu[i], label="Clean", lw=1.5)
            ax_spec.plot(x, spectra_recon[i], label="Reconstruit", lw=1.2, ls="--")
            ax_spec.set_ylabel("Transmission")
            ax_spec.set_title(f"Epoch {trainer.current_epoch}")
            ax_spec.legend(frameon=False, fontsize=9)

            resid_clean = spectra_recon[i] - clean_cpu[i]
            resid_noisy = spectra_recon[i] - noisy_cpu[i]
            ax_res.plot(x, resid_noisy, lw=1.0, label="Reconstruit - Noisy")
            ax_res.plot(x, resid_clean, lw=1.2, label="Reconstruit - Clean")
            ax_res.axhline(0, ls=":", lw=0.8)
            ax_res.set_xlabel("Points spectraux")
            ax_res.set_ylabel("Résidu")
            ax_res.legend(frameon=False, fontsize=9)

            header = f"Métriques (epoch {trainer.current_epoch})"
            lines = [
                f"train_loss : {train_loss}",
                f"val_loss   : {val_loss}",
                f"val_phys_huber : {val_phys_huber}",
                f"val_corr   : {val_phys_corr}",
                f"val_param_group : {val_param_group}",
                "",
                "Pertes par paramètre (val) :",
                per_param_text,
            ]
            ax_tbl.text(0.02, 0.98, header, va="top", ha="left", fontsize=12, fontweight="bold")
            ax_tbl.text(0.02, 0.90, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
            for axis in (ax_spec, ax_res):
                axis.grid(alpha=0.25)

            out_png = os.path.join(self.save_dir, f"{self.stage_tag}_val_epoch{trainer.current_epoch:04d}.png")
            save_fig(fig, out_png)


class UpdateEpochInDataset(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        dataset = getattr(trainer, "train_dataloaders", trainer.train_dataloader).dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(trainer.current_epoch)


class AdvanceDistributedSamplerEpoch(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)

