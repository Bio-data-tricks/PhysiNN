"""Entry point for PhysiNN training pipeline."""

from __future__ import annotations

import os

import pytorch_lightning as pl
import torch

from physinn.builders import build_data_and_model, trainer_common_kwargs
from physinn.callbacks import PlotAndMetricsCallback, UpdateEpochInDataset
from physinn.evaluation import evaluate_and_plot
from physinn.training import train_stage_A
from physinn.utils.distributed import get_master_addr_and_port, on_rank_zero
from physinn.utils.io import make_run_dir


def main() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pl.seed_everything(42, workers=True)
    get_master_addr_and_port()

    run_dir = make_run_dir(base="runs")
    model, train_loader, val_loader = build_data_and_model(
        backbone_variant="m",
        backbone_width_mult=1.0,
        backbone_depth_mult=1.0,
        refiner_variant="m",
        refiner_width_mult=1.0,
        refiner_depth_mult=1.0,
        refiner_feature_pool="avg",
        refiner_shared_hidden_scale=0.5,
        huber_beta=0.002,
        backbone_drop_path=0.0,
        refiner_drop_path=0.0,
    )

    figs_root = os.path.join(run_dir, "figs")
    eval_dir = os.path.join(run_dir, "eval")

    trainer_kwargs = trainer_common_kwargs()
    trainer_kwargs["default_root_dir"] = os.path.join(run_dir, "logs")

    callbacks = [
        PlotAndMetricsCallback(val_loader, model.param_names, num_examples=1, save_dir=figs_root, stage_tag="stage_A"),
        UpdateEpochInDataset(),
    ]

    ckpt_A = os.path.join(run_dir, "checkpoints", "stage_A.ckpt")
    model = train_stage_A(
        model,
        train_loader,
        val_loader,
        epochs=100,
        base_lr=1e-4,
        train_film=False,
        use_film=False,
        film_subset=[],
        heads_subset=["sig0", "dsig", "P", "T", "mf_CH4", "baseline1", "baseline2"],
        callbacks=callbacks,
        ckpt_out=ckpt_A,
        **trainer_kwargs,
    )

    if on_rank_zero():
        evaluate_and_plot(
            model,
            val_loader,
            n_show=5,
            refine=True,
            robust_smape=False,
            baseline_correction={"enabled": False, "edge_pts": 50, "deg": 2, "iters": 1},
            save_dir=eval_dir,
            tag="stage_A",
        )


if __name__ == "__main__":
    main()

