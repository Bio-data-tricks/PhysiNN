"""Entry point for PhysiNN training pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch

from physinn.builders import build_data_and_model, trainer_common_kwargs
from physinn.callbacks import PlotAndMetricsCallback, UpdateEpochInDataset
from physinn.evaluation import evaluate_and_plot
from physinn.training import fine_tune_global, train_A, train_B
from physinn.utils.distributed import get_master_addr_and_port, on_rank_zero
from physinn.utils.io import make_run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhysiNN staged training pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["A"],
        help="Ordered list of stages to run (choose from: A, B, FT or fine_tune_global).",
    )
    parser.add_argument(
        "--runs-base",
        default="runs",
        help="Base directory where run folders are created (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=5,
        help="Number of validation examples to visualise during callbacks and evaluation.",
    )
    args = parser.parse_args()

    mapping = {"A": "A", "STAGE_A": "A", "B": "B", "STAGE_B": "B", "FT": "FT", "FINE_TUNE": "FT", "FINE_TUNE_GLOBAL": "FT"}
    normalised: list[str] = []
    for entry in args.stages:
        token = entry.strip().upper().replace("-", "_")
        if token not in mapping:
            parser.error(f"Unknown stage '{entry}'. Valid options are A, B, FT (fine_tune_global).")
        stage = mapping[token]
        if stage not in normalised:
            normalised.append(stage)
    args.stages = normalised or ["A"]
    return args


def main() -> None:
    args = _parse_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pl.seed_everything(42, workers=True)
    get_master_addr_and_port()

    run_dir = make_run_dir(base=args.runs_base)
    ckpt_dir = Path(run_dir) / "checkpoints"
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

    num_examples = max(1, args.eval_samples)
    last_checkpoint: str | None = None

    if "A" in args.stages:
        callbacks_a = [
            PlotAndMetricsCallback(
                val_loader,
                model.param_names,
                num_examples=num_examples,
                save_dir=figs_root,
                stage_tag="stage_A",
            ),
            UpdateEpochInDataset(),
        ]

        ckpt_a = ckpt_dir / "stage_A.ckpt"
        model = train_A(
            model,
            train_loader,
            val_loader,
            epochs=100,
            base_lr=1e-4,
            train_film=False,
            use_film=False,
            film_subset=[],
            heads_subset=["sig0", "dsig", "P", "T", "mf_CH4", "baseline1", "baseline2"],
            callbacks=callbacks_a,
            ckpt_out=str(ckpt_a),
            **trainer_kwargs,
        )
        last_checkpoint = str(ckpt_a)

        if on_rank_zero():
            evaluate_and_plot(
                model,
                val_loader,
                n_show=num_examples,
                refine=True,
                robust_smape=False,
                baseline_correction={"enabled": False, "edge_pts": 50, "deg": 2, "iters": 1},
                save_dir=eval_dir,
                tag="stage_A",
            )

    if "B" in args.stages:
        callbacks_b1 = [
            PlotAndMetricsCallback(
                val_loader,
                model.param_names,
                num_examples=num_examples,
                save_dir=figs_root,
                stage_tag="stage_B1",
            ),
            UpdateEpochInDataset(),
        ]
        callbacks_b2 = [
            PlotAndMetricsCallback(
                val_loader,
                model.param_names,
                num_examples=num_examples,
                save_dir=figs_root,
                stage_tag="stage_B2",
            ),
            UpdateEpochInDataset(),
        ]

        ckpt_in = last_checkpoint if last_checkpoint and os.path.exists(last_checkpoint) else None
        model = train_B(
            model,
            train_loader,
            val_loader,
            ckpt_in=ckpt_in,
            ckpt_dir=ckpt_dir,
            callbacks_b1=callbacks_b1,
            callbacks_b2=callbacks_b2,
            **trainer_kwargs,
        )
        last_checkpoint = str(ckpt_dir / "stage_B2.ckpt")

        if on_rank_zero():
            evaluate_and_plot(
                model,
                val_loader,
                n_show=num_examples,
                refine=True,
                robust_smape=False,
                baseline_correction={"enabled": False, "edge_pts": 50, "deg": 2, "iters": 1},
                save_dir=eval_dir,
                tag="stage_B",
            )

    if "FT" in args.stages:
        callbacks_ft = [
            PlotAndMetricsCallback(
                val_loader,
                model.param_names,
                num_examples=num_examples,
                save_dir=figs_root,
                stage_tag="fine_tune",
            ),
            UpdateEpochInDataset(),
        ]

        ckpt_ft = ckpt_dir / "fine_tune.ckpt"
        ckpt_in = last_checkpoint if last_checkpoint and os.path.exists(last_checkpoint) else None
        model = fine_tune_global(
            model,
            train_loader,
            val_loader,
            callbacks=callbacks_ft,
            ckpt_in=ckpt_in,
            ckpt_out=str(ckpt_ft),
            **trainer_kwargs,
        )
        last_checkpoint = str(ckpt_ft)

        if on_rank_zero():
            evaluate_and_plot(
                model,
                val_loader,
                n_show=num_examples,
                refine=True,
                robust_smape=False,
                baseline_correction={"enabled": False, "edge_pts": 50, "deg": 2, "iters": 1},
                save_dir=eval_dir,
                tag="fine_tune",
            )


if __name__ == "__main__":
    main()

