"""Training stage orchestration for PhysiNN."""

from __future__ import annotations

from typing import List, Optional

import pytorch_lightning as pl
import torch

# callbacks are intentionally imported by the caller modules


def _freeze_all(module: pl.LightningModule) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def _set_trainable_heads(model, names: Optional[List[str]]) -> None:
    if model.head_mode == "single":
        for param in model.out_head.parameters():
            param.requires_grad_(True)
        return
    desired = set(names) if names is not None else set(model.predict_params)
    for name, head in model.out_heads.items():
        required = name in desired
        for param in head.parameters():
            param.requires_grad_(required)


def _apply_stage_freeze(
    model,
    *,
    train_base: bool,
    train_heads: bool,
    train_film: bool,
    train_refiner: bool,
    heads_subset: Optional[List[str]],
) -> None:
    _freeze_all(model)
    if train_base:
        for param in model.backbone.parameters():
            param.requires_grad_(True)
        for param in model.shared_head.parameters():
            param.requires_grad_(True)
    if model.head_mode == "single":
        if train_heads:
            for param in model.out_head.parameters():
                param.requires_grad_(True)
    else:
        _set_trainable_heads(model, heads_subset if train_heads else [])
    if model.film is not None and train_film:
        for param in model.film.parameters():
            param.requires_grad_(True)
    if train_refiner:
        for param in model.refiner.parameters():
            param.requires_grad_(True)


def train_stage_custom(
    model: pl.LightningModule,
    train_loader,
    val_loader,
    *,
    stage_name: str,
    epochs: int,
    base_lr: float,
    refiner_lr: float,
    train_base: bool,
    train_heads: bool,
    train_film: bool,
    train_refiner: bool,
    refine_steps: int,
    delta_scale: float,
    use_film: Optional[bool] = None,
    film_subset: Optional[List[str]] = None,
    heads_subset: Optional[List[str]] = None,
    callbacks: Optional[list] = None,
    enable_progress_bar: bool = False,
    **trainer_kwargs,
):
    print(f"\n===== Stage {stage_name} =====")
    ckpt_in = trainer_kwargs.pop("ckpt_in", None)
    ckpt_out = trainer_kwargs.pop("ckpt_out", None)

    if ckpt_in:
        state = torch.load(ckpt_in, map_location="cpu")
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ weights loaded from: {ckpt_in}")

    if use_film is not None:
        model.set_film_usage(bool(use_film))
    if film_subset is not None:
        model.set_film_subset(film_subset)

    model.set_stage_mode(stage_name, refine_steps=refine_steps, delta_scale=delta_scale)
    _apply_stage_freeze(
        model,
        train_base=train_base,
        train_heads=train_heads,
        train_film=train_film,
        train_refiner=train_refiner,
        heads_subset=heads_subset,
    )

    trainer_kwargs.setdefault("log_every_n_steps", 1)

    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks or [],
        **trainer_kwargs,
    )

    trainer.fit(model, train_loader, val_loader)
    if ckpt_out:
        trainer.save_checkpoint(ckpt_out)
        print(f"✓ checkpoint saved to: {ckpt_out}")
    return model


def train_stage_A(model, train_loader, val_loader, **kwargs):
    defaults = dict(
        stage_name="A",
        epochs=20,
        base_lr=2e-4,
        refiner_lr=1e-6,
        train_base=True,
        train_heads=True,
        train_film=False,
        train_refiner=False,
        refine_steps=0,
        delta_scale=0.1,
        use_film=False,
        film_subset=None,
        heads_subset=None,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)


def train_stage_B1(model, train_loader, val_loader, **kwargs):
    defaults = dict(
        stage_name="B1",
        epochs=12,
        base_lr=1e-6,
        refiner_lr=1e-5,
        train_base=False,
        train_heads=False,
        train_film=False,
        train_refiner=True,
        refine_steps=2,
        delta_scale=0.12,
        use_film=True,
        film_subset=["T"],
        heads_subset=None,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)


def train_stage_B2(model, train_loader, val_loader, **kwargs):
    defaults = dict(
        stage_name="B2",
        epochs=15,
        base_lr=3e-5,
        refiner_lr=3e-6,
        train_base=True,
        train_heads=True,
        train_film=True,
        train_refiner=True,
        refine_steps=2,
        delta_scale=0.08,
        use_film=True,
        film_subset=["P", "T"],
        heads_subset=None,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return train_stage_custom(model, train_loader, val_loader, **defaults)

