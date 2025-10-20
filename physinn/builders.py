"""High level helpers to assemble datasets and models."""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler as _DistributedSampler

from .config import LOG_FLOOR, NORMALIZATION, PARAMS
from .datasets import SpectraDataset
from .models.module import PhysicallyInformedAE
from .physics.qtpy import Tips2021QTpy
from .physics.spectra import parse_csv_transitions
from .ranges import assert_subset, expand_interval, map_ranges


def ensure_distributed_samplers(train_loader, val_loader):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return train_loader, val_loader
    return _with_sampler(train_loader, True), _with_sampler(val_loader, False)


def _with_sampler(loader, shuffle_default):
    if loader is None:
        return None
    if isinstance(getattr(loader, "sampler", None), _DistributedSampler):
        return loader
    dataset = loader.dataset
    sampler = _DistributedSampler(dataset, shuffle=shuffle_default)
    return DataLoader(
        dataset,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
        sampler=sampler,
        shuffle=False,
    )


def trainer_common_kwargs():
    import pytorch_lightning as pl

    return dict(
        accelerator="auto",
        strategy="auto",
        devices="auto",
        precision="32-true",
        accumulate_grad_batches=1,
        deterministic=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )


def build_data_and_model(
    *,
    seed=42,
    n_points=800,
    n_train=500000,
    n_val=5000,
    batch_size=16,
    train_ranges=None,
    val_ranges=None,
    noise_train=None,
    noise_val=None,
    predict_list=None,
    film_list=None,
    lrs=(1e-4, 1e-5),
    backbone_variant="s",
    refiner_variant="s",
    backbone_width_mult=1.0,
    backbone_depth_mult=1.0,
    refiner_width_mult=1.0,
    refiner_depth_mult=1.0,
    backbone_stem_channels=None,
    refiner_stem_channels=None,
    backbone_drop_path=0.0,
    refiner_drop_path=0.0,
    backbone_se_ratio=0.25,
    refiner_se_ratio=0.25,
    refiner_feature_pool="avg",
    refiner_shared_hidden_scale=0.5,
    refiner_time_embed_dim=None,
    huber_beta=0.002,
    qtpy_dir: str | None = None,
):
    import pytorch_lightning as pl

    pl.seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_windows = sys.platform == "win32"
    num_workers = 0 if is_windows else 4

    qtpy_dir = qtpy_dir or os.environ.get("QTPY_DIR", "./QTpy")
    tipspy = Tips2021QTpy(qtpy_dir, device="cpu")

    poly_freq_CH4 = [-2.3614803e-07, 1.2103413e-10, -3.1617856e-14]
    transitions_ch4_str = """6;1;3085.861015;1.013E-19;0.06;0.078;219.9411;0.73;-0.00712;0.0;0.0221;0.96;0.584;1.12
6;1;3085.832038;1.693E-19;0.0597;0.078;219.9451;0.73;-0.00712;0.0;0.0222;0.91;0.173;1.11
6;1;3085.893769;1.011E-19;0.0602;0.078;219.9366;0.73;-0.00711;0.0;0.0184;1.14;-0.516;1.37
6;1;3086.030985;1.659E-19;0.0595;0.078;219.9197;0.73;-0.00711;0.0;0.0193;1.17;-0.204;0.97
6;1;3086.071879;1.000E-19;0.0585;0.078;219.9149;0.73;-0.00703;0.0;0.0232;1.09;-0.0689;0.82
6;1;3086.085994;6.671E-20;0.055;0.078;219.9133;0.70;-0.00610;0.0;0.0300;0.54;0.00;0.0"""
    transitions_h2o_str = """1;2;3083.831748;2.874e-24;0.0971;0.460;78.9886;0.87;-0.00653
1;1;3085.357520;9.562e-25;0.0452;0.282;2254.2838;0.51;0.001433
1;1;3085.506609;1.396e-25;0.0662;0.344;2927.9412;0.63;0.00324
1;1;3085.558839;3.186e-25;0.0491;0.293;2254.2844;0.82;-0.00464
1;1;3085.689600;3.912e-25;0.0508;0.333;2612.7999;0.64;-0.00649
1;1;3086.133208;2.369e-25;0.0457;0.272;2414.7234;0.44;-0.00591
1;1;3087.192118;2.070e-22;0.0768;0.413;648.9787;0.60;-0.00803"""

    transitions_dict = {
        "CH4": parse_csv_transitions(transitions_ch4_str),
        "H2O": parse_csv_transitions(transitions_h2o_str),
    }

    default_val = {
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
    expand_factors = {
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
    default_train = map_ranges(default_val, expand_interval, per_param=expand_factors)

    lo, hi = default_train["mf_CH4"]
    default_train["mf_CH4"] = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR * 10))
    lo, hi = default_val["mf_CH4"]
    default_val["mf_CH4"] = (max(lo, LOG_FLOOR), max(hi, LOG_FLOOR * 10))

    val_ranges = val_ranges or default_val
    train_ranges = train_ranges or default_train
    assert_subset(val_ranges, train_ranges, "VAL", "TRAIN")
    NORMALIZATION.update(train_ranges)

    noise_train = noise_train or dict(
        std_add_range=(0, 1e-3),
        std_mult_range=(0, 1e-3),
        p_drift=0.2,
        drift_sigma_range=(10.0, 120.0),
        drift_amp_range=(0.004, 0.05),
        p_fringes=0.2,
        n_fringes_range=(1, 2),
        fringe_freq_range=(0.3, 50.0),
        fringe_amp_range=(0.001, 0.015),
        p_spikes=0.2,
        spikes_count_range=(1, 6),
        spike_amp_range=(0.002, 1),
        spike_width_range=(1.0, 200.0),
        clip=(0.0, 1.1),
    )
    noise_val = noise_val or dict(
        std_add_range=(0, 1e-5),
        std_mult_range=(0, 1e-5),
        p_drift=0,
        drift_sigma_range=(20.0, 120.0),
        drift_amp_range=(0.0, 0.01),
        p_fringes=0,
        n_fringes_range=(1, 2),
        fringe_freq_range=(0.5, 10.0),
        fringe_amp_range=(0.0, 0.004),
        p_spikes=0.0,
        spikes_count_range=(1, 2),
        spike_amp_range=(0.0, 0.01),
        spike_width_range=(1.0, 3.0),
        clip=(0.0, 1.1),
    )

    dataset_train = SpectraDataset(
        n_samples=n_train,
        num_points=n_points,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=train_ranges,
        strict_check=True,
        with_noise=True,
        noise_profile=noise_train,
        freeze_noise=False,
        tipspy=tipspy,
    )
    dataset_val = SpectraDataset(
        n_samples=n_val,
        num_points=n_points,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        sample_ranges=val_ranges,
        strict_check=True,
        with_noise=True,
        noise_profile=noise_val,
        freeze_noise=True,
        tipspy=tipspy,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    predict_list = predict_list or ["sig0", "dsig", "mf_CH4", "P", "T", "baseline1", "baseline2"]
    film_list = film_list or []

    model = PhysicallyInformedAE(
        n_points=n_points,
        param_names=PARAMS,
        poly_freq_CH4=poly_freq_CH4,
        transitions_dict=transitions_dict,
        lr=lrs[0],
        alpha_param=0.3,
        alpha_phys=0.7,
        head_mode="multi",
        predict_params=predict_list,
        film_params=film_list,
        refine_steps=1,
        refine_delta_scale=0.1,
        refine_target="noisy",
        refine_warmup_epochs=30,
        freeze_base_epochs=20,
        base_lr=lrs[0],
        refiner_lr=lrs[1],
        recon_max1=True,
        corr_mode="none",
        corr_savgol_win=15,
        corr_savgol_poly=3,
        huber_beta=huber_beta,
        backbone_variant=backbone_variant,
        refiner_variant=refiner_variant,
        backbone_width_mult=backbone_width_mult,
        backbone_depth_mult=backbone_depth_mult,
        refiner_width_mult=refiner_width_mult,
        refiner_depth_mult=refiner_depth_mult,
        backbone_stem_channels=backbone_stem_channels,
        refiner_stem_channels=refiner_stem_channels,
        backbone_drop_path=backbone_drop_path,
        refiner_drop_path=refiner_drop_path,
        backbone_se_ratio=backbone_se_ratio,
        refiner_se_ratio=refiner_se_ratio,
        refiner_feature_pool=refiner_feature_pool,
        refiner_shared_hidden_scale=refiner_shared_hidden_scale,
        refiner_time_embed_dim=refiner_time_embed_dim,
        ranges_train=train_ranges,
        ranges_val=val_ranges,
        noise_train=noise_train,
        noise_val=noise_val,
        tipspy=tipspy,
    )

    model.hparams.optimizer = "lion"
    model.hparams.betas = (0.9, 0.99)
    model.weight_decay = 1e-4

    def _serializable_ranges(d):
        return {k: [float(d[k][0]), float(d[k][1])] for k in d}

    extra_hparams = {
        "ranges_train": _serializable_ranges(train_ranges),
        "ranges_val": _serializable_ranges(val_ranges),
        "noise_train": noise_train,
        "noise_val": noise_val,
    }

    try:
        model.save_hyperparameters(extra_hparams)
    except Exception:
        if hasattr(model, "hparams"):
            for k, v in extra_hparams.items():
                setattr(model.hparams, k, v)

    return model, train_loader, val_loader

