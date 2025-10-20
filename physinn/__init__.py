"""PhysiNN package regrouping physics-based neural network utilities."""

from .config import (
    PARAMS,
    PARAM_TO_IDX,
    LOG_FLOOR,
    LOG_SCALE_PARAMS,
    NORMALIZATION,
)
from .normalization import norm_param_value, norm_param_torch, unnorm_param_torch
from .datasets import SpectraDataset
from .models.module import PhysicallyInformedAE
from .builders import build_data_and_model

__all__ = [
    "PARAMS",
    "PARAM_TO_IDX",
    "LOG_FLOOR",
    "LOG_SCALE_PARAMS",
    "NORMALIZATION",
    "norm_param_value",
    "norm_param_torch",
    "unnorm_param_torch",
    "SpectraDataset",
    "PhysicallyInformedAE",
    "build_data_and_model",
]
