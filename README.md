# PhysiNN

This repository contains a modular implementation of a physics-informed auto-encoder for spectroscopy modelling. The core package lives under `physinn/` and is organised into focused modules:

- `config.py`, `normalization.py`: shared constants and parameter normalisation utilities.
- `lowess.py`, `noise.py`: signal processing helpers.
- `physics/`: physical constants, QTpy/TIPS interpolation, and spectral synthesis routines.
- `models/`: EfficientNet-based encoder, refiner, and Lightning module definitions.
- `datasets.py`: dataset generation with configurable noise profiles.
- `builders.py`: convenience entry points to construct datasets and the Lightning module.
- `callbacks.py`, `training.py`, `evaluation.py`: orchestration helpers for training workflows.

The `train.py` script demonstrates how to assemble the end-to-end training pipeline using these components.
