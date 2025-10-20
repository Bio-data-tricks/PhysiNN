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

## Training stages

The training entry point now accepts staged execution. By default it trains only the "A" stage, but you can run the full pipeline (stage A, combined stage B, and a global fine-tuning pass) by specifying the stages to execute:

```bash
python train.py --stages A B FT
```

Additional options allow you to change the root directory for run artefacts (`--runs-base`) and the number of validation samples visualised during training (`--eval-samples`).

## Data generation example

An end-to-end example for dataset synthesis and visualisation is provided under `examples/data_generation.py`. It lets you choose the normalisation preset, generate synthetic spectra, and export both the spectral samples and corresponding parameters as Pandas DataFrames:

```bash
python examples/data_generation.py --samples 128 --normalization train_wide --output-dir ./my_demo
```

The script saves CSV files containing the spectra (`spectra.csv`) and parameters (`parameters.csv`), alongside illustrative plots. Use `--show` to display the figures interactively.
