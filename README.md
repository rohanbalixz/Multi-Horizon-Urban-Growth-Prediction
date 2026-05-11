<h1 align="center">Multi-Horizon Urban Growth Prediction</h1>

<p align="center">
  Spatio-temporal deep learning for continental-scale urban built-up density forecasting,
  <br>with calibrated per-pixel uncertainty over the Continental United States.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.10%2B-red.svg">
  <img alt="Data: GHSL R2023A" src="https://img.shields.io/badge/Data-GHSL%20R2023A-green.svg">
  <a href="#citation"><img alt="Cite this work" src="https://img.shields.io/badge/Cite-this%20work-orange.svg"></a>
</p>

<p align="center">
  <b>Author:</b> Rohan Bali
  &nbsp;·&nbsp;
  <b>Contact:</b>
  <a href="mailto:rohanbaliwork@gmail.com">rohanbaliwork@gmail.com</a>
  &nbsp;|&nbsp;
  <a href="mailto:rbali@umassd.edu">rbali@umassd.edu</a>
</p>

---

<a id="tldr"></a>

## TL;DR

A six-model benchmark (CNN, U-Net, ConvLSTM with Monte Carlo Dropout, linear extrapolation, SLEUTH cellular automata, persistence) for predicting urban built-up surface density across the Continental United States (CONUS) at 250 m resolution from 45 years of Global Human Settlement Layer (GHSL) observations. The repository contains the full experimental pipeline behind two independent holdouts (821-tile 2015 spatial block, fully sealed 2020 temporal), four change-threshold sensitivities, channel-matched multi-horizon experiments, and MC Dropout calibration on 8.68 million validation pixels. Trained weights, raw rasters, and the manuscript are intentionally not committed; see [Data](#data) for download instructions.

<p align="center">
  <img src="results/figures/fig_title_maps.png" alt="Representative validation tile">
  <br><em>Representative validation tile from the 821-tile 2015 spatial holdout: GHSL ground truth, ConvLSTM prediction, and MC Dropout uncertainty (20 stochastic forward passes).</em>
</p>

<a id="table-of-contents"></a>

## Table of Contents

1. [Headline results](#headline-results)
2. [Repository contents](#repository-contents)
3. [Quick start](#quick-start)
4. [Experiments](#experiments)
5. [Architecture](#architecture)
6. [Data](#data)
7. [Installation](#installation)
8. [Reproducing all experiments](#reproducing-all-experiments)
9. [Evaluation protocol](#evaluation-protocol)
10. [Hardware and runtime](#hardware-and-runtime)
11. [Pre-trained weights](#pre-trained-weights)
12. [Citation](#citation)
13. [Acknowledgments](#acknowledgments)
14. [License](#license)

<a id="headline-results"></a>

## Headline results

### 2015 spatial block holdout

Pixel-level Figure of Merit (FoM) on 1,542,747 evaluation pixels (`gt > 0.01` mask, 25.1% growth). Mean ± std across two random seeds for the neural models.

| Model                     | FoM (↑)         | Parameters |
|---------------------------|------------------|------------|
| SimpleCNN (flat stacking) | 0.702 ± 0.019   | 74 K       |
| U-Net                     | 0.676            | 474 K      |
| ConvLSTM + MC Dropout     | 0.508 ± 0.0001  | 481 K      |
| Linear extrapolation      | 0.375            | n/a        |
| SLEUTH                    | 0.056            | 4 (hyperparameters) |
| Persistence               | 0.000            | n/a        |

> **Figure of Merit** (Pontius et al., 2008): FoM = B / (A + B + C), where B is correctly predicted growth, A is missed growth, and C is false growth. True negatives are excluded from the denominator, making FoM robust to the 89% stable background of CONUS, a regime in which MSE structurally favors a no-change predictor.

### 2020 blind temporal holdout

Pre-2015 training only. GHSL 2020 was downloaded after all model selection had finished. Evaluation domain (`gt > 0.01`) is identical across the six models: 1,574,225 pixels with 10.57% growth.

| Model                 | FoM (↑)        | MSE (↓)        |
|-----------------------|-----------------|------------------|
| SimpleCNN             | 0.252 ± 0.012  | 4.90 × 10⁻⁴    |
| U-Net                 | 0.229           | 4.30 × 10⁻⁴    |
| ConvLSTM + MC Dropout | 0.156           | 4.84 × 10⁻⁴    |
| SLEUTH                | 0.121           | n/a              |
| Linear extrapolation  | 0.053           | 9.40 × 10⁻⁴    |
| Persistence           | 0.000           | 2.40 × 10⁻⁴ (lowest) |

Persistence achieves the lowest MSE because 89% of evaluated pixels are stable; this metric inversion is the reason FoM is reported as the primary metric for sparse-change spatial tasks.

<p align="center">
  <img src="results/figures/fig_paper_fom_comparison.png" alt="Headline FoM comparison and threshold sensitivity">
  <br><em>(a) 2015 spatial block holdout. (b) 2020 blind temporal holdout. (c) CNN vs. ConvLSTM FoM gap across four change thresholds on both holdouts. CNN leads at every threshold and on every holdout.</em>
</p>

### Statistical confirmation

A paired Wilcoxon signed-rank test on per-tile FoM values across all 821 validation tiles confirms the model ranking. Bootstrap 95% confidence intervals are reported on the per-tile FoM gap (n_bootstrap = 2000).

| Test                       | Median Δ FoM | Wilcoxon p-value     | Bootstrap 95% CI on Δ FoM |
|----------------------------|---------------|------------------------|----------------------------|
| CNN > ConvLSTM             | +0.137         | 3.0 × 10⁻¹³⁶          | [0.142, 0.152]              |
| U-Net > ConvLSTM           | +0.118         | 5.9 × 10⁻¹³⁶          | [0.126, 0.135]              |
| CNN > U-Net                | +0.016         | 2.3 × 10⁻⁶³            | n/a                          |

### Multi-horizon analysis

<p align="center">
  <img src="results/figures/fig_multihorizon.png" alt="Multi-horizon FoM with the channel-matched CNN control">
  <br><em>(a) ConvLSTM and SimpleCNN FoM as a function of forecast horizon, with the channel-matched CNN control (open markers) isolating the true horizon effect from the channel-count confound. (b) Decomposition of the 5-year to 10-year gap compression.</em>
</p>

The apparent ConvLSTM catch-up at longer horizons is mostly a channel-count confound, not a representational benefit. When input channel count is held fixed, the CNN advantage shrinks by less than 0.015 FoM across horizons.

### Calibrated uncertainty

<p align="center">
  <img src="results/figures/fig_calibration.png" alt="MC Dropout calibration plot">
  <br><em>Predicted standard deviation σ̂ from 20 MC Dropout forward passes versus mean absolute error, binned into 10 equal-count deciles of 8.68 M pixels each. Pearson r = 0.9834; Spearman ρ = 1.000.</em>
</p>

Every higher-uncertainty decile bin has strictly larger mean actual error than the bin below it. The ConvLSTM model is the only one in the benchmark producing calibrated spatial uncertainty.

<a id="repository-contents"></a>

## Repository contents

| Path              | Purpose                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `src/models/`     | ConvLSTM (with MC Dropout) and SLEUTH cellular automaton                 |
| `src/utils/`      | Training loop, inference, evaluation metrics (FoM, growth decomposition) |
| `scripts/`        | One driver per experiment (preprocessing, training, evaluation, figures) |
| `tests/`          | Unit tests for the model and metric implementations                      |
| `results/metrics/`| Final JSON outputs of every experiment used in the analysis              |
| `results/figures/`| Final PDF and PNG figures generated from `results/metrics/`              |
| `docs/`           | Installation guide and model card                                        |

Local-only directories that are not pushed: `data/`, `geotiff_exports/`, `models/`, `paper/`, `logs/`, `notebooks/`.

<a id="quick-start"></a>

## Quick start

After [installation](#installation), the following snippet verifies the ConvLSTM model loads and runs a forward pass on a dummy batch:

```python
import torch
from src.models.convlstm import ConvLSTM

model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1)
x = torch.randn(1, 8, 3, 128, 128)        # batch, time, channels, H, W
y = model(x)
print(y.shape)                             # torch.Size([1, 1, 128, 128])
```

A full training run takes roughly 24 hours on a single A100 (see [Hardware and runtime](#hardware-and-runtime)).

<a id="experiments"></a>

## Experiments

Each experiment is implemented as a single self-contained script under `scripts/`. The list below maps each experimental claim to the script that produces it.

### 1. Main 2015 spatial-block holdout

CONUS is split into 1280 × 1280 pixel geographic blocks of roughly 320 km². Every fifth block is held out, producing 821 validation tiles that share no pixels with the training set.

| Script                                 | Produces                                                    |
|----------------------------------------|-------------------------------------------------------------|
| `scripts/preprocess_all_data.py`       | Resamples and reprojects GHSL rasters into `geotiff_exports/` |
| `scripts/train_3channel.py`            | ConvLSTM + MC Dropout (seed 42)                              |
| `scripts/train_convlstm_seed0.py`      | ConvLSTM + MC Dropout (seed 0)                               |
| `scripts/cnn_multiseed.py`             | Flat-stacking CNN (seeds 0 and 42)                           |
| `scripts/train_unet_seed0.py`          | U-Net (seeds 0 and 42)                                       |
| `scripts/train_sota_baselines.py`      | SLEUTH cellular automaton calibration                        |
| `scripts/validate_2015.py`             | Aggregated 2015 pixel-level FoM, MSE, MAE, R²                |

### 2. 2020 blind temporal holdout

GHSL 2020 was downloaded only after all model selection on the 2015 holdout had completed. The input window is shifted to [1980, 2015] with target 2020 and every model is evaluated on the same 1.57 M-pixel domain.

| Script                                 | Produces                                                              |
|----------------------------------------|-----------------------------------------------------------------------|
| `scripts/preprocess_2020.py`           | Builds the 2020 evaluation tensors                                    |
| `scripts/validate_2020.py`             | ConvLSTM, linear extrapolation, persistence                            |
| `scripts/cnn_2020_holdout.py`          | CNN multi-seed on 2020                                                 |
| `scripts/unet_2020_holdout.py`         | U-Net on 2020                                                          |
| `scripts/unet_2020_pixel_filtered.py`  | U-Net 2020 recomputed on the `gt > 0.01` mask used by all other models |
| `scripts/sleuth_2020_holdout.py`       | SLEUTH on 2020                                                         |

### 3. Multi-horizon forecasting

All experiments predict the 2015 epoch. Forecast horizon is varied by truncating the input window so the last observed epoch is 2010 (5-year), 2005 (10-year), or 1995 (20-year).

| Script                                 | Produces                                                                  |
|----------------------------------------|---------------------------------------------------------------------------|
| `scripts/train_multihorizon.py`        | ConvLSTM at 5, 10, and 20 year horizons                                    |
| `scripts/run_cnn_multihorizon.py`      | CNN at 5, 10, and 20 year horizons                                         |
| `scripts/run_cnn_channel_control.py`   | CNN with input channel count fixed; isolates true horizon from channel confound |
| `scripts/reeval_multihorizon.py`       | Re-evaluates every horizon checkpoint under the unified pixel mask        |
| `scripts/make_multihorizon_figure.py`  | Generates `fig_multihorizon.{pdf,png}`                                    |

### 4. Ablations

| Script                                | Variable                                                       |
|---------------------------------------|----------------------------------------------------------------|
| `scripts/run_ablations_3ch.py`        | Sequence length and channel ablations                          |
| `scripts/channel_attribution_ci.py`   | Bootstrap confidence intervals on the channel-attribution effect |

### 5. Threshold sensitivity

| Script                                | Produces                                                                       |
|---------------------------------------|--------------------------------------------------------------------------------|
| `scripts/threshold_sensitivity.py`    | FoM at change thresholds t ∈ {0.005, 0.01, 0.02, 0.05} on both holdouts        |

### 6. Per-tile statistics

| Script                                | Produces                                                                |
|---------------------------------------|-------------------------------------------------------------------------|
| `scripts/compute_pertile_fom.py`      | Per-tile FoM, paired Wilcoxon signed-rank test, bootstrap 95% CIs        |
| `scripts/gen_pertile_scatter.py`      | Per-tile scatter plot                                                    |

### 7. Calibrated uncertainty

| Script                                  | Produces                                                                 |
|-----------------------------------------|--------------------------------------------------------------------------|
| `scripts/generate_uncertainty_maps.py`  | MC Dropout uncertainty maps over the validation tiles (20 forward passes) |

### 8. Cross-region transfer

| Script                                 | Region                                                |
|----------------------------------------|-------------------------------------------------------|
| `scripts/eval_lagos_full.py`           | Zero-shot transfer evaluation to Lagos, Nigeria       |
| `scripts/eval_lagos_transfer.py`       | Zero-shot transfer evaluation to Lagos, Nigeria       |

### 9. Figure generation

| Script                                  | Produces                                                |
|-----------------------------------------|---------------------------------------------------------|
| `scripts/generate_figures.py`           | Re-creates every figure under `results/figures/`         |
| `scripts/make_architecture_figure.py`   | ConvLSTM architecture diagram                            |
| `scripts/make_architecture_cnn.py`      | CNN architecture diagram                                 |
| `scripts/make_architecture_unet.py`     | U-Net architecture diagram                               |
| `scripts/make_paper_fom_figure.py`      | Headline FoM comparison panel                            |

<a id="architecture"></a>

## Architecture

<p align="center">
  <img src="results/figures/fig_architecture.png" alt="ConvLSTM + MC Dropout architecture diagram">
  <br><em>Two stacked ConvLSTM layers (hidden 64, kernel 3 × 3) with MC Dropout p = 0.1, followed by a skip-connection decoder of four Conv + BN + ReLU blocks ending in a 1 × 1 sigmoid head.</em>
</p>

The primary model has 481,153 parameters. Input tensor shape is T = 8 time steps × 3 channels × 128 × 128 pixels. The two ablation architectures (flat-stacking CNN at 74 K parameters and U-Net at 474 K parameters) share the same decoder. The only architectural variable across the three deep models is how the eight input epochs are encoded: channel concatenation versus a recurrent hidden state.

<a id="data"></a>

## Data

The pipeline uses the European Commission Joint Research Centre Global Human Settlement Layer (GHSL) R2023A products. Raw rasters are not stored in this repository.

| Layer                     | Product             | Epochs                                                   |
|---------------------------|---------------------|----------------------------------------------------------|
| Built-up surface density  | GHS-BUILT-S R2023A  | 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020 |
| Built-up volume           | GHS-BUILT-V R2023A  | 1975 through 2020                                        |
| Population density        | GHS-POP R2023A      | 1975 through 2020                                        |

The official portal:

> https://human-settlement.emergency.copernicus.eu/download.php

Example download for the GHS-BUILT-S 2015 epoch (100 m, Mollweide projection):

```bash
mkdir -p data
cd data
curl -O https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100/V1-0/GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100_V1_0.zip
unzip GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100_V1_0.zip
```

After downloading every epoch into `data/`, run the preprocessing scripts. They reproject all layers to EPSG:5070 (Albers Equal Area, CONUS), resample to 250 m, clip to the CONUS extent (12,717 × 23,996 pixels = 305 M pixels), and write the result to `geotiff_exports/`.

```bash
python scripts/preprocess_all_data.py        # 1975 to 2015
python scripts/preprocess_2020.py            # 2020 blind temporal holdout
```

<a id="installation"></a>

## Installation

```bash
git clone https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction.git
cd Multi-Horizon-Urban-Growth-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Full system requirements and platform-specific notes are in [docs/INSTALLATION.md](docs/INSTALLATION.md).

<a id="reproducing-all-experiments"></a>

## Reproducing all experiments

After installing dependencies and running `scripts/preprocess_all_data.py`:

```bash
# Main models
python scripts/train_3channel.py
python scripts/train_convlstm_seed0.py
python scripts/cnn_multiseed.py
python scripts/train_unet_seed0.py
python scripts/train_sota_baselines.py

# Multi-horizon
python scripts/train_multihorizon.py
python scripts/run_cnn_multihorizon.py
python scripts/run_cnn_channel_control.py

# Ablations
python scripts/run_ablations_3ch.py

# 2015 and 2020 evaluation
python scripts/validate_2015.py
python scripts/preprocess_2020.py
python scripts/validate_2020.py
python scripts/cnn_2020_holdout.py
python scripts/unet_2020_holdout.py
python scripts/unet_2020_pixel_filtered.py
python scripts/sleuth_2020_holdout.py

# Per-tile statistics and threshold sensitivity
python scripts/compute_pertile_fom.py
python scripts/threshold_sensitivity.py

# Uncertainty maps (MC Dropout, 20 passes)
python scripts/generate_uncertainty_maps.py

# Figures
python scripts/generate_figures.py
python scripts/make_multihorizon_figure.py
python scripts/make_paper_fom_figure.py
python scripts/make_architecture_figure.py
```

Final aggregated numbers are written to `results/metrics/all_results.json` by `scripts/consolidate_results.py`.

<a id="evaluation-protocol"></a>

## Evaluation protocol

* **Validation tiles.** 821 geographic blocks selected by `block_id mod 5 == 0`, stored in `results/metrics/val_tile_indices.json`. The same tile list is used by every model and every script.
* **Evaluation mask.** `gt > 0.01` on the target epoch, restricting Figure of Merit to pixels with meaningful built-up coverage.
* **Figure of Merit.** Pontius et al., 2008: FoM = B / (A + B + C), where B is correctly predicted growth, A is missed growth, C is false growth. True negatives are excluded.
* **Multi-seed.** Every neural model is trained with two seeds (0 and 42); reported numbers are mean ± std across seeds.
* **Statistical confirmation.** Paired Wilcoxon signed-rank test on per-tile FoM values plus bootstrap 95% confidence intervals (`scripts/compute_pertile_fom.py`).

<a id="hardware-and-runtime"></a>

## Hardware and runtime

| Stage                                          | Hardware                | Wall time      |
|------------------------------------------------|-------------------------|-----------------|
| Preprocessing (all GHSL epochs to GeoTIFF)     | 16-core CPU, 64 GB RAM  | ~2 hours        |
| ConvLSTM training, one seed                    | 1 × NVIDIA A100 40 GB   | ~24 hours       |
| CNN training, one seed                         | 1 × NVIDIA A100 40 GB   | ~76 minutes     |
| U-Net training, one seed                       | 1 × NVIDIA A100 40 GB   | ~92 minutes     |
| MC Dropout inference (20 passes, 821 tiles)    | 1 × NVIDIA A100 40 GB   | ~40 minutes     |
| SLEUTH calibration (8-step prediction chain)   | 16-core CPU             | ~6 hours        |

Total reproducible compute for the full benchmark across two seeds is roughly five GPU-days on a single A100 plus a half day of CPU work.

<a id="pre-trained-weights"></a>

## Pre-trained weights

Trained model checkpoints are not distributed with this repository. To regenerate them, run the training scripts listed above on a machine with a CUDA-capable GPU. Each ConvLSTM training run produces a `models/best_3ch_mc_model.pth` checkpoint along with a JSON training history at `results/metrics/training_3ch_history.json`.

<a id="citation"></a>

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{bali_mhugp_2026,
  author = {Bali, Rohan},
  title  = {Multi-Horizon Urban Growth Prediction},
  year   = {2026},
  url    = {https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction}
}
```

A peer-reviewed manuscript describing this benchmark is currently in submission; this citation block will be updated once it is published.

<a id="acknowledgments"></a>

## Acknowledgments

The Global Human Settlement Layer (GHSL) products used throughout this work are produced and distributed by the European Commission Joint Research Centre under the Copernicus programme. The SLEUTH cellular automaton baseline implementation follows Clarke et al., 1997. The Figure of Merit metric follows Pontius et al., 2008.

<a id="contact"></a>

## Contact

Questions, issues, or collaboration enquiries:

- Open a GitHub issue at [https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction/issues](https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction/issues)
- Email **[rohanbaliwork@gmail.com](mailto:rohanbaliwork@gmail.com)** or **[rbali@umassd.edu](mailto:rbali@umassd.edu)**

<a id="license"></a>

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.
