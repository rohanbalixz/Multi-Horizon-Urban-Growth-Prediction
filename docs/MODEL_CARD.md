# Model Card: Multi-Horizon Urban Growth Prediction

## Model Details

**Model Name**: Multi-Horizon Urban Growth Prediction (ConvLSTM, CNN, U-Net variants)
**Version**: 3.0
**Date**: April 2026
**Authors**: Rohan Bali
**License**: MIT
**Repository**: https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction

---

## Model Description

Multi-Horizon Urban Growth Prediction is a framework for continental-scale urban growth forecasting from multi-decadal satellite observations. Three architectures are evaluated:

### CNN (best performing)
Flat temporal channel stacking — all T×3 input channels concatenated and processed by 4 convolutional layers. No recurrence.
- **Parameters**: 74,273
- **Input**: T×3 channels × 128×128 px (T=8 → 24 channels at 5yr horizon)
- **Training time**: ~76 minutes

### ConvLSTM + MC Dropout (primary UQ model)
Two ConvLSTM layers with skip-connection decoder and MC Dropout for uncertainty quantification.
- **Parameters**: 481,153
- **Architecture**: 2 ConvLSTM layers (hidden=64) → skip-projection → Conv decoder (64→32→16→1)
- **MC Dropout**: p=0.1 applied after each ConvLSTM layer; 20 passes for uncertainty
- **Training time**: ~24 hours

### U-Net
Encoder-decoder with skip connections.
- **Parameters**: 473,857

**Common configuration**:
- Optimizer: Adam (lr=5e-4, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=15, factor=0.5)
- Loss: MSE
- Batch size: 8 | Epochs: 25 | Seed: 42
- Tile size: 128×128 px at 250m resolution

---

## Training Data

**GHSL Built-Up Surface** (GHS-BUILT-S R2023A):
- Provider: European Commission Joint Research Centre (JRC)
- Native resolution: ~90m, resampled to 250m, EPSG:5070
- Input epochs: 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010 (8 timesteps)
- Target epoch: 2015
- Normalization: min-max [0, 1]

**GHS-BUILT-V Built-Up Volume** (R2023A):
- Native resolution: 100m (Mollweide), resampled to 250m
- Same epochs as built-up surface
- Normalization: log1p / global max

**GHS-POP Population Density** (R2023A):
- Native resolution: 100m (Mollweide), resampled to 250m
- Same epochs as built-up surface
- Normalization: log1p / global max

**Coverage**: Continental United States (CONUS), 12,717 × 23,996 pixels at 250m = ~305M pixels

---

## Validation Design

**Spatial block holdout**: CONUS divided into 1280×1280px geographic blocks (~320 km²). Every 5th block (`block_id % 5 == 0`) held out — 5 geographically interleaved regions across CONUS. Eliminates spatial autocorrelation leakage from tile overlap.

- Total tiles: 5,698 | Train: 4,877 | Val: 821
- Same `val_tile_indices.json` used for all experiments

**Evaluation metric**: Figure of Merit (Pontius et al. 2008) — intersection over union of predicted vs observed growth pixels. MSE is dominated by stable non-urban background (≥97% of CONUS area); FoM isolates growth zone accuracy.

---

## Evaluation Results

### Main Results — CONUS 2015 Spatial Block Holdout

| Model | **FoM** ↑ | MSE ↓ | MAE ↓ | R² | Params |
|-------|-----------|-------|-------|-----|--------|
| **CNN 3ch (flat stacking)** | **0.739** | 0.000025 | 0.00291 | 0.995 | 74K |
| U-Net 3ch | 0.698 | 0.000034 | 0.00360 | — | 474K |
| ConvLSTM 3ch + MC Dropout | 0.511 | 0.000075 | 0.00380 | 0.978 | 481K |
| Linear extrapolation | 0.381 | 0.000069 | 0.00332 | 0.979 | — |
| SLEUTH CA (Clarke et al. 1997) | 0.056 | 0.006273 | 0.06867 | — | 4 |

SLEUTH calibrated on 200 training tiles using real TIGER/Line primary roads (261,488 segments) with an 8-step prediction chain (1975→1980→…→2010→2015).

### Growth-Zone vs Stable-Zone Decomposition (ConvLSTM, 2015)

| Region | ConvLSTM MSE | Linear MSE | Advantage |
|--------|-------------|------------|-----------|
| Growth pixels (Δ > 0.01) | 0.000437 | 0.000454 | +3.6% |
| Stable pixels | 0.000096 | 0.000073 | −31.6% |

ConvLSTM's advantage concentrates in the growth zones. Linear extrapolation's lower overall MSE reflects dominance of stable background pixels.

### 2020 Temporal Holdout (true out-of-sample)

Model trained on [1975–2010]→2015; evaluated on [1980–2015]→2020. 2020 GHSL data downloaded after training.

| Model | **FoM** | R² | Growth MSE | Stable MSE |
|-------|---------|-----|-----------|-----------|
| ConvLSTM (best) | **0.156** | 0.931 | 0.000610 | 0.000470 |
| Linear extrapolation | 0.053 | 0.866 | 0.001480 | 0.000877 |
| Persistence (no change) | 0.000 | — | 0.000869 | — |

FoM=0.156 is 3× above linear (+10.3pp). The FoM drop from 0.504 (2015) to 0.156 (2020) reflects a 5× slowdown in CONUS urban growth rate (2015–2020 vs 2010–2015), not model degradation. Only 10.6% of pixels showed growth 2015–2020 vs 23.6% for 2010–2015.

### Multi-Horizon (all predicting 2015, varying input window)

| Horizon | CNN FoM | ConvLSTM FoM | Linear FoM | CNN params |
|---------|---------|--------------|-----------|-----------|
| 5-year (1975–2010) | **0.739** | 0.523 | 0.302 | 74K |
| 10-year (1975–2005) | **0.675** | 0.596 | 0.462 | 73K |
| 20-year (1975–1995) | **0.687** | 0.597 | 0.527 | 69K |

CNN wins at every horizon. Channel-count confound experiment: when input channel count is held fixed, FoM difference across horizons reduces to <0.015 — the apparent horizon degradation is an input count artifact, not forecast difficulty.

### Ablation Study (ConvLSTM architecture, 2015 target)

| Configuration | **FoM** | MSE | Δ MSE |
|--------------|---------|-----|-------|
| 3ch ConvLSTM (full) | 0.511 | 0.000075 | — |
| Built-up + Volume (2ch) | 0.514 | 0.000074 | −1% |
| Built-up only (1ch) | 0.500 | 0.000089 | +19% |
| Volume only (1ch) | — | 0.000265 | +253% |
| Population only (1ch) | — | 0.001332 | +1,676% |
| 1-layer ConvLSTM (3ch) | 0.481 | 0.000078 | +4% |

Built-up surface dominates the signal. Volume adds structure (+17% MSE improvement over built-up alone). Population is collinear with volume and adds noise when volume is present.

### Sequence Length Ablation (ConvLSTM, 3ch)

| Seq length | Epochs used | MSE |
|-----------|-------------|-----|
| 4 | 1995–2010 | 0.000081 |
| 6 | 1985–2010 | 0.000081 |
| 8 | 1975–2010 | **0.000075** |

Minimal returns from longer history (7% improvement, 4→8 steps). Supports spatial dominance hypothesis.

### Uncertainty Quantification (ConvLSTM, MC Dropout)

20 stochastic forward passes over 8.69M val pixels:

| Metric | Value |
|--------|-------|
| Mean predictive std | 0.00279 |
| Median predictive std | 0.00166 |
| Mean coefficient of variation | 0.254 |
| Mean 95% CI width | 0.00952 |
| **Calibration r (std vs actual error)** | **0.983** |

Calibration is monotonically ordered across all 10 equal-count decile bins — the model reliably identifies uncertain predictions.

---

## Limitations

- **Resolution**: 250m tiles miss fine-grained building or parcel dynamics
- **Covariates**: No zoning, terrain, road network, or economic inputs in neural models
- **Geography**: Trained on CONUS only; not validated elsewhere
- **Temporal**: 5-year epoch spacing; cannot resolve within-epoch dynamics
- **Growth regime**: Post-2015 CONUS slowdown (5× lower growth rate) reduces FoM on 2020 holdout — urban growth models are highly sensitive to growth rate regime

---

## Ethical Considerations

- Predictions are probabilistic and should inform, not replace, planning decisions
- Growth forecasts may encode historical inequities in development patterns
- Validate for demographic and geographic bias before use in resource allocation or zoning

---

## Reproducibility

- **Code**: https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction
- **Weights**: `models/best_3ch_mc_model.pth`
- **Random seed**: 42
- **Val split**: `results/metrics/val_tile_indices.json`

---

## Citation

```bibtex
@software{bali_mhugp_2026,
  author = {Bali, Rohan},
  title  = {Multi-Horizon Urban Growth Prediction},
  year   = {2026},
  url    = {https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction}
}
```

*Model Card Framework: Mitchell et al. (2019)*
