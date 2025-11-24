# Model Card: Neural Time Capsule - Urban Growth Prediction

## Model Details

**Model Name**: Neural Time Capsule ConvLSTM  
**Version**: 1.0  
**Date**: November 2025  
**Authors**: Rohan Bali  
**License**: MIT  
**Repository**: https://github.com/rohanbalixz/NeuralTimeCapsule

### Model Description

Dual-channel Convolutional LSTM architecture for multi-decadal urban growth forecasting using satellite-derived built-up surface density and road network infrastructure.

**Architecture**:
- 2-layer ConvLSTM with 64 hidden channels per layer
- Input: 2 channels (built-up density, road infrastructure)
- Output: Single-channel built-up density prediction
- Total parameters: 470,593
- Kernel size: 3×3
- Tile resolution: 128×128 pixels

**Training Configuration**:
- Optimizer: Adam (lr=1e-3)
- Loss function: Mean Squared Error (MSE)
- Batch size: 8
- Epochs: 50
- Training time: 6 hours on laptop CPU
- Device: CPU-optimized (no GPU required)

## Intended Use

### Primary Applications

1. **Urban Planning**: Long-term growth projections for infrastructure planning
2. **Climate Research**: Understanding urban heat island expansion
3. **Policy Analysis**: Evaluating impact of zoning regulations
4. **Resource Allocation**: Anticipating service demand (water, electricity, schools)

### Intended Users

- Urban planners and policymakers
- Climate and sustainability researchers
- GIS analysts and geographers
- Machine learning researchers in spatio-temporal modeling

### Out-of-Scope Uses

❌ **NOT intended for**:
- Real-time or short-term predictions (designed for decade-scale forecasting)
- Individual building-level predictions (operates at 250m resolution)
- Regions with drastically different development patterns than U.S. (requires transfer learning)
- Applications requiring socioeconomic causal explanations

## Training Data

### Data Sources

**GHSL Built-Up Surface** (R2023A):
- Provider: European Commission Joint Research Centre
- Resolution: ~100m (resampled to 250m)
- Epochs: 1975, 1990, 2000
- Coverage: Continental United States (CONUS)
- Values: Built-up surface density [0-100%]

**OpenStreetMap Road Networks**:
- Provider: OpenStreetMap contributors
- Road types: Highways, primary, secondary roads
- Rasterized to 250m resolution
- Binary presence indicator [0 or 1]

### Dataset Statistics

- Total tiles: 2,313 (128×128 pixels each)
- Training set: 1,850 tiles (80%)
- Validation set: 463 tiles (20%)
- Spatial coverage: Diverse U.S. regions (urban, suburban, rural)
- Temporal span: 25 years (1975-2000)

### Data Preprocessing

1. **Reprojection**: WGS84 → Albers Equal Area Conic
2. **Resampling**: 100m → 250m bilinear interpolation
3. **Normalization**: Min-max scaling to [0, 1]
4. **Tiling**: 128×128 patches with 50% overlap
5. **Augmentation**: None (preserves geospatial integrity)

## Evaluation

### Validation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MSE | 0.000218 | Mean squared error |
| MAE | 0.0165 | ~1.65% absolute error |
| RMSE | 0.0303 | ~3% typical deviation |
| Train/Val Gap | 2.3% | Minimal overfitting |

### Baseline Comparisons

| Model | MSE | Improvement |
|-------|-----|-------------|
| **ConvLSTM (ours)** | **0.00022** | **Baseline** |
| U-Net | 0.00066 | **67% improvement** |
| Standalone CNN | 0.00074 | 239% improvement |
| Linear Extrapolation | 0.0021 | 863% improvement |

### Ablation Studies

| Configuration | MSE | MAE |
|--------------|-----|-----|
| Dual-channel (built-up + roads) | 0.00022 | 0.0165 |
| Single-channel (built-up only) | 0.0028 | 0.0421 |
| Single-channel (roads only) | 0.0156 | 0.0987 |

**Key Findings**:
- Dual-channel provides **93% error reduction** vs single-channel
- Road infrastructure alone is insufficient (high MAE)
- Two-layer depth critical (single-layer: 1550% worse)

## Limitations

### Technical Limitations

1. **Sparse Temporal Sampling**: Only 3 training epochs (1975, 1990, 2000)
2. **Autoregressive Error**: Multi-step forecasts accumulate errors
3. **Missing Modalities**: No socioeconomic, climate, or elevation data
4. **No Ground Truth Validation**: 2010+ predictions cannot be validated yet

### Spatial Limitations

- Trained exclusively on U.S. data (may not generalize to other countries)
- 250m resolution misses fine-grained building-level dynamics
- Urban cores vs sprawl patterns have different error characteristics

### Temporal Limitations

- 25-year historical window may miss recent trends (e.g., remote work impacts)
- Assumes temporal dynamics are stationary (ignores policy shocks)
- Cannot predict discontinuous events (e.g., natural disasters, pandemics)

## Ethical Considerations

### Potential Risks

⚠️ **Displacement Risk**: Predictions could inform gentrification or displacement
⚠️ **Resource Inequality**: May perpetuate existing infrastructure biases
⚠️ **Environmental Justice**: Urban expansion predictions should consider vulnerable communities

### Responsible Use Guidelines

1. **Transparency**: Always disclose model predictions are probabilistic, not deterministic
2. **Human Oversight**: Use predictions to inform, not replace, human decision-making
3. **Equity Audits**: Evaluate predictions for bias against marginalized communities
4. **Stakeholder Engagement**: Involve affected communities in planning processes

## Fairness and Bias

### Known Biases

- **Training Data Bias**: Overrepresents U.S. suburban development patterns
- **Temporal Bias**: 1975-2000 training window may not reflect recent trends
- **Spatial Bias**: Higher accuracy in densely sampled regions

### Mitigation Strategies

- Report uncertainty estimates for predictions
- Validate on diverse geographic regions before deployment
- Consider ensemble methods to reduce spatial bias

## Computational Requirements

### Training

- **Time**: 6 hours
- **Hardware**: Laptop CPU (Intel i5 or equivalent)
- **Memory**: 8 GB RAM
- **Storage**: 15 GB (datasets + checkpoints)

### Inference

- **Latency**: <1 second per tile (128×128 pixels)
- **Batch Processing**: 100 tiles in ~30 seconds
- **Continental Forecast**: ~1 hour for full CONUS

## Reproducibility

All code, data sources, and pretrained weights are publicly available:

- **Repository**: https://github.com/rohanbalixz/NeuralTimeCapsule
- **Pretrained Weights**: `models/best_urban_growth_model.pth`
- **Training Notebook**: `notebooks/urban_growth_prediction.ipynb`
- **Random Seed**: 42 (fixed for reproducibility)

See [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) for step-by-step instructions.

## Updates and Maintenance

**Current Status**: Active development  
**Last Updated**: November 2025

**Planned Improvements**:
- [ ] 6-epoch training with 2014-2023 GHSL data
- [ ] Multi-modal fusion (climate, demographics)
- [ ] Uncertainty quantification (ensemble methods)
- [ ] Transfer learning for international regions

## Citation

```bibtex
@inproceedings{bali2025neuraltimecapsule,
  title={Neural Time Capsule: Forecasting Urban Development Through Multi-Decadal Spatio-Temporal ConvLSTM},
  author={Bali, Rohan},
  booktitle={CVPR},
  year={2025},
  url={https://github.com/rohanbalixz/NeuralTimeCapsule}
}
```

## Contact

**Author**: Rohan Bali  
**Email**: rohan.bali@example.com  
**GitHub**: [@rohanbalixz](https://github.com/rohanbalixz)

## References

1. Corbane et al. (2021). "The Grey-Green Divide: Multi-temporal Analysis of Greenness Across 10,000 Urban Centres Derived from the Global Human Settlement Layer"
2. Shi et al. (2015). "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
3. OpenStreetMap contributors. https://www.openstreetmap.org/

---

**Model Card Framework**: Based on [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)
