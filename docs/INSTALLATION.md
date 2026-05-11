# Installation Guide

## Prerequisites

- **Python**: 3.12 or higher
- **pip**: Latest version recommended
- **Git**: For cloning the repository
- **Storage**: ~10 GB for GHSL datasets
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: Optional (CPU training takes varies by hardware)

## Quick Installation

### 1. Clone Repository

```bash
git clone https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction.git
cd Multi-Horizon-Urban-Growth-Prediction
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

Core packages installed:
- `torch>=1.10.0` - Deep learning framework
- `torchvision>=0.11.0` - Vision utilities
- `numpy>=1.21.0` - Numerical computing
- `rasterio>=1.2.0` - Geospatial raster I/O
- `geopandas>=0.10.0` - Geographic data processing
- `matplotlib>=3.4.0` - Visualization
- `jupyter>=1.0.0` - Interactive notebooks
- `tqdm>=4.62.0` - Progress bars

See [requirements.txt](../requirements.txt) for complete list.

## Dataset Setup

### GHSL Built-Up Surface Data

1. Download GHSL R2023A from [JRC Data Catalogue](https://ghsl.jrc.ec.europa.eu/download.php):
   - `GHS_BUILT_S_E1975_GLOBE_R2023A_4326_3ss_V1_0.tif`
   - `GHS_BUILT_S_E1990_GLOBE_R2023A_4326_3ss_V1_0.tif`
   - `GHS_BUILT_S_E2000_GLOBE_R2023A_4326_3ss_V1_0.tif`

2. Place files in `data/ghsl/` directory:
```bash
mkdir -p data/ghsl
# Move downloaded .tif files to data/ghsl/
```

### OpenStreetMap Road Networks

1. Download U.S. OSM data from [Geofabrik](https://download.geofabrik.de/north-america/us.html):
```bash
cd data
wget https://download.geofabrik.de/north-america/us-latest.osm.pbf
```

2. Or download specific region:
```bash
# Example: California only
wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf
```

## Verify Installation

### Test PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

### Test CUDA (if using GPU)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Run Model Test

```bash
python -c "from src.models.convlstm import create_model; model = create_model(); print('✓ Model created successfully')"
```

## Pretrained Weights

Train the 3-channel ConvLSTM model:

```bash
python scripts/train_3channel.py
```

This produces `models/best_3ch_mc_model.pth`. To also train baselines:
```bash
python scripts/run_ablations_3ch.py
python scripts/train_sota_baselines.py
```

## Jupyter Notebook Setup

Launch notebook server:
```bash
jupyter notebook notebooks/urban_growth_prediction.ipynb
```

Or use JupyterLab:
```bash
jupyter lab
```

## Docker Installation (Alternative)

Coming soon - containerized environment for reproducibility.

```bash
# docker build -t neural-timecapsule .
# docker run -p 8888:8888 neural-timecapsule
```

## Troubleshooting

### Issue: GDAL/Rasterio Installation Fails

**Solution (macOS):**
```bash
brew install gdal
pip install rasterio --no-binary rasterio
```

**Solution (Linux):**
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install rasterio
```

**Solution (Windows):**
Download precompiled wheel from [Christoph Gohlke's repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)

### Issue: PyTorch CUDA Version Mismatch

Check CUDA version:
```bash
nvidia-smi
```

Install matching PyTorch:
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of Memory During Training

Reduce batch size in training config:
```python
batch_size = 4  # Reduce from default 8
```

Or use CPU-only mode:
```python
device = 'cpu'
```

## Development Installation

For contributing to the project:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.12 | 3.12+ |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB | 50 GB |
| CPU | 4 cores | 8+ cores |
| GPU | None (CPU works) | NVIDIA GPU with 8+ GB VRAM |

## Next Steps

After installation:
1. ✅ Verify all dependencies installed
2. ✅ Download GHSL and OSM datasets
3. ✅ Run preprocessing pipeline
4. ✅ Start training or load pretrained weights
5. ✅ Explore notebooks for examples

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for complete reproduction guide.

## Support

- 📧 **Email**: rohan.bali@example.com
- 💬 **GitHub Issues**: [Report problems](https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction/issues)
- 📖 **Documentation**: [Project Wiki](https://github.com/rohanbalixz/Multi-Horizon-Urban-Growth-Prediction/wiki)
