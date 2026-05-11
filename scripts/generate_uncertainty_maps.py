#!/usr/bin/env python3
"""
Generate Uncertainty Maps via MC Dropout (20 Forward Passes)
============================================================
Loads the trained 3-channel ConvLSTM (with MC Dropout p=0.1),
runs 20 stochastic forward passes per tile, and produces:

  1. Mean prediction map (CONUS-wide built-up 2000)
  2. Pixel-wise uncertainty (std) map
  3. Coefficient of variation (CV) map
  4. Prediction interval width (95% CI) map
  5. GeoTIFF exports of all maps

Visualizations saved to results/figures/uncertainty_*.png
GeoTIFFs saved to geotiff_exports/CONUS_uncertainty_*.tif

Reference: Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation"
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("UNCERTAINTY MAP GENERATION (MC Dropout)", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
import rasterio
from pathlib import Path

print("  Imports complete.", flush=True)

from src.models.convlstm import ConvLSTM

N_MC_PASSES = 20
MC_DROPOUT = 0.1
TILE_SIZE = 128
STRIDE = 128  # no overlap (fast); increase to 64 for smooth compositing
BATCH_SIZE = 32  # batch tiles together for faster MC inference
SKIP_THRESHOLD = 0.01  # match training dataset filter

# =====================================================
# Load Data from pre-processed GeoTIFFs
# =====================================================

print("\n[DATA] Loading all channels...", flush=True)

# 8-timestep input matching training setup
INPUT_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH = 2015

# Load GHSL (already normalized [0,1])
print("  Loading built-up epochs...", flush=True)
ghsl = {}
geo_info = None
for epoch in INPUT_EPOCHS + [TARGET_EPOCH]:
    fpath = str(OUTPUT_DIR / f"CONUS_builtup_{epoch}.tif")
    with rasterio.open(fpath) as src:
        ghsl[epoch] = src.read(1)
        if geo_info is None:
            geo_info = {'transform': src.transform, 'crs': str(src.crs),
                        'height': src.height, 'width': src.width}

ref_shape = ghsl[TARGET_EPOCH].shape
print(f"  Grid: {ref_shape[0]} x {ref_shape[1]}", flush=True)

# Load built-up volume (temporal, one per epoch)
print("  Loading built-up volume...", flush=True)
volume = {}
for year in INPUT_EPOCHS:
    vpath = str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")
    if not os.path.exists(vpath):
        print(f"  ERROR: {vpath} not found! Run scripts/preprocess_all_data.py first.", flush=True)
        sys.exit(1)
    with rasterio.open(vpath) as src:
        volume[year] = src.read(1)
    print(f"  Volume {year}: shape={volume[year].shape}, mean={volume[year].mean():.4f}", flush=True)

# Load population density
print("  Loading population density...", flush=True)
population = {}
for year in INPUT_EPOCHS:
    fpath = str(OUTPUT_DIR / f"CONUS_population_{year}.tif")
    with rasterio.open(fpath) as src:
        population[year] = src.read(1)
gc.collect()

# =====================================================
# Load Model
# =====================================================
print("\n[MODEL] Loading 3-channel ConvLSTM with MC Dropout...", flush=True)

model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)

model_path = str(MODELS_DIR / "best_3ch_mc_model.pth")
if os.path.exists(model_path):
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded: {model_path}", flush=True)
else:
    print(f"  ERROR: {model_path} not found!", flush=True)
    print(f"  Run scripts/train_3channel.py first to train the 3ch model.", flush=True)
    sys.exit(1)

print(f"  Parameters: {model.count_parameters():,}", flush=True)


# =====================================================
# MC Dropout Inference
# =====================================================
def predict_batch_mc(model, batch_input, n_passes=N_MC_PASSES):
    """
    Run MC Dropout inference on a BATCH of tiles.
    
    Args:
        model: ConvLSTM with MC Dropout
        batch_input: (B, 2, 3, 128, 128) input tensor
        n_passes: Number of MC forward passes
    
    Returns:
        means: (B, 128, 128) mean predictions
        stds: (B, 128, 128) uncertainties
        all_preds: (B, n_passes, 128, 128) all predictions
    """
    model.enable_mc_dropout()
    B = batch_input.shape[0]
    
    all_preds = np.zeros((B, n_passes, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    with torch.no_grad():
        for p in range(n_passes):
            _, hidden_states = model(batch_input)
            last_input = batch_input[:, -1]
            x_t = last_input
            layer_hiddens = []
            for layer_idx, layer in enumerate(model.convlstm_layers):
                h, c = hidden_states[layer_idx]
                h, c = layer(x_t, (h, c))
                h = model.mc_dropouts[layer_idx](h)
                hidden_states[layer_idx] = (h, c)
                x_t = h
                layer_hiddens.append(h)
            pred = model.decode(layer_hiddens)  # (B, 1, H, W)
            all_preds[:, p] = pred.squeeze(1).cpu().numpy()
    
    means = all_preds.mean(axis=1)   # (B, H, W)
    stds = all_preds.std(axis=1)     # (B, H, W)
    
    return means, stds, all_preds


print(f"\n[MC] Running {N_MC_PASSES} MC forward passes on all tiles...", flush=True)

height, width = ref_shape
n_tiles_y = (height - TILE_SIZE) // STRIDE + 1
n_tiles_x = (width - TILE_SIZE) // STRIDE + 1
total_tiles = n_tiles_y * n_tiles_x

# Accumulation maps for overlap averaging
mean_accum = np.zeros(ref_shape, dtype=np.float64)
std_accum = np.zeros(ref_shape, dtype=np.float64)
count_accum = np.zeros(ref_shape, dtype=np.float64)
# Also accumulate all MC passes for ci_width
ci95_accum = np.zeros(ref_shape, dtype=np.float64)

# ---- Phase 1: Collect valid tile coordinates ----
print(f"  Grid: {n_tiles_y} x {n_tiles_x} = {total_tiles} potential tiles", flush=True)
print(f"  Filtering with threshold={SKIP_THRESHOLD}...", flush=True)

tile_coords = []
for y in range(0, height - TILE_SIZE + 1, STRIDE):
    for x in range(0, width - TILE_SIZE + 1, STRIDE):
        # Check if any input epoch has meaningful urban content
        has_urban = any(
            ghsl[ep][y:y+TILE_SIZE, x:x+TILE_SIZE].mean() >= SKIP_THRESHOLD
            for ep in INPUT_EPOCHS
        )
        if has_urban:
            tile_coords.append((y, x))

tiles_skipped = total_tiles - len(tile_coords)
print(f"  Valid tiles: {len(tile_coords)} (skipped {tiles_skipped})", flush=True)

# ---- Phase 2: Batched MC inference ----
t_start = time.time()
tiles_processed = 0

for batch_start in range(0, len(tile_coords), BATCH_SIZE):
    batch_coords = tile_coords[batch_start:batch_start + BATCH_SIZE]
    B = len(batch_coords)
    
    # Build batch tensor: 4 timesteps x 3 channels
    batch_inputs = np.zeros((B, len(INPUT_EPOCHS), 3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    for i, (y, x) in enumerate(batch_coords):
        for t, epoch in enumerate(INPUT_EPOCHS):
            bu = ghsl[epoch][y:y+TILE_SIZE, x:x+TILE_SIZE]
            vol = volume[epoch][y:y+TILE_SIZE, x:x+TILE_SIZE]
            pop = population[epoch][y:y+TILE_SIZE, x:x+TILE_SIZE]
            batch_inputs[i, t] = np.stack([bu, vol, pop], axis=0)
    
    batch_tensor = torch.FloatTensor(batch_inputs)
    means, stds, all_preds = predict_batch_mc(model, batch_tensor)
    
    # 95% CI width per tile
    ci_low = np.percentile(all_preds, 2.5, axis=1)   # (B, H, W)
    ci_high = np.percentile(all_preds, 97.5, axis=1)  # (B, H, W)
    ci_width = ci_high - ci_low
    
    # Accumulate into full maps
    for i, (y, x) in enumerate(batch_coords):
        mean_accum[y:y+TILE_SIZE, x:x+TILE_SIZE] += means[i]
        std_accum[y:y+TILE_SIZE, x:x+TILE_SIZE] += stds[i]
        ci95_accum[y:y+TILE_SIZE, x:x+TILE_SIZE] += ci_width[i]
        count_accum[y:y+TILE_SIZE, x:x+TILE_SIZE] += 1
    
    tiles_processed += B
    
    if tiles_processed % 100 < BATCH_SIZE or batch_start == 0:
        elapsed = time.time() - t_start
        rate = tiles_processed / max(elapsed, 0.1)
        remaining = (len(tile_coords) - tiles_processed) / max(rate, 0.01)
        print(f"  Tiles: {tiles_processed}/{len(tile_coords)} "
              f"({elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining)", flush=True)

# Normalize by overlap count
mask = count_accum > 0
mean_map = np.zeros(ref_shape, dtype=np.float32)
std_map = np.zeros(ref_shape, dtype=np.float32)
ci95_map = np.zeros(ref_shape, dtype=np.float32)

mean_map[mask] = (mean_accum[mask] / count_accum[mask]).astype(np.float32)
std_map[mask] = (std_accum[mask] / count_accum[mask]).astype(np.float32)
ci95_map[mask] = (ci95_accum[mask] / count_accum[mask]).astype(np.float32)

# Coefficient of variation
cv_map = np.zeros(ref_shape, dtype=np.float32)
valid = (mean_map > 0.001) & mask
cv_map[valid] = std_map[valid] / mean_map[valid]

total_time = time.time() - t_start
print(f"\n  MC inference complete: {tiles_processed} tiles, {total_time/60:.0f} min", flush=True)
print(f"  Mean prediction range: [{mean_map[mask].min():.4f}, {mean_map[mask].max():.4f}]", flush=True)
print(f"  Uncertainty (std) range: [{std_map[mask].min():.6f}, {std_map[mask].max():.6f}]", flush=True)
print(f"  Mean uncertainty: {std_map[mask].mean():.6f}", flush=True)
print(f"  95% CI width range: [{ci95_map[mask].min():.6f}, {ci95_map[mask].max():.6f}]", flush=True)


# =====================================================
# Save GeoTIFFs
# =====================================================
print("\n[EXPORT] Saving uncertainty GeoTIFFs...", flush=True)

def save_geotiff(data, filename, geo_info):
    path = str(OUTPUT_DIR / filename)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=geo_info['height'], width=geo_info['width'],
        count=1, dtype='float32',
        crs=geo_info['crs'], transform=geo_info['transform'],
        compress='lzw'
    ) as dst:
        # Ensure shape matches
        out = np.zeros((geo_info['height'], geo_info['width']), dtype=np.float32)
        h = min(data.shape[0], geo_info['height'])
        w = min(data.shape[1], geo_info['width'])
        out[:h, :w] = data[:h, :w]
        dst.write(out, 1)
    print(f"  {path}", flush=True)

save_geotiff(mean_map, "CONUS_mc_mean_prediction.tif", geo_info)
save_geotiff(std_map, "CONUS_mc_uncertainty_std.tif", geo_info)
save_geotiff(cv_map, "CONUS_mc_uncertainty_cv.tif", geo_info)
save_geotiff(ci95_map, "CONUS_mc_uncertainty_ci95_width.tif", geo_info)


# =====================================================
# Generate Visualizations
# =====================================================
print("\n[VIS] Generating uncertainty visualizations...", flush=True)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.gridspec as gridspec
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Figure 1: 4-panel overview ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("MC Dropout Uncertainty Analysis (20 Forward Passes)", fontsize=16, fontweight='bold')
    
    # Ground truth
    im1 = axes[0, 0].imshow(ghsl[TARGET_EPOCH], cmap='YlOrRd', vmin=0, vmax=0.5)
    axes[0, 0].set_title(f"Ground Truth (GHSL {TARGET_EPOCH})")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    # Mean prediction
    im2 = axes[0, 1].imshow(mean_map, cmap='YlOrRd', vmin=0, vmax=0.5)
    axes[0, 1].set_title("MC Mean Prediction")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Uncertainty (std)
    im3 = axes[1, 0].imshow(std_map, cmap='hot_r', vmin=0, vmax=std_map[mask].mean() * 3)
    axes[1, 0].set_title("Pixel-wise Uncertainty (Std)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
    
    # 95% CI width
    im4 = axes[1, 1].imshow(ci95_map, cmap='RdYlGn_r', vmin=0, vmax=ci95_map[mask].mean() * 3)
    axes[1, 1].set_title("95% CI Width")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "uncertainty_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/figures/uncertainty_overview.png", flush=True)
    
    # --- Figure 2: Uncertainty histogram ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    valid_std = std_map[mask & (std_map > 0)]
    ax1.hist(valid_std, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
    ax1.axvline(valid_std.mean(), color='red', linestyle='--', label=f'Mean={valid_std.mean():.5f}')
    ax1.set_xlabel("Prediction Uncertainty (Std)")
    ax1.set_ylabel("Pixel Count")
    ax1.set_title("Distribution of Prediction Uncertainty")
    ax1.legend()
    
    valid_cv = cv_map[valid & (cv_map > 0) & (cv_map < 2)]
    ax2.hist(valid_cv, bins=100, color='coral', edgecolor='none', alpha=0.7)
    ax2.axvline(valid_cv.mean(), color='red', linestyle='--', label=f'Mean CV={valid_cv.mean():.3f}')
    ax2.set_xlabel("Coefficient of Variation")
    ax2.set_ylabel("Pixel Count")
    ax2.set_title("Distribution of CV (Std/Mean)")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "uncertainty_histograms.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/figures/uncertainty_histograms.png", flush=True)
    
    # --- Figure 3: Error vs Uncertainty correlation ---
    error_map = np.abs(mean_map - ghsl[TARGET_EPOCH])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sample_mask = mask & (std_map > 0)
    # Subsample for scatter plot
    indices = np.where(sample_mask)
    n_points = min(50000, len(indices[0]))
    np.random.seed(42)
    sel = np.random.choice(len(indices[0]), n_points, replace=False)
    
    ax.scatter(std_map[indices[0][sel], indices[1][sel]],
               error_map[indices[0][sel], indices[1][sel]],
               alpha=0.1, s=1, c='steelblue')
    ax.set_xlabel("MC Dropout Uncertainty (Std)")
    ax.set_ylabel("Absolute Prediction Error")
    ax.set_title("Uncertainty vs. Error Calibration")
    
    # Fit and plot trend line
    x_vals = std_map[indices[0][sel], indices[1][sel]]
    y_vals = error_map[indices[0][sel], indices[1][sel]]
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = P.polyval(x_line, coeffs)
    ax.plot(x_line, y_line, 'r-', linewidth=2, label='Linear trend')
    
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    ax.legend(title=f'Pearson r = {corr:.3f}')
    
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "uncertainty_calibration.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/figures/uncertainty_calibration.png", flush=True)
    
except ImportError:
    print("  matplotlib not available — skipping visualizations", flush=True)


# =====================================================
# Uncertainty Calibration Analysis
# A well-calibrated uncertainty measure should show:
# pixels with higher predicted std → higher actual error.
# We bin all evaluated pixels by predicted std (deciles),
# compute mean actual absolute error per bin, and check monotonicity.
# Saved to JSON for the calibration figure in generate_figures.py.
# =====================================================
print("\n[CALIBRATION] Computing uncertainty calibration...", flush=True)

gt_map     = ghsl[TARGET_EPOCH].astype(np.float32)
abs_err    = np.abs(mean_map - gt_map)
N_BINS     = 10

# Use only pixels that were processed (mask) and have urban signal
calib_mask = mask & ((gt_map > 0.01) | (mean_map > 0.01))
std_vals   = std_map[calib_mask].flatten()
err_vals   = abs_err[calib_mask].flatten()

bin_edges = np.percentile(std_vals, np.linspace(0, 100, N_BINS + 1))
calibration_bins = []
for k in range(N_BINS):
    lo, hi = bin_edges[k], bin_edges[k + 1]
    in_bin = (std_vals >= lo) & (std_vals < hi)
    if k == N_BINS - 1:          # include right edge in last bin
        in_bin = (std_vals >= lo)
    if in_bin.sum() == 0:
        continue
    calibration_bins.append({
        "bin": k + 1,
        "std_lower": round(float(lo), 6),
        "std_upper": round(float(hi), 6),
        "mean_predicted_std": round(float(std_vals[in_bin].mean()), 6),
        "mean_actual_error":  round(float(err_vals[in_bin].mean()), 6),
        "n_pixels": int(in_bin.sum()),
    })

# Calibration slope: positive = uncertainty correlates with error (good)
calib_stds  = np.array([b["mean_predicted_std"] for b in calibration_bins])
calib_errs  = np.array([b["mean_actual_error"]  for b in calibration_bins])
calib_corr  = float(np.corrcoef(calib_stds, calib_errs)[0, 1]) if len(calib_stds) > 1 else 0.0

print(f"  Calibration Pearson r(std, error) = {calib_corr:.4f}", flush=True)
print(f"  (r close to 1.0 = well-calibrated uncertainty)", flush=True)

# =====================================================
# Save Summary Statistics
# =====================================================
print("\n[SUMMARY] Saving uncertainty statistics...", flush=True)

summary = {
    "date": datetime.datetime.now().isoformat(),
    "model": "ConvLSTM_3ch_MCDropout",
    "mc_passes": N_MC_PASSES,
    "mc_dropout_p": MC_DROPOUT,
    "tiles_processed": tiles_processed,
    "tiles_skipped": tiles_skipped,
    "inference_time_min": round(total_time / 60, 1),
    "uncertainty_stats": {
        "mean_std": round(float(std_map[mask].mean()), 6),
        "median_std": round(float(np.median(std_map[mask])), 6),
        "max_std": round(float(std_map[mask].max()), 6),
        "mean_cv": round(float(cv_map[valid].mean()), 4),
        "mean_ci95_width": round(float(ci95_map[mask].mean()), 6),
    },
    "prediction_stats": {
        "mean_prediction": round(float(mean_map[mask].mean()), 6),
        "mean_ground_truth": round(float(gt_map[mask].mean()), 6),
        "mean_abs_error": round(float(abs_err[mask].mean()), 6),
    },
    "calibration": {
        "n_bins": N_BINS,
        "pearson_r_std_vs_error": round(calib_corr, 4),
        "interpretation": "r close to 1.0 = well-calibrated; uncertain pixels have higher actual error",
        "bins": calibration_bins,
    },
    "output_files": [
        str(OUTPUT_DIR / "CONUS_mc_mean_prediction.tif"),
        str(OUTPUT_DIR / "CONUS_mc_uncertainty_std.tif"),
        str(OUTPUT_DIR / "CONUS_mc_uncertainty_cv.tif"),
        str(OUTPUT_DIR / "CONUS_mc_uncertainty_ci95_width.tif"),
    ]
}

with open(str(RESULTS_DIR / "uncertainty_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2), flush=True)
print("\n" + "=" * 60, flush=True)
print("UNCERTAINTY MAP GENERATION COMPLETE", flush=True)
print(f"Ended: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)
