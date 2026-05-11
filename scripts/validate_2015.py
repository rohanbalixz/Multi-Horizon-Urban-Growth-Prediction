#!/usr/bin/env python3
"""
2015 Forecast Validation
========================
Uses the trained 3-channel ConvLSTM to predict 2015 built-up patterns,
then compares with real GHSL 2015 ground truth.

Validation is performed ONLY on spatially held-out tiles (the ~20% of
geographic blocks never seen during training). This gives a clean,
leak-free evaluation where spatial autocorrelation is not a factor.

Val tile indices are loaded from results/metrics/val_tile_indices.json,
which is written by train_3channel.py at training time.

Compares against:
  - Linear extrapolation (2005→2010 trend projected to 2015)
  - Persistence baseline (predict 2010 stays same)

Outputs:
  - results/metrics/validation_2015_results.json
  - geotiff_exports/CONUS_builtup_2015_predicted.tif
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import rasterio

print("=" * 60, flush=True)
print("2015 FORECAST VALIDATION (spatial block holdout)", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

from src.models.convlstm import ConvLSTM

# =====================================================
# Autoregressive future-step prediction (matches training)
# =====================================================
def predict_future_step(model, sequences):
    _, hidden_states = model(sequences)
    last_input = sequences[:, -1]
    x_t = last_input
    layer_hiddens = []
    for layer_idx, layer in enumerate(model.convlstm_layers):
        h, c = hidden_states[layer_idx]
        h, c = layer(x_t, (h, c))
        h = model.mc_dropouts[layer_idx](h)
        hidden_states[layer_idx] = (h, c)
        x_t = h
        layer_hiddens.append(h)
    return model.decode(layer_hiddens)  # (1, 1, H, W)


# =====================================================
# Load val tile indices saved during training
# =====================================================
val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
if not os.path.exists(val_idx_path):
    print("ERROR: val_tile_indices.json not found.")
    print("  Run train_3channel.py first to generate spatial block holdout indices.")
    sys.exit(1)

with open(val_idx_path) as f:
    val_tiles = [tuple(t) for t in json.load(f)]

print(f"\n[HOLDOUT] Loaded {len(val_tiles):,} held-out val tiles from spatial block split", flush=True)

# =====================================================
# Load data
# =====================================================
INPUT_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]

print("\n[DATA] Loading all data channels...", flush=True)
builtup = {}
for year in INPUT_EPOCHS + [2015]:
    path = str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found!")
        if year == 2015:
            print("  Run scripts/preprocess_2015.py first.")
        sys.exit(1)
    with rasterio.open(path) as src:
        builtup[year] = src.read(1)
        if year == 2015:
            ref_profile = src.profile.copy()
    print(f"  {year}: shape={builtup[year].shape}, mean={builtup[year].mean():.6f}", flush=True)

ref_shape = builtup[2010].shape

volume = {}
for year in INPUT_EPOCHS:
    vpath = str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")
    with rasterio.open(vpath) as src:
        volume[year] = src.read(1)
    print(f"  Volume {year}: mean={volume[year].mean():.4f}", flush=True)

population = {}
for year in INPUT_EPOCHS:
    fpath = str(OUTPUT_DIR / f"CONUS_population_{year}.tif")
    with rasterio.open(fpath) as src:
        population[year] = src.read(1)
    print(f"  Population {year}: mean={population[year].mean():.4f}", flush=True)

gc.collect()

# =====================================================
# Load trained model
# =====================================================
print("\n[MODEL] Loading trained 3ch ConvLSTM...", flush=True)
model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1)
ckpt = torch.load(str(MODELS_DIR / "best_3ch_mc_model.pth"), map_location='cpu', weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# =====================================================
# Predict on held-out val tiles only
# =====================================================
TILE_SIZE = 128
height, width = ref_shape

prediction = np.zeros(ref_shape, dtype=np.float64)
counts      = np.zeros(ref_shape, dtype=np.float64)

print(f"\n[PREDICT] Running on {len(val_tiles):,} held-out tiles...", flush=True)
t_start = time.time()

for n, (i, j) in enumerate(val_tiles):
    frames = []
    for epoch in INPUT_EPOCHS:
        bu  = builtup[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
        vol = volume[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
        pop = population[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
        frames.append(np.stack([bu, vol, pop], axis=0))

    seq = np.stack(frames, axis=0)
    x   = torch.FloatTensor(seq).unsqueeze(0)

    with torch.no_grad():
        pred = predict_future_step(model, x)

    pred_np = pred.squeeze().numpy().clip(0, 1)
    prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred_np
    counts[i:i+TILE_SIZE, j:j+TILE_SIZE] += 1

    if (n + 1) % 500 == 0:
        print(f"    {n+1}/{len(val_tiles)} tiles, {time.time()-t_start:.0f}s elapsed", flush=True)

mask = counts > 0
prediction[mask] /= counts[mask]
prediction = prediction.astype(np.float32)
print(f"  Done. {time.time()-t_start:.0f}s elapsed", flush=True)

# =====================================================
# Save prediction GeoTIFF
# =====================================================
print("\n[SAVE] Writing predicted 2015 GeoTIFF...", flush=True)
out_path = str(OUTPUT_DIR / "CONUS_builtup_2015_predicted.tif")
ref_profile.update(dtype='float32', count=1, compress='deflate')
with rasterio.open(out_path, 'w', **ref_profile) as dst:
    dst.write(prediction, 1)
print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)

# =====================================================
# Metrics on held-out tiles only
# =====================================================
print("\n[METRICS] Computing metrics on held-out val tiles only...", flush=True)

gt = builtup[2015]
# eval_mask: held-out tiles only, urban pixels defined by GROUND TRUTH only.
# Do NOT include (prediction > 0.01) — that would let the model define its own eval set.
eval_mask = (counts > 0) & (gt > 0.01)
n_eval_pixels = int(eval_mask.sum())

gt_eval   = gt[eval_mask]
pred_eval = prediction[eval_mask]

mse  = float(np.mean((gt_eval - pred_eval) ** 2))
mae  = float(np.mean(np.abs(gt_eval - pred_eval)))
rmse = float(np.sqrt(mse))
ss_res = np.sum((gt_eval - pred_eval) ** 2)
ss_tot = np.sum((gt_eval - gt_eval.mean()) ** 2)
r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

# Per-tile metrics for error bars
tile_mses, tile_maes = [], []
for (i, j) in val_tiles:
    tile_gt   = gt[i:i+TILE_SIZE, j:j+TILE_SIZE]
    tile_pred = prediction[i:i+TILE_SIZE, j:j+TILE_SIZE]
    tile_cnt  = counts[i:i+TILE_SIZE, j:j+TILE_SIZE]
    if tile_gt.mean() < 0.01 or tile_cnt.mean() < 0.5:
        continue
    tile_mses.append(float(np.mean((tile_gt - tile_pred) ** 2)))
    tile_maes.append(float(np.mean(np.abs(tile_gt - tile_pred))))

tile_mses = np.array(tile_mses)
tile_maes = np.array(tile_maes)

print(f"\n  --- ConvLSTM 3ch (spatial block holdout) ---")
print(f"  Evaluated pixels: {n_eval_pixels:,}")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  R²:   {r2:.4f}")
print(f"  Tile MSE: {tile_mses.mean():.6f} ± {tile_mses.std():.6f} (n={len(tile_mses)})")
print(f"  Tile MAE: {tile_maes.mean():.6f} ± {tile_maes.std():.6f}")

# Linear baseline — same eval_mask (held-out pixels only)
print("\n  --- LINEAR EXTRAPOLATION BASELINE ---")
linear_pred = np.clip(2 * builtup[2010] - builtup[2005], 0, 1)
lin_eval    = linear_pred[eval_mask]
lin_mse     = float(np.mean((gt_eval - lin_eval) ** 2))
lin_mae     = float(np.mean(np.abs(gt_eval - lin_eval)))
lin_rmse    = float(np.sqrt(lin_mse))
lin_ss_res  = np.sum((gt_eval - lin_eval) ** 2)
lin_r2      = float(1 - lin_ss_res / ss_tot) if ss_tot > 0 else float('nan')
print(f"  Linear MSE:  {lin_mse:.6f}")
print(f"  Linear MAE:  {lin_mae:.6f}")
print(f"  Linear R²:   {lin_r2:.4f}")

imp_linear = (1 - mse / lin_mse) * 100 if lin_mse > 0 else 0
print(f"\n  ConvLSTM vs Linear: {imp_linear:+.1f}% MSE improvement")

# Persistence baseline
print("\n  --- PERSISTENCE BASELINE ---")
per_eval = builtup[2010][eval_mask]
per_mse  = float(np.mean((gt_eval - per_eval) ** 2))
per_mae  = float(np.mean(np.abs(gt_eval - per_eval)))
per_rmse = float(np.sqrt(per_mse))
print(f"  Persistence MSE: {per_mse:.6f}")
print(f"  Persistence MAE: {per_mae:.6f}")

imp_persist = (1 - mse / per_mse) * 100 if per_mse > 0 else 0
print(f"  ConvLSTM vs Persistence: {imp_persist:+.1f}% MSE improvement")

# =====================================================
# Figure of Merit (FoM)
# Standard metric in urban growth modeling since SLEUTH (Clarke 1997).
# FoM = |predicted_change ∩ observed_change| / |predicted_change ∪ observed_change|
# Measures spatial accuracy of growth prediction, not just pixel magnitude.
# =====================================================
print("\n[METRICS] Computing Figure of Merit (FoM)...", flush=True)

CHANGE_THRESHOLD = 0.01  # >1% built-up density change = meaningful urban growth

obs_change  = (gt - builtup[2010]) > CHANGE_THRESHOLD   # where growth actually occurred
pred_change = (prediction - builtup[2010]) > CHANGE_THRESHOLD  # where model predicts growth
lin_change  = (linear_pred - builtup[2010]) > CHANGE_THRESHOLD
per_change  = np.zeros(ref_shape, dtype=bool)  # persistence: never predicts change

# Restrict to held-out evaluation pixels
obs_ch_val  = obs_change[eval_mask]
pred_ch_val = pred_change[eval_mask]
lin_ch_val  = lin_change[eval_mask]

def figure_of_merit(obs, pred):
    inter = int((obs & pred).sum())
    union = int((obs | pred).sum())
    return inter / union if union > 0 else 0.0

convlstm_fom  = figure_of_merit(obs_ch_val, pred_ch_val)
linear_fom    = figure_of_merit(obs_ch_val, lin_ch_val)
persist_fom   = 0.0  # persistence predicts no change: zero intersection with any growth

n_obs_change = int(obs_ch_val.sum())
pct_growth   = 100.0 * n_obs_change / n_eval_pixels

print(f"\n  --- FIGURE OF MERIT (change threshold={CHANGE_THRESHOLD}) ---")
print(f"  Observed growth pixels: {n_obs_change:,} of {n_eval_pixels:,} ({pct_growth:.1f}%)")
print(f"  ConvLSTM FoM:   {convlstm_fom:.4f}")
print(f"  Linear FoM:     {linear_fom:.4f}")
print(f"  Persistence FoM:{persist_fom:.4f}")
print(f"  ConvLSTM vs Linear FoM: {(convlstm_fom - linear_fom)*100:.1f} pp")

# =====================================================
# Growth-region evaluation
# Separate metrics for pixels that grew vs stayed stable.
# Linear extrapolation struggles in high-growth areas;
# ConvLSTM's spatiotemporal reasoning shows its advantage here.
# =====================================================
print("\n[METRICS] Growth-region vs stable-region evaluation...", flush=True)

growth_mask = eval_mask & obs_change
stable_mask = eval_mask & ~obs_change

n_growth = int(growth_mask.sum())
n_stable = int(stable_mask.sum())

# ConvLSTM
growth_gt_px   = gt[growth_mask]
growth_pred_px = prediction[growth_mask]
growth_mse     = float(np.mean((growth_gt_px - growth_pred_px) ** 2))
growth_mae     = float(np.mean(np.abs(growth_gt_px - growth_pred_px)))
growth_ss_res  = np.sum((growth_gt_px - growth_pred_px) ** 2)
growth_ss_tot  = np.sum((growth_gt_px - growth_gt_px.mean()) ** 2)
growth_r2      = float(1 - growth_ss_res / growth_ss_tot) if growth_ss_tot > 0 else float('nan')

stable_gt_px   = gt[stable_mask]
stable_pred_px = prediction[stable_mask]
stable_mse     = float(np.mean((stable_gt_px - stable_pred_px) ** 2))
stable_mae     = float(np.mean(np.abs(stable_gt_px - stable_pred_px)))

# Linear on growth/stable pixels
lin_growth_mse = float(np.mean((growth_gt_px - linear_pred[growth_mask]) ** 2))
lin_stable_mse = float(np.mean((stable_gt_px - linear_pred[stable_mask]) ** 2))

# Persistence on growth pixels
per_growth_mse = float(np.mean((growth_gt_px - builtup[2010][growth_mask]) ** 2))

imp_growth  = (1 - growth_mse / lin_growth_mse) * 100 if lin_growth_mse > 0 else 0
imp_persist_growth = (1 - growth_mse / per_growth_mse) * 100 if per_growth_mse > 0 else 0

print(f"\n  --- GROWTH-REGION (pixels where built-up increased >{CHANGE_THRESHOLD}) ---")
print(f"  Growth pixels: {n_growth:,} ({100*n_growth/n_eval_pixels:.1f}% of evaluated)")
print(f"  ConvLSTM growth MSE:    {growth_mse:.6f}  R²={growth_r2:.4f}")
print(f"  Linear    growth MSE:   {lin_growth_mse:.6f}")
print(f"  Persistence growth MSE: {per_growth_mse:.6f}")
print(f"  ConvLSTM vs Linear (growth):      {imp_growth:+.1f}%")
print(f"  ConvLSTM vs Persistence (growth): {imp_persist_growth:+.1f}%")
print(f"\n  --- STABLE-REGION (pixels with no meaningful growth) ---")
print(f"  Stable pixels: {n_stable:,}")
print(f"  ConvLSTM stable MSE: {stable_mse:.6f}")
print(f"  Linear    stable MSE:{lin_stable_mse:.6f}")

# =====================================================
# SSIM — Structural Similarity Index
# Measures spatial pattern quality: are growth corridors,
# road-aligned sprawl, and urban clusters in the right places?
# A model can have good MSE but poor spatial structure, or vice versa.
# =====================================================
print("\n[METRICS] Computing SSIM (structural similarity)...", flush=True)
from scipy.ndimage import uniform_filter

def tile_ssim(gt_tile, pred_tile):
    """SSIM for a single tile (Wang et al. 2004)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = gt_tile.astype(np.float64)
    p = pred_tile.astype(np.float64)
    mu_g  = uniform_filter(g, size=11)
    mu_p  = uniform_filter(p, size=11)
    sg2   = uniform_filter(g ** 2, size=11) - mu_g ** 2
    sp2   = uniform_filter(p ** 2, size=11) - mu_p ** 2
    sgp   = uniform_filter(g * p,  size=11) - mu_g * mu_p
    num   = (2 * mu_g * mu_p + C1) * (2 * sgp + C2)
    den   = (mu_g ** 2 + mu_p ** 2 + C1) * (sg2 + sp2 + C2)
    return float(np.mean(num / den))

ssim_conv, ssim_lin, ssim_per = [], [], []
for (i, j) in val_tiles:
    gt_t = gt[i:i+TILE_SIZE, j:j+TILE_SIZE]
    if gt_t.mean() < 0.01:
        continue
    pred_t = prediction[i:i+TILE_SIZE, j:j+TILE_SIZE]
    lin_t  = linear_pred[i:i+TILE_SIZE, j:j+TILE_SIZE]
    per_t  = builtup[2010][i:i+TILE_SIZE, j:j+TILE_SIZE]
    ssim_conv.append(tile_ssim(gt_t, pred_t))
    ssim_lin.append(tile_ssim(gt_t, lin_t))
    ssim_per.append(tile_ssim(gt_t, per_t))

mean_ssim_conv = float(np.mean(ssim_conv))
mean_ssim_lin  = float(np.mean(ssim_lin))
mean_ssim_per  = float(np.mean(ssim_per))

print(f"\n  --- SSIM (higher = better spatial structure match) ---")
print(f"  ConvLSTM SSIM:   {mean_ssim_conv:.4f}")
print(f"  Linear    SSIM:  {mean_ssim_lin:.4f}")
print(f"  Persistence SSIM:{mean_ssim_per:.4f}")

# =====================================================
# Save results
# =====================================================
results = {
    "experiment": "2015_forecast_validation",
    "description": "Spatial block holdout — val tiles from geographic regions never seen during training",
    "model": "3ch_ConvLSTM_MCDropout_SkipDecoder",
    "model_path": str(MODELS_DIR / "best_3ch_mc_model.pth"),
    "split_method": "spatial_block_holdout",
    "block_size_px": 1280,
    "val_block_rule": "block_id % 5 == 0",
    "training_period": "1975-2010 (8 input epochs)",
    "target_epoch": 2015,
    "input_epochs": INPUT_EPOCHS,
    "n_val_tiles": len(val_tiles),
    "n_eval_pixels": n_eval_pixels,
    "change_threshold": CHANGE_THRESHOLD,
    "convlstm": {
        "mse": mse, "mae": mae, "rmse": rmse, "r2": r2,
        "tile_mse_mean": float(tile_mses.mean()),
        "tile_mse_std":  float(tile_mses.std()),
        "tile_mae_mean": float(tile_maes.mean()),
        "tile_mae_std":  float(tile_maes.std()),
        "n_tiles": int(len(tile_mses)),
        "fom": convlstm_fom,
        "ssim": mean_ssim_conv,
        "growth_mse": growth_mse,
        "growth_mae": growth_mae,
        "growth_r2":  growth_r2,
        "stable_mse": stable_mse,
        "stable_mae": stable_mae,
        "n_growth_pixels": n_growth,
        "n_stable_pixels": n_stable,
    },
    "linear_extrapolation": {
        "mse": lin_mse, "mae": lin_mae, "rmse": lin_rmse, "r2": lin_r2,
        "method": "2*builtup_2010 - builtup_2005",
        "fom": linear_fom,
        "ssim": mean_ssim_lin,
        "growth_mse": lin_growth_mse,
        "stable_mse": lin_stable_mse,
    },
    "persistence": {
        "mse": per_mse, "mae": per_mae, "rmse": per_rmse,
        "method": "builtup_2010 unchanged",
        "fom": persist_fom,
        "ssim": mean_ssim_per,
        "growth_mse": per_growth_mse,
    },
    "improvement_over_linear_pct": imp_linear,
    "improvement_over_persistence_pct": imp_persist,
    "fom_vs_linear_pp": float((convlstm_fom - linear_fom) * 100),
    "growth_mse_vs_linear_pct": imp_growth,
    "growth_mse_vs_persistence_pct": imp_persist_growth,
    "timestamp": str(datetime.datetime.now()),
}

results_path = str(RESULTS_DIR / "validation_2015_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {results_path}", flush=True)

# Auto-consolidate
from scripts.consolidate_results import consolidate
consolidate()

print(f"\n{'='*60}")
print("2015 FORECAST VALIDATION COMPLETE")
print(f"{'='*60}")
print(f"\n  ConvLSTM R²={r2:.4f} | FoM={convlstm_fom:.4f} | SSIM={mean_ssim_conv:.4f}")
print(f"  vs Linear:      MSE {imp_linear:+.1f}% | FoM {(convlstm_fom-linear_fom)*100:+.1f}pp | Growth MSE {imp_growth:+.1f}%")
print(f"  vs Persistence: MSE {imp_persist:+.1f}% | Growth MSE {imp_persist_growth:+.1f}%")
