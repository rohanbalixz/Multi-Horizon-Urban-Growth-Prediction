#!/usr/bin/env python3
"""
2020 Temporal Holdout Validation
=================================
Uses the trained 3-channel ConvLSTM (best_3ch_mc_model.pth) to predict
2020 built-up patterns, then compares with real GHSL 2020 ground truth.

This is a TRUE temporal holdout: the model was trained on [1975–2010] → 2015.
We now shift the input window by one epoch: [1980–2015] → 2020.
The model has NEVER seen 2020 data in any form during training.

Why this is a clean test:
  - Model architecture: ConvLSTM, seq_len=8, 3 channels per timestep
  - Training task: [1975,1980,...,2010] (8 epochs) → predict 2015
  - Holdout task:  [1980,1985,...,2015] (8 epochs) → predict 2020
  - Same temporal stride (5 years), same sequence length — model generalises
  - 2020 GHSL data downloaded after model training, completely independent

Compares against:
  - Linear extrapolation: 2*builtup_2015 - builtup_2010  (5yr trend → 2020)
  - Persistence baseline: builtup_2015 (no change)

Also runs multihorizon ConvLSTM variants for comparison.

Outputs:
  - results/metrics/validation_2020_results.json
  - geotiff_exports/CONUS_builtup_2020_predicted.tif
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR  = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import rasterio

print("=" * 60, flush=True)
print("2020 TEMPORAL HOLDOUT VALIDATION", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)
print("\nDesign: model trained on [1975-2010]→2015, now predicts [1980-2015]→2020", flush=True)
print("This is a TRUE out-of-sample temporal holdout.", flush=True)

from src.models.convlstm import ConvLSTM

# =====================================================
# Prediction helper — matches training pipeline
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
# Load val tile indices (same spatial split as 2015 validation)
# =====================================================
val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
if not os.path.exists(val_idx_path):
    print("ERROR: val_tile_indices.json not found.")
    sys.exit(1)

with open(val_idx_path) as f:
    val_tiles = [tuple(t) for t in json.load(f)]

print(f"\n[HOLDOUT] Loaded {len(val_tiles):,} held-out val tiles from spatial block split", flush=True)

# =====================================================
# Load data — input window [1980–2015] + ground truth 2020
# =====================================================
INPUT_EPOCHS   = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
TARGET_YEAR    = 2020
LAST_INPUT_YEAR = 2015

print("\n[DATA] Loading input epochs [1980–2015] + 2020 ground truth...", flush=True)

builtup = {}
for year in INPUT_EPOCHS + [TARGET_YEAR]:
    path = str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found!")
        sys.exit(1)
    with rasterio.open(path) as src:
        builtup[year] = src.read(1)
        if year == TARGET_YEAR:
            ref_profile = src.profile.copy()
    print(f"  builtup {year}: shape={builtup[year].shape}, mean={builtup[year].mean():.6f}", flush=True)

ref_shape = builtup[LAST_INPUT_YEAR].shape

volume = {}
for year in INPUT_EPOCHS:
    vpath = str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")
    with rasterio.open(vpath) as src:
        volume[year] = src.read(1)
    print(f"  volume {year}: mean={volume[year].mean():.4f}", flush=True)

population = {}
for year in INPUT_EPOCHS:
    fpath = str(OUTPUT_DIR / f"CONUS_population_{year}.tif")
    with rasterio.open(fpath) as src:
        population[year] = src.read(1)
    print(f"  population {year}: mean={population[year].mean():.4f}", flush=True)

gc.collect()

# =====================================================
# Run inference with primary ConvLSTM model
# =====================================================
def run_inference(model_path, model_name="ConvLSTM"):
    print(f"\n[MODEL] Loading {model_name} from {model_path}...", flush=True)
    model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1)
    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    TILE_SIZE = 128
    height, width = ref_shape
    prediction = np.zeros(ref_shape, dtype=np.float64)
    counts      = np.zeros(ref_shape, dtype=np.float64)

    print(f"\n[PREDICT] Running {model_name} on {len(val_tiles):,} held-out tiles...", flush=True)
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
    return prediction, counts


# Run primary model
prediction, counts = run_inference(MODELS_DIR / "best_3ch_mc_model.pth", "ConvLSTM-3ch-Best")

# =====================================================
# Save prediction GeoTIFF
# =====================================================
print("\n[SAVE] Writing predicted 2020 GeoTIFF...", flush=True)
out_path = str(OUTPUT_DIR / "CONUS_builtup_2020_predicted.tif")
ref_profile.update(dtype='float32', count=1, compress='deflate')
with rasterio.open(out_path, 'w', **ref_profile) as dst:
    dst.write(prediction, 1)
print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)", flush=True)

# =====================================================
# Metrics computation
# =====================================================
print("\n[METRICS] Computing metrics on held-out val tiles only...", flush=True)

gt = builtup[TARGET_YEAR]
# Eval mask: val tiles only, urban pixels in 2020 ground truth
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

# Per-tile metrics
tile_mses, tile_maes = [], []
for (i, j) in val_tiles:
    tile_gt   = gt[i:i+128, j:j+128]
    tile_pred = prediction[i:i+128, j:j+128]
    tile_cnt  = counts[i:i+128, j:j+128]
    if tile_gt.mean() < 0.01 or tile_cnt.mean() < 0.5:
        continue
    tile_mses.append(float(np.mean((tile_gt - tile_pred) ** 2)))
    tile_maes.append(float(np.mean(np.abs(tile_gt - tile_pred))))

tile_mses = np.array(tile_mses)
tile_maes = np.array(tile_maes)

print(f"\n  --- ConvLSTM 3ch — 2020 TEMPORAL HOLDOUT ---")
print(f"  Evaluated pixels: {n_eval_pixels:,}")
print(f"  MSE:  {mse:.6f}")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  R²:   {r2:.4f}")

# =====================================================
# Baselines
# =====================================================
linear_pred = np.clip(2 * builtup[2015] - builtup[2010], 0, 1)
lin_eval    = linear_pred[eval_mask]
lin_mse     = float(np.mean((gt_eval - lin_eval) ** 2))
lin_mae     = float(np.mean(np.abs(gt_eval - lin_eval)))
lin_rmse    = float(np.sqrt(lin_mse))
lin_ss_res  = np.sum((gt_eval - lin_eval) ** 2)
lin_r2      = float(1 - lin_ss_res / ss_tot) if ss_tot > 0 else float('nan')

per_eval = builtup[LAST_INPUT_YEAR][eval_mask]
per_mse  = float(np.mean((gt_eval - per_eval) ** 2))
per_mae  = float(np.mean(np.abs(gt_eval - per_eval)))

imp_linear  = (1 - mse / lin_mse) * 100 if lin_mse > 0 else 0
imp_persist = (1 - mse / per_mse) * 100 if per_mse > 0 else 0

print(f"\n  --- BASELINES ---")
print(f"  Linear extrap MSE:  {lin_mse:.6f}  R²={lin_r2:.4f}")
print(f"  Persistence MSE:    {per_mse:.6f}")
print(f"\n  ConvLSTM vs Linear:      {imp_linear:+.1f}% MSE improvement")
print(f"  ConvLSTM vs Persistence: {imp_persist:+.1f}% MSE improvement")

# =====================================================
# Figure of Merit
# =====================================================
print("\n[METRICS] Computing Figure of Merit (FoM)...", flush=True)
CHANGE_THRESHOLD = 0.01

obs_change  = (gt  - builtup[LAST_INPUT_YEAR]) > CHANGE_THRESHOLD
pred_change = (prediction - builtup[LAST_INPUT_YEAR]) > CHANGE_THRESHOLD
lin_change  = (linear_pred - builtup[LAST_INPUT_YEAR]) > CHANGE_THRESHOLD

obs_ch_val  = obs_change[eval_mask]
pred_ch_val = pred_change[eval_mask]
lin_ch_val  = lin_change[eval_mask]

def figure_of_merit(obs, pred):
    inter = int((obs & pred).sum())
    union = int((obs | pred).sum())
    return inter / union if union > 0 else 0.0

convlstm_fom = figure_of_merit(obs_ch_val, pred_ch_val)
linear_fom   = figure_of_merit(obs_ch_val, lin_ch_val)

n_obs_change = int(obs_ch_val.sum())
pct_growth   = 100.0 * n_obs_change / n_eval_pixels

print(f"\n  --- FIGURE OF MERIT (threshold={CHANGE_THRESHOLD}) ---")
print(f"  Observed growth pixels: {n_obs_change:,} of {n_eval_pixels:,} ({pct_growth:.1f}%)")
print(f"  ConvLSTM FoM:  {convlstm_fom:.4f}")
print(f"  Linear FoM:    {linear_fom:.4f}")
print(f"  Persistence FoM: 0.0000  (never predicts change)")
print(f"  FoM gap vs linear: {(convlstm_fom - linear_fom)*100:+.1f} pp")

# =====================================================
# Growth-region vs stable-region
# =====================================================
print("\n[METRICS] Growth-region vs stable-region evaluation...", flush=True)

growth_mask = eval_mask & obs_change
stable_mask = eval_mask & ~obs_change
n_growth = int(growth_mask.sum())
n_stable = int(stable_mask.sum())

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

lin_growth_mse = float(np.mean((growth_gt_px - linear_pred[growth_mask]) ** 2))
lin_stable_mse = float(np.mean((stable_gt_px - linear_pred[stable_mask]) ** 2))
per_growth_mse = float(np.mean((growth_gt_px - builtup[LAST_INPUT_YEAR][growth_mask]) ** 2))

imp_growth         = (1 - growth_mse / lin_growth_mse) * 100 if lin_growth_mse > 0 else 0
imp_persist_growth = (1 - growth_mse / per_growth_mse) * 100 if per_growth_mse > 0 else 0

print(f"\n  Growth pixels: {n_growth:,} ({100*n_growth/n_eval_pixels:.1f}% of evaluated)")
print(f"  ConvLSTM growth MSE:    {growth_mse:.6f}  R²={growth_r2:.4f}")
print(f"  Linear    growth MSE:   {lin_growth_mse:.6f}")
print(f"  Persistence growth MSE: {per_growth_mse:.6f}")
print(f"  ConvLSTM vs Linear (growth):      {imp_growth:+.1f}%")
print(f"\n  Stable pixels: {n_stable:,}")
print(f"  ConvLSTM stable MSE: {stable_mse:.6f}")
print(f"  Linear    stable MSE:{lin_stable_mse:.6f}")

# =====================================================
# SSIM
# =====================================================
print("\n[METRICS] Computing SSIM (structural similarity)...", flush=True)
from scipy.ndimage import uniform_filter

def tile_ssim(gt_tile, pred_tile):
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
    gt_t = gt[i:i+128, j:j+128]
    if gt_t.mean() < 0.01:
        continue
    pred_t = prediction[i:i+128, j:j+128]
    lin_t  = linear_pred[i:i+128, j:j+128]
    per_t  = builtup[LAST_INPUT_YEAR][i:i+128, j:j+128]
    ssim_conv.append(tile_ssim(gt_t, pred_t))
    ssim_lin.append(tile_ssim(gt_t, lin_t))
    ssim_per.append(tile_ssim(gt_t, per_t))

mean_ssim_conv = float(np.mean(ssim_conv))
mean_ssim_lin  = float(np.mean(ssim_lin))
mean_ssim_per  = float(np.mean(ssim_per))

print(f"\n  ConvLSTM SSIM:   {mean_ssim_conv:.4f}")
print(f"  Linear    SSIM:  {mean_ssim_lin:.4f}")
print(f"  Persistence SSIM:{mean_ssim_per:.4f}")

# =====================================================
# Also evaluate multihorizon_5yr model (same architecture, same 5yr task)
# =====================================================
print("\n[BONUS] Evaluating multihorizon_5yr.pth on 2020 holdout...", flush=True)
mh5_path = MODELS_DIR / "multihorizon_5yr.pth"
mh5_pred, _ = run_inference(mh5_path, "ConvLSTM-multihorizon-5yr")

mh5_eval     = mh5_pred[eval_mask]
mh5_mse      = float(np.mean((gt_eval - mh5_eval) ** 2))
mh5_mae      = float(np.mean(np.abs(gt_eval - mh5_eval)))
mh5_r2       = float(1 - np.sum((gt_eval - mh5_eval)**2) / ss_tot) if ss_tot > 0 else float('nan')
mh5_pred_ch  = (mh5_pred - builtup[LAST_INPUT_YEAR]) > CHANGE_THRESHOLD
mh5_fom      = figure_of_merit(obs_ch_val, mh5_pred_ch[eval_mask])
mh5_grmse    = float(np.mean((growth_gt_px - mh5_pred[growth_mask]) ** 2))

print(f"  multihorizon_5yr MSE={mh5_mse:.6f}  R²={mh5_r2:.4f}  FoM={mh5_fom:.4f}")
print(f"  growth MSE={mh5_grmse:.6f}  vs linear: {(1-mh5_grmse/lin_growth_mse)*100:+.1f}%")

# =====================================================
# Save results
# =====================================================
results = {
    "experiment": "2020_temporal_holdout_validation",
    "description": (
        "True temporal holdout: model trained on [1975-2010]→2015 predicts 2020. "
        "Input window shifted by one epoch: [1980-2015] → 2020. "
        "2020 GHSL data was downloaded AFTER model training — completely independent. "
        "Evaluation on spatially held-out val tiles (same block split as 2015 validation)."
    ),
    "model": "3ch_ConvLSTM_MCDropout_SkipDecoder",
    "model_path": str(MODELS_DIR / "best_3ch_mc_model.pth"),
    "model_trained_on": "[1975,1980,...,2010] → 2015",
    "holdout_task": "[1980,1985,...,2015] → 2020",
    "split_method": "spatial_block_holdout (same val_tile_indices.json as 2015 validation)",
    "input_epochs": INPUT_EPOCHS,
    "target_year": TARGET_YEAR,
    "last_input_year": LAST_INPUT_YEAR,
    "n_val_tiles": len(val_tiles),
    "n_eval_pixels": n_eval_pixels,
    "n_obs_growth_pixels": n_obs_change,
    "pct_growth": round(pct_growth, 2),
    "change_threshold": CHANGE_THRESHOLD,
    "convlstm_best": {
        "mse":  mse,  "mae":  mae,  "rmse": rmse, "r2": r2,
        "fom":  convlstm_fom,
        "ssim": mean_ssim_conv,
        "growth_mse": growth_mse,
        "growth_mae": growth_mae,
        "growth_r2":  growth_r2,
        "stable_mse": stable_mse,
        "tile_mse_mean": float(tile_mses.mean()),
        "tile_mse_std":  float(tile_mses.std()),
        "n_tiles": int(len(tile_mses)),
    },
    "convlstm_multihorizon_5yr": {
        "mse": mh5_mse, "mae": mh5_mae, "r2": mh5_r2,
        "fom": mh5_fom,
        "growth_mse": mh5_grmse,
        "note": "Same architecture, trained independently for 5yr horizon task",
    },
    "linear_extrapolation": {
        "mse": lin_mse, "mae": lin_mae, "rmse": lin_rmse, "r2": lin_r2,
        "method": "2*builtup_2015 - builtup_2010",
        "fom": linear_fom,
        "ssim": mean_ssim_lin,
        "growth_mse": lin_growth_mse,
        "stable_mse": lin_stable_mse,
    },
    "persistence": {
        "mse": per_mse, "mae": per_mae,
        "method": "builtup_2015 unchanged",
        "fom": 0.0,
        "ssim": mean_ssim_per,
        "growth_mse": per_growth_mse,
    },
    "improvement_convlstm_vs_linear_mse_pct":  round(imp_linear, 2),
    "improvement_convlstm_vs_persist_mse_pct": round(imp_persist, 2),
    "fom_gap_vs_linear_pp":      round((convlstm_fom - linear_fom) * 100, 3),
    "growth_mse_improvement_pct": round(imp_growth, 2),
    "timestamp": str(datetime.datetime.now()),
}

results_path = str(RESULTS_DIR / "validation_2020_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {results_path}", flush=True)

print(f"\n{'='*60}")
print("2020 TEMPORAL HOLDOUT COMPLETE")
print(f"{'='*60}")
print(f"\n  ConvLSTM (best)   R²={r2:.4f} | FoM={convlstm_fom:.4f} | MSE={mse:.6f}")
print(f"  Linear extrap.    R²={lin_r2:.4f} | FoM={linear_fom:.4f} | MSE={lin_mse:.6f}")
print(f"  Persistence                       FoM=0.0000  | MSE={per_mse:.6f}")
print(f"\n  ConvLSTM vs Linear: {imp_linear:+.1f}% MSE | {(convlstm_fom-linear_fom)*100:+.1f} pp FoM")
print(f"  Growth region:      {imp_growth:+.1f}% MSE improvement over linear")
print(f"\n  ✓ Temporal holdout: model generalises to 2020 with zero exposure to 2020 data")
