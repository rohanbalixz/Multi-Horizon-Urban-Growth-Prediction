#!/usr/bin/env python3
"""
Re-evaluate Multi-Horizon Models with Consistent Urban Pixel Mask
==================================================================
Fixes the methodological inconsistency in the original train_multihorizon.py
where ConvLSTM MSE was computed over ALL tile pixels but linear baseline MSE
was computed over only urban pixels (gt > 0.01 | pred > 0.01).

This script:
  1. Loads already-trained horizon models (no retraining)
  2. Runs inference on the same val tiles (val_tile_indices.json)
  3. Computes ConvLSTM MSE AND linear baseline MSE on the same mask:
       mask = (gt > 0.01) | (convlstm_pred > 0.01)
     — identical to validate_2015.py's eval_mask
  4. Rewrites multihorizon_results.json with corrected numbers
  5. Calls consolidate_results.py to sync all_results.json

FoM was already consistent (same tile-level computation for both models)
and is preserved unchanged.
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR  = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

print("=" * 60, flush=True)
print("MULTI-HORIZON RE-EVALUATION (consistent urban pixel mask)", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import rasterio

from src.models.convlstm import ConvLSTM

TARGET_EPOCH     = 2015
TILE_SIZE        = 128
CHANGE_THRESHOLD = 0.01

HORIZONS = {
    "5yr":  {"epochs": [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010], "years_ahead": 5},
    "10yr": {"epochs": [1975, 1980, 1985, 1990, 1995, 2000, 2005],       "years_ahead": 10},
    "20yr": {"epochs": [1975, 1980, 1985, 1990, 1995],                   "years_ahead": 20},
}

# =====================================================
# Load val tile indices
# =====================================================
val_idx_path = RESULTS_DIR / "val_tile_indices.json"
if not val_idx_path.exists():
    print("ERROR: val_tile_indices.json not found. Run train_3channel.py first.")
    sys.exit(1)
with open(val_idx_path) as f:
    val_tiles = [tuple(t) for t in json.load(f)]
print(f"\n[HOLDOUT] {len(val_tiles):,} val tiles loaded", flush=True)

# =====================================================
# Load all data
# =====================================================
ALL_EPOCHS = sorted(set(
    e for cfg in HORIZONS.values() for e in cfg["epochs"]
) | {TARGET_EPOCH})

print("\n[DATA] Loading all data channels...", flush=True)
ghsl, volume, population = {}, {}, {}

for year in ALL_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")) as src:
        ghsl[year] = src.read(1)
    print(f"  Builtup {year}: mean={ghsl[year].mean():.6f}", flush=True)

for year in [e for cfg in HORIZONS.values() for e in cfg["epochs"]]:
    if year not in volume:
        with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")) as src:
            volume[year] = src.read(1)
    if year not in population:
        with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{year}.tif")) as src:
            population[year] = src.read(1)

print("  All channels loaded.", flush=True)
gc.collect()

gt_map = ghsl[TARGET_EPOCH]


# =====================================================
# Autoregressive future-step prediction
# =====================================================
def predict_future_step(model, sequences):
    _, hidden_states = model(sequences)
    last_input = sequences[:, -1]
    x_t = last_input
    layer_hiddens = []
    for layer_idx, layer in enumerate(model.convlstm_layers):
        h, c = hidden_states[layer_idx]
        h, c = layer(x_t, (h, c))
        h    = model.mc_dropouts[layer_idx](h)
        hidden_states[layer_idx] = (h, c)
        x_t  = h
        layer_hiddens.append(h)
    return model.decode(layer_hiddens)


# =====================================================
# Pixel-wise OLS linear baseline (same as original)
# Applied to same urban pixel mask as ConvLSTM.
# =====================================================
def linear_prediction_tile(input_epochs, i, j):
    """Pixel-wise OLS regression through input epochs, projected to 2015."""
    years   = np.array(input_epochs, dtype=np.float64)
    t_target = float(TARGET_EPOCH)
    stack   = np.stack([ghsl[y][i:i+TILE_SIZE, j:j+TILE_SIZE].flatten()
                        for y in input_epochs])   # (T, HW)
    t_mean   = years.mean()
    t_denom  = np.sum((years - t_mean) ** 2)
    b        = np.sum((stack - stack.mean(axis=0)) * (years - t_mean)[:, None], axis=0) / t_denom
    a        = stack.mean(axis=0) - b * t_mean
    pred_px  = (a + b * t_target).clip(0, 1).reshape(TILE_SIZE, TILE_SIZE)
    return pred_px


# =====================================================
# Evaluate one horizon
# =====================================================
def evaluate_horizon(horizon_name, input_epochs, years_ahead):
    print(f"\n{'='*60}", flush=True)
    print(f"HORIZON: {horizon_name}  (last input={input_epochs[-1]} → 2015)", flush=True)

    model_path = MODELS_DIR / f"multihorizon_{horizon_name}.pth"
    if not model_path.exists():
        print(f"  ERROR: {model_path} not found. Run train_multihorizon.py first.")
        sys.exit(1)

    model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1)
    ckpt  = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"  Loaded {model_path.name}  ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)

    last_input_map = ghsl[input_epochs[-1]]

    # Accumulate pixels for all val tiles
    all_gt, all_conv, all_lin, all_last_bu = [], [], [], []
    t0 = time.time()

    for n, (i, j) in enumerate(val_tiles):
        tile_gt = gt_map[i:i+TILE_SIZE, j:j+TILE_SIZE]
        if tile_gt.mean() < 0.01:
            continue   # skip non-urban tiles (same filter as original)

        # --- ConvLSTM prediction ---
        frames = []
        for epoch in input_epochs:
            bu  = ghsl[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
            vol = volume[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
            pop = population[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
            frames.append(np.stack([bu, vol, pop], axis=0))
        seq = torch.FloatTensor(np.stack(frames, axis=0)).unsqueeze(0)
        with torch.no_grad():
            conv_pred = predict_future_step(model, seq).squeeze().numpy().clip(0, 1)

        # --- Linear OLS prediction ---
        lin_pred = linear_prediction_tile(input_epochs, i, j)

        # --- Urban pixel mask: same for both models ---
        # Defined by ground truth OR ConvLSTM prediction (identical to validate_2015.py)
        mask = (tile_gt > CHANGE_THRESHOLD) | (conv_pred > CHANGE_THRESHOLD)
        if mask.sum() == 0:
            continue

        all_gt.append(tile_gt[mask])
        all_conv.append(conv_pred[mask])
        all_lin.append(lin_pred[mask])
        all_last_bu.append(last_input_map[i:i+TILE_SIZE, j:j+TILE_SIZE][mask])

        if (n + 1) % 200 == 0:
            print(f"    {n+1}/{len(val_tiles)} tiles, {time.time()-t0:.0f}s", flush=True)

    all_gt      = np.concatenate(all_gt)
    all_conv    = np.concatenate(all_conv)
    all_lin     = np.concatenate(all_lin)
    all_last_bu = np.concatenate(all_last_bu)

    n_pixels = len(all_gt)
    print(f"  Urban pixels evaluated: {n_pixels:,}", flush=True)

    # --- MSE / MAE / RMSE / R² (consistent mask for both) ---
    def metrics(gt, pred, label):
        mse  = float(np.mean((gt - pred) ** 2))
        mae  = float(np.mean(np.abs(gt - pred)))
        rmse = float(np.sqrt(mse))
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - gt.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        print(f"  {label}: MSE={mse:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}", flush=True)
        return {"mse": round(mse, 6), "mae": round(mae, 6),
                "rmse": round(rmse, 6), "r2": round(r2, 4)}

    conv_m = metrics(all_gt, all_conv, "ConvLSTM")
    lin_m  = metrics(all_gt, all_lin,  "Linear  ")

    # --- FoM (unchanged — already consistent) ---
    obs_ch   = (all_gt      - all_last_bu) > CHANGE_THRESHOLD
    pred_ch  = (all_conv    - all_last_bu) > CHANGE_THRESHOLD
    lin_ch   = (all_lin     - all_last_bu) > CHANGE_THRESHOLD

    def fom(obs, pred):
        inter = int((obs & pred).sum())
        union = int((obs | pred).sum())
        return round(inter / union, 4) if union > 0 else 0.0

    conv_fom = fom(obs_ch, pred_ch)
    lin_fom  = fom(obs_ch, lin_ch)
    print(f"  ConvLSTM FoM={conv_fom:.4f}  |  Linear FoM={lin_fom:.4f}", flush=True)

    # --- Improvement (now valid: same mask) ---
    imp = round((1 - conv_m["mse"] / lin_m["mse"]) * 100, 2) if lin_m["mse"] > 0 else 0.0
    sign = "better" if imp > 0 else "worse"
    print(f"  ConvLSTM vs Linear MSE: {imp:+.1f}% ({sign})", flush=True)

    del model; gc.collect()

    return {
        "horizon": horizon_name,
        "years_ahead": years_ahead,
        "input_epochs": input_epochs,
        "last_input_year": input_epochs[-1],
        "target_year": TARGET_EPOCH,
        "n_input_epochs": len(input_epochs),
        "n_urban_pixels_evaluated": n_pixels,
        "eval_mask": "(gt > 0.01) | (convlstm_pred > 0.01)  — identical to validate_2015.py",
        "convlstm": {
            **conv_m,
            "val_fom": conv_fom,
            "params": 481153,
        },
        "linear_regression_baseline": {
            **lin_m,
            "fom": lin_fom,
            "method": "pixel-wise OLS through all input epochs projected to 2015",
        },
        "improvement_vs_linear_pct": imp,
        "model_path": str(MODELS_DIR / f"multihorizon_{horizon_name}.pth"),
    }


# =====================================================
# Run all horizons
# =====================================================
corrected = {}
for name, cfg in HORIZONS.items():
    corrected[name] = evaluate_horizon(name, cfg["epochs"], cfg["years_ahead"])
    gc.collect()

# =====================================================
# Summary
# =====================================================
print(f"\n{'='*60}", flush=True)
print("CORRECTED MULTI-HORIZON SUMMARY (consistent urban pixel mask)", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Horizon':<8} {'ConvLSTM MSE':>14} {'Linear MSE':>12} {'Improvement':>12} {'ConvR²':>8} {'ConvFoM':>9} {'LinFoM':>9}", flush=True)
print("-" * 75, flush=True)
for name, r in corrected.items():
    print(
        f"{name:<8} {r['convlstm']['mse']:>14.6f} "
        f"{r['linear_regression_baseline']['mse']:>12.6f} "
        f"{r['improvement_vs_linear_pct']:>+11.1f}% "
        f"{r['convlstm']['r2']:>8.4f} "
        f"{r['convlstm']['val_fom']:>9.4f} "
        f"{r['linear_regression_baseline']['fom']:>9.4f}",
        flush=True,
    )

# =====================================================
# Write corrected multihorizon_results.json
# =====================================================
output = {
    "experiment": "multihorizon_forecasting",
    "description": (
        "ConvLSTM trained separately at 5/10/20-year horizons, all targeting GHSL 2015. "
        "Linear baseline: pixel-wise OLS regression projected to 2015. "
        "BOTH models evaluated on the same urban pixel mask: "
        "(gt > 0.01) | (convlstm_pred > 0.01), identical to validate_2015.py. "
        "Previous improvement_vs_linear_pct values (81-83%) were invalid due to "
        "inconsistent masks; this file contains the corrected values."
    ),
    "target_epoch": TARGET_EPOCH,
    "split_method": "spatial_block_holdout (val_tile_indices.json)",
    "architecture": "ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1)",
    "eval_mask": "(gt > 0.01) | (convlstm_pred > 0.01)",
    "horizons": corrected,
    "timestamp": datetime.datetime.now().isoformat(),
}

out_path = RESULTS_DIR / "multihorizon_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nCorrected results saved to {out_path}", flush=True)

# =====================================================
# Sync all_results.json
# =====================================================
from scripts.consolidate_results import consolidate
consolidate()

print(f"\n{'='*60}", flush=True)
print("MULTI-HORIZON RE-EVALUATION COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
