#!/usr/bin/env python3
"""
SLEUTH 2020 Temporal Holdout
=============================
Uses already-calibrated SLEUTH parameters (from sota_baselines_results.json)
to predict 2020 built-up patterns from 2015 as the single input step.

No recalibration. Just inference: builtup[2015] → predict 2020.

This mirrors the DL 2020 holdout design:
  - DL models: trained on [1975-2010]→2015, predict [1980-2015]→2020
  - SLEUTH:    calibrated on [1975-2000] chain, predict 2015→2020 (one step)

Uses the same val_tile_indices.json and eval_mask as all other experiments.
Compares against the same linear and persistence baselines.

Outputs:
  - results/metrics/sleuth_2020_holdout.json

Runtime: ~5 minutes
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR  = PROJECT_ROOT / "geotiff_exports"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("SLEUTH 2020 TEMPORAL HOLDOUT", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import rasterio

from src.models.sleuth import SLEUTH

# ── Load calibrated SLEUTH parameters ─────────────────────────
sota_path = str(RESULTS_DIR / "sota_baselines_results.json")
with open(sota_path) as f:
    sota = json.load(f)
params = sota["sleuth_ca"]["calibrated_params"]
print(f"\n[SLEUTH] Calibrated parameters: {params}", flush=True)

model = SLEUTH(
    dispersion  = params["dispersion"],
    breed       = params["breed"],
    spread      = params["spread"],
    road_gravity= params["road_gravity"],
    random_seed = 42,
)
model.calibrated = True

# ── Load val tile indices ──────────────────────────────────────
val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
if not os.path.exists(val_idx_path):
    print("ERROR: val_tile_indices.json not found. Run train_3channel.py first.")
    sys.exit(1)
with open(val_idx_path) as f:
    val_tiles = [tuple(t) for t in json.load(f)]
print(f"[SPLIT] Loaded {len(val_tiles):,} val tiles from spatial block holdout.", flush=True)

# ── Load data ─────────────────────────────────────────────────
print("\n[DATA] Loading builtup 2015, 2020 + volume/population 2015...", flush=True)

with rasterio.open(str(OUTPUT_DIR / "CONUS_builtup_2015.tif")) as src:
    builtup_2015 = src.read(1)
print(f"  builtup 2015: shape={builtup_2015.shape}, mean={builtup_2015.mean():.6f}", flush=True)

with rasterio.open(str(OUTPUT_DIR / "CONUS_builtup_2020.tif")) as src:
    builtup_2020 = src.read(1)
print(f"  builtup 2020: shape={builtup_2020.shape}, mean={builtup_2020.mean():.6f}", flush=True)

with rasterio.open(str(OUTPUT_DIR / "CONUS_builtup_2010.tif")) as src:
    builtup_2010 = src.read(1)
print(f"  builtup 2010: mean={builtup_2010.mean():.6f}", flush=True)

# Volume as road proxy (same as SLEUTH 2015 evaluation)
with rasterio.open(str(OUTPUT_DIR / "CONUS_volume_2015.tif")) as src:
    volume_2015 = src.read(1)
print(f"  volume 2015:  mean={volume_2015.mean():.6f}", flush=True)

with rasterio.open(str(OUTPUT_DIR / "CONUS_population_2015.tif")) as src:
    population_2015 = src.read(1)
print(f"  population 2015: mean={population_2015.mean():.6f}", flush=True)

gc.collect()
ref_shape = builtup_2015.shape

# ── Run SLEUTH inference on val tiles ─────────────────────────
print(f"\n[PREDICT] Running SLEUTH on {len(val_tiles):,} val tiles (2015 → 2020)...", flush=True)

TILE_SIZE  = 128
prediction = np.zeros(ref_shape, dtype=np.float64)
counts     = np.zeros(ref_shape, dtype=np.float64)

t0 = time.time()
for n, (i, j) in enumerate(val_tiles):
    bu_tile  = builtup_2015[i:i+TILE_SIZE, j:j+TILE_SIZE].astype(np.float32)
    vol_tile = volume_2015[i:i+TILE_SIZE, j:j+TILE_SIZE].astype(np.float32)
    pop_tile = population_2015[i:i+TILE_SIZE, j:j+TILE_SIZE].astype(np.float32)

    # One step: 2015 → 2020
    pred_tile = model.predict_step(bu_tile, roads=vol_tile, population=pop_tile)

    prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred_tile
    counts[i:i+TILE_SIZE, j:j+TILE_SIZE] += 1

    if (n + 1) % 200 == 0:
        print(f"  {n+1}/{len(val_tiles)} tiles [{time.time()-t0:.0f}s]", flush=True)

mask = counts > 0
prediction[mask] /= counts[mask]
prediction = prediction.astype(np.float32)
inference_min = (time.time() - t0) / 60
print(f"  Done in {inference_min:.1f} min.", flush=True)

# ── Compute metrics ───────────────────────────────────────────
print("\n[METRICS] Computing 2020 holdout metrics...", flush=True)

gt         = builtup_2020
eval_mask  = (counts > 0) & (gt > 0.01)
n_eval_px  = int(eval_mask.sum())

gt_eval   = gt[eval_mask]
pred_eval = prediction[eval_mask]

sleuth_mse  = float(np.mean((gt_eval - pred_eval) ** 2))
sleuth_mae  = float(np.mean(np.abs(gt_eval - pred_eval)))
sleuth_rmse = float(np.sqrt(sleuth_mse))
ss_res = np.sum((gt_eval - pred_eval) ** 2)
ss_tot = np.sum((gt_eval - gt_eval.mean()) ** 2)
sleuth_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

# Baselines
linear_pred  = np.clip(2 * builtup_2015 - builtup_2010, 0, 1)
persist_pred = builtup_2015

lin_eval  = linear_pred[eval_mask]
per_eval  = persist_pred[eval_mask]

lin_mse  = float(np.mean((gt_eval - lin_eval) ** 2))
per_mse  = float(np.mean((gt_eval - per_eval) ** 2))

# FoM
LAST_INPUT_YEAR  = 2015
CHANGE_THRESHOLD = 0.01

obs_change    = (gt           - builtup_2015) > CHANGE_THRESHOLD
pred_change   = (prediction   - builtup_2015) > CHANGE_THRESHOLD
lin_change    = (linear_pred  - builtup_2015) > CHANGE_THRESHOLD

obs_ch_val  = obs_change[eval_mask]
pred_ch_val = pred_change[eval_mask]
lin_ch_val  = lin_change[eval_mask]

def fom(obs, pred):
    inter = int((obs & pred).sum())
    union = int((obs | pred).sum())
    return inter / union if union > 0 else 0.0

sleuth_fom_2020 = fom(obs_ch_val, pred_ch_val)
linear_fom_2020 = fom(obs_ch_val, lin_ch_val)
n_growth = int(obs_ch_val.sum())
pct_growth = 100.0 * n_growth / n_eval_px

# Growth-zone vs stable-zone MSE
growth_mask = eval_mask & obs_change
stable_mask = eval_mask & ~obs_change

sleuth_growth_mse = float(np.mean((gt[growth_mask] - prediction[growth_mask]) ** 2))
lin_growth_mse    = float(np.mean((gt[growth_mask] - linear_pred[growth_mask]) ** 2))
per_growth_mse    = float(np.mean((gt[growth_mask] - persist_pred[growth_mask]) ** 2))

print(f"\n  === SLEUTH 2020 TEMPORAL HOLDOUT RESULTS ===")
print(f"  Eval pixels:        {n_eval_px:,}")
print(f"  Growth pixels:      {n_growth:,} ({pct_growth:.2f}%)")
print(f"")
print(f"  SLEUTH FoM={sleuth_fom_2020:.4f}  MSE={sleuth_mse:.6f}  R²={sleuth_r2:.4f}")
print(f"  Linear FoM={linear_fom_2020:.4f}  MSE={lin_mse:.6f}")
print(f"  Persist             MSE={per_mse:.6f}  FoM=0.0000")
print(f"")
print(f"  Linear vs SLEUTH FoM: {linear_fom_2020/sleuth_fom_2020:.2f}× better" if sleuth_fom_2020 > 0 else "  SLEUTH FoM=0")
print(f"  SLEUTH growth MSE: {sleuth_growth_mse:.6f} vs Linear {lin_growth_mse:.6f}")

# ── Save results ──────────────────────────────────────────────
results = {
    "experiment": "sleuth_2020_temporal_holdout",
    "description": (
        "SLEUTH CA with calibrated parameters (from sota_baselines_results.json) "
        "predicts 2020 from 2015 in one step. Same val_tile_indices.json and "
        "eval_mask as all other 2020 holdout experiments."
    ),
    "calibrated_params": params,
    "calibration_source": "sota_baselines_results.json (calibrated on 1975→1990→2000 chain)",
    "prediction_step": "2015 → 2020 (one step, same 5yr interval as training steps)",
    "holdout_2020": {
        "n_eval_pixels": n_eval_px,
        "n_growth_pixels": n_growth,
        "pct_growth_pixels": round(pct_growth, 2),
        "sleuth": {
            "fom": round(sleuth_fom_2020, 4),
            "mse": round(sleuth_mse, 6),
            "mae": round(sleuth_mae, 6),
            "r2": round(sleuth_r2, 4),
            "growth_mse": round(sleuth_growth_mse, 6),
        },
        "linear": {
            "fom": round(linear_fom_2020, 4),
            "mse": round(lin_mse, 6),
            "growth_mse": round(lin_growth_mse, 6),
        },
        "persistence": {
            "fom": 0.0,
            "mse": round(per_mse, 6),
            "growth_mse": round(per_growth_mse, 6),
        },
    },
    "timestamp": datetime.datetime.now().isoformat(),
}

out_path = str(RESULTS_DIR / "sleuth_2020_holdout.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[DONE] Results saved to {out_path}", flush=True)
print(f"[DONE] Total time: {inference_min:.1f} min", flush=True)
