#!/usr/bin/env python3
"""
SLEUTH Cellular Automata Baseline
==================================
Calibrates and evaluates SLEUTH CA on the same spatial block holdout
(val_tile_indices.json) used by all DL models — providing the
30-year domain-standard comparison for NeuralTimeCapsule.

Clarke et al. (1997), SLEUTH: the canonical urban growth CA model.

Output: results/metrics/sota_baselines_results.json
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

print("=" * 60, flush=True)
print("SLEUTH CELLULAR AUTOMATA BASELINE", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

from src.models.sleuth import SLEUTH

# =====================================================
# Config
# =====================================================
TILE_SIZE  = 128
BATCH_SIZE = 8

TRAIN_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH = 2015

# =====================================================
# Load data
# =====================================================
print("\n[DATA] Loading all data channels...", flush=True)

ghsl = {}
for year in TRAIN_EPOCHS + [TARGET_EPOCH]:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")) as src:
        ghsl[year] = src.read(1)
    print(f"  Built-up {year}: shape={ghsl[year].shape}, mean={ghsl[year].mean():.6f}", flush=True)

ref_shape = ghsl[TARGET_EPOCH].shape

volume = {}
for year in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")) as src:
        volume[year] = src.read(1)

population = {}
for year in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{year}.tif")) as src:
        population[year] = src.read(1)

gc.collect()
print("  All data loaded.\n", flush=True)

# =====================================================
# Dataset — needed to build val_loader for SLEUTH eval
# (same class as train_3channel.py for split consistency)
# =====================================================
class ThreeChannelDataset(Dataset):
    def __init__(self, temporal_stack, volume, population, tile_size=128):
        self.temporal_stack = temporal_stack
        self.volume         = volume
        self.population     = population
        self.tile_size      = tile_size
        self.epochs         = sorted(temporal_stack.keys())
        height, width = list(temporal_stack.values())[0].shape
        self.tiles = []
        stride = tile_size // 2
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                if temporal_stack[self.epochs[-1]][i:i+tile_size, j:j+tile_size].mean() > 0.01:
                    self.tiles.append((i, j))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts = self.tile_size
        sequence = []
        input_epochs = [e for e in self.epochs if e != TARGET_EPOCH]
        for epoch in input_epochs:
            builtup = self.temporal_stack[epoch][i:i+ts, j:j+ts]
            vol     = self.volume[epoch][i:i+ts, j:j+ts]
            pop     = self.population[epoch][i:i+ts, j:j+ts]
            sequence.append(np.stack([builtup, vol, pop], axis=0))
        target = self.temporal_stack[TARGET_EPOCH][i:i+ts, j:j+ts]
        return (torch.FloatTensor(np.stack(sequence, axis=0)),
                torch.FloatTensor(target).unsqueeze(0))


# Build val_loader using the same spatial block holdout as all DL models
torch.manual_seed(42)
np.random.seed(42)
dataset = ThreeChannelDataset(ghsl, volume, population, tile_size=TILE_SIZE)

val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
if not os.path.exists(val_idx_path):
    print("ERROR: val_tile_indices.json not found. Run train_3channel.py first.")
    sys.exit(1)
with open(val_idx_path) as f:
    val_tiles_set = set(tuple(t) for t in json.load(f))

val_indices = [idx for idx, tile in enumerate(dataset.tiles)
               if tuple(tile) in val_tiles_set]
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

print(f"[DATA] Val tiles for SLEUTH evaluation: {len(val_indices)}", flush=True)

# =====================================================
# SLEUTH Calibration + Evaluation
# =====================================================
results = {"date": datetime.datetime.now().isoformat()}

print("\n" + "=" * 60, flush=True)
print("[1] SLEUTH Cellular Automata (calibrated)", flush=True)
print("=" * 60, flush=True)
t0 = time.time()

# ── Load real road proximity raster ──────────────────────────
road_prox_path = OUTPUT_DIR / "CONUS_road_proximity.tif"
if road_prox_path.exists():
    with rasterio.open(str(road_prox_path)) as src:
        road_proximity = src.read(1)
    print(f"  Loaded real road proximity: min={road_proximity.min():.4f}, "
          f"max={road_proximity.max():.4f}, mean={road_proximity.mean():.4f}", flush=True)
    using_real_roads = True
else:
    print("  WARNING: CONUS_road_proximity.tif not found.", flush=True)
    print("  Run scripts/preprocess_roads.py first for real road data.", flush=True)
    print("  Falling back to built-up volume proxy (less accurate).", flush=True)
    road_proximity = None
    using_real_roads = False

# ── Calibrate on TRAINING tiles only (same spatial split) ────
# Sample a representative subset of training tiles for calibration
# (not a fixed center crop — that biases to Kansas/Oklahoma geography)
print(f"\n  Calibrating SLEUTH on training-tile pixels...", flush=True)

# Build training tile indices (complement of val set)
train_indices = [idx for idx, tile in enumerate(dataset.tiles)
                 if tuple(tile) not in val_tiles_set]

# Sample up to 200 training tiles for calibration (faster than all 4877)
rng_cal = np.random.RandomState(42)
n_cal_tiles = min(200, len(train_indices))
cal_sample = rng_cal.choice(train_indices, n_cal_tiles, replace=False)

# Build calibration arrays by stacking sampled tiles
cal_bu_1975, cal_bu_1990, cal_bu_2000 = [], [], []
cal_roads, cal_pop_1975, cal_pop_1990  = [], [], []

for idx in cal_sample:
    i, j = dataset.tiles[idx]
    ts = TILE_SIZE
    cal_bu_1975.append(ghsl[1975][i:i+ts, j:j+ts])
    cal_bu_1990.append(ghsl[1990][i:i+ts, j:j+ts])
    cal_bu_2000.append(ghsl[2000][i:i+ts, j:j+ts])
    cal_pop_1975.append(population[1975][i:i+ts, j:j+ts])
    cal_pop_1990.append(population[1990][i:i+ts, j:j+ts])
    if using_real_roads:
        cal_roads.append(road_proximity[i:i+ts, j:j+ts])
    else:
        cal_roads.append(volume[1990][i:i+ts, j:j+ts])

# Stack into single arrays for calibration
cal_bu_1975  = np.concatenate([x.flatten() for x in cal_bu_1975])
cal_bu_1990  = np.concatenate([x.flatten() for x in cal_bu_1990])
cal_bu_2000  = np.concatenate([x.flatten() for x in cal_bu_2000])
cal_roads_arr   = np.concatenate([x.flatten() for x in cal_roads])
cal_pop_1975_arr = np.concatenate([x.flatten() for x in cal_pop_1975])
cal_pop_1990_arr = np.concatenate([x.flatten() for x in cal_pop_1990])

# Reshape to 2D for SLEUTH (treat all sampled pixels as one big tile)
H_cal = len(cal_sample)
W_cal = TILE_SIZE * TILE_SIZE
cal_bu_1975  = cal_bu_1975.reshape(H_cal, W_cal)
cal_bu_1990  = cal_bu_1990.reshape(H_cal, W_cal)
cal_bu_2000  = cal_bu_2000.reshape(H_cal, W_cal)
cal_roads_arr    = cal_roads_arr.reshape(H_cal, W_cal)
cal_pop_1975_arr = cal_pop_1975_arr.reshape(H_cal, W_cal)
cal_pop_1990_arr = cal_pop_1990_arr.reshape(H_cal, W_cal)

print(f"  Calibration set: {H_cal} tiles ({H_cal*W_cal:,} pixels)", flush=True)

sleuth = SLEUTH()
best_params, cal_mse = sleuth.calibrate(
    urban_t0=cal_bu_1975,
    urban_t1=cal_bu_1990,
    urban_t2=cal_bu_2000,
    roads=cal_roads_arr,
    population_t0=cal_pop_1975_arr,
    population_t1=cal_pop_1990_arr,
    n_dispersion=6, n_breed=5, n_spread=6, n_road=5,
)

# ── Evaluate on val tiles — full 8-step prediction chain ─────
# Correct: chain through all 8 epochs 1975→1980→...→2010→2015
# Iterate directly over val tile coords (like validate_2015.py)
# so we can slice road_proximity at the correct spatial location.
PRED_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]   # 8 steps → 2015

print(f"\n  Evaluating SLEUTH on {len(val_tiles_set)} val tiles "
      f"(8-step chain: 1975→1980→...→2010→2015)...", flush=True)
print(f"  Road input: {'real TIGER/Line proximity' if using_real_roads else 'volume proxy (fallback)'}", flush=True)

sleuth_preds   = []
sleuth_gts     = []
sleuth_last_bu = []

val_tiles_list = sorted(val_tiles_set)   # deterministic order

for n, (i, j) in enumerate(val_tiles_list):
    ts = TILE_SIZE

    # Skip tiles with negligible urban signal
    if ghsl[2015][i:i+ts, j:j+ts].mean() < 0.005:
        continue

    # Road proximity for this tile
    if using_real_roads:
        roads_tile = road_proximity[i:i+ts, j:j+ts]
    else:
        roads_tile = volume[1990][i:i+ts, j:j+ts]

    # 8-step prediction chain from 1975 builtup
    sleuth.rng = np.random.RandomState(42)
    current = ghsl[1975][i:i+ts, j:j+ts].copy()

    for step_idx, epoch in enumerate(PRED_EPOCHS):
        pop_t = population[epoch][i:i+ts, j:j+ts]
        current = sleuth.predict_step(current, roads_tile, pop_t)

    sleuth_preds.append(current)
    sleuth_gts.append(ghsl[2015][i:i+ts, j:j+ts])
    sleuth_last_bu.append(ghsl[2010][i:i+ts, j:j+ts])

    if (n + 1) % 200 == 0:
        print(f"    {n+1}/{len(val_tiles_list)} tiles", flush=True)

sleuth_preds   = np.array(sleuth_preds)
sleuth_gts     = np.array(sleuth_gts)
sleuth_last_bu = np.array(sleuth_last_bu)

sleuth_mse  = np.mean((sleuth_preds - sleuth_gts) ** 2)
sleuth_mae  = np.mean(np.abs(sleuth_preds - sleuth_gts))
sleuth_rmse = np.sqrt(sleuth_mse)

CHANGE_THRESHOLD = 0.01
obs_ch_s  = (sleuth_gts   - sleuth_last_bu) > CHANGE_THRESHOLD
pred_ch_s = (sleuth_preds - sleuth_last_bu) > CHANGE_THRESHOLD
s_inter   = int((obs_ch_s & pred_ch_s).sum())
s_union   = int((obs_ch_s | pred_ch_s).sum())
sleuth_fom = s_inter / s_union if s_union > 0 else 0.0

sleuth_time = time.time() - t0

print(f"  SLEUTH FINAL: MSE={sleuth_mse:.6f}, MAE={sleuth_mae:.6f}, "
      f"RMSE={sleuth_rmse:.6f}, FoM={sleuth_fom:.4f} ({sleuth_time/60:.0f}m)", flush=True)

results["sleuth_ca"] = {
    "val_mse":           round(float(sleuth_mse),  6),
    "val_mae":           round(float(sleuth_mae),  6),
    "val_rmse":          round(float(sleuth_rmse), 6),
    "val_fom":           round(float(sleuth_fom),  4),
    "calibration_mse":   round(float(cal_mse),     6),
    "calibrated_params": best_params,
    "training_time_min": round(sleuth_time / 60,   1),
    "params": 4,
    "road_input":        "TIGER/Line primary roads (real)" if using_real_roads else "built-up volume proxy",
    "prediction_chain":  "8-step: 1975→1980→1985→1990→1995→2000→2005→2010→2015",
    "calibration_tiles": n_cal_tiles,
}

# =====================================================
# Comparison with 3ch ConvLSTM
# =====================================================
convlstm_mse = None
hist_path = str(RESULTS_DIR / "training_3ch_history.json")
if os.path.exists(hist_path):
    with open(hist_path) as f:
        convlstm_mse = json.load(f).get("val_mse")

if convlstm_mse is not None:
    other = results["sleuth_ca"]["val_mse"]
    pct = ((other - convlstm_mse) / other) * 100
    results["sleuth_ca"]["vs_convlstm_3ch_pct"] = round(pct, 1)
    print(f"\n  SLEUTH vs ConvLSTM 3ch: ConvLSTM better by {pct:.1f}%", flush=True)

# Save
results_path = str(RESULTS_DIR / "sota_baselines_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}", flush=True)

print("\n" + "=" * 60, flush=True)
print("SLEUTH BASELINE COMPLETE", flush=True)
print(f"Ended: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

# Auto-consolidate into all_results.json
from scripts.consolidate_results import consolidate
consolidate()