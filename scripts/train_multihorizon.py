#!/usr/bin/env python3
"""
Multi-Horizon Urban Growth Forecasting
=======================================
Trains separate ConvLSTM models for three forecast horizons,
all predicting the same 2015 GHSL ground truth:

  5-year  horizon: input 1975–2010 (8 epochs), last input = 2010 → target = 2015
  10-year horizon: input 1975–2005 (7 epochs), last input = 2005 → target = 2015
  20-year horizon: input 1975–1995 (5 epochs), last input = 1995 → target = 2015

Using the same 2015 ground truth for all horizons allows a clean comparison:
as the horizon lengthens, linear extrapolation degrades rapidly while the
ConvLSTM's spatiotemporal reasoning maintains accuracy.

Each horizon uses the SAME spatial block holdout split (val_tile_indices.json)
for fully consistent comparisons.

Linear baseline per horizon: pixel-wise least-squares trend through all
available input epochs, projected to 2015. This is the strongest possible
linear baseline at each horizon.

Outputs:
  - models/multihorizon_5yr.pth
  - models/multihorizon_10yr.pth
  - models/multihorizon_20yr.pth
  - results/metrics/multihorizon_results.json
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR   = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("MULTI-HORIZON ConvLSTM TRAINING (5 / 10 / 20 year)", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

from src.models.convlstm import ConvLSTM

NUM_EPOCHS    = 25
BATCH_SIZE    = 8
LEARNING_RATE = 5e-4
MC_DROPOUT    = 0.1
TILE_SIZE     = 128
TARGET_EPOCH  = 2015

# All possible input epochs — superset across all horizons
ALL_INPUT_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]

# Horizon definitions: name → input epochs used
HORIZONS = {
    "5yr":  {"epochs": [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010], "years_ahead": 5},
    "10yr": {"epochs": [1975, 1980, 1985, 1990, 1995, 2000, 2005],       "years_ahead": 10},
    "20yr": {"epochs": [1975, 1980, 1985, 1990, 1995],                   "years_ahead": 20},
}

# =====================================================
# Load all data
# =====================================================
print("\n[DATA] Loading all data channels...", flush=True)

ghsl = {}
for year in ALL_INPUT_EPOCHS + [TARGET_EPOCH]:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")) as src:
        ghsl[year] = src.read(1)
    print(f"  Builtup {year}: shape={ghsl[year].shape}, mean={ghsl[year].mean():.6f}", flush=True)

ref_shape = ghsl[TARGET_EPOCH].shape

volume = {}
for year in ALL_INPUT_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")) as src:
        volume[year] = src.read(1)

population = {}
for year in ALL_INPUT_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{year}.tif")) as src:
        population[year] = src.read(1)

print("  All channels loaded.", flush=True)
gc.collect()

# =====================================================
# Dataset: 3-channel, horizon-specific input epochs
# =====================================================
class HorizonDataset(Dataset):
    """3-channel ConvLSTM dataset for a specific set of input epochs."""
    def __init__(self, input_epochs, tile_size=128):
        self.input_epochs = input_epochs
        self.tile_size    = tile_size

        height, width = ghsl[TARGET_EPOCH].shape
        self.tiles = []
        stride = tile_size // 2
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                if ghsl[TARGET_EPOCH][i:i+tile_size, j:j+tile_size].mean() > 0.01:
                    self.tiles.append((i, j))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts   = self.tile_size
        sequence = []
        for epoch in self.input_epochs:
            bu  = ghsl[epoch][i:i+ts, j:j+ts]
            vol = volume[epoch][i:i+ts, j:j+ts]
            pop = population[epoch][i:i+ts, j:j+ts]
            sequence.append(np.stack([bu, vol, pop], axis=0))
        target = ghsl[TARGET_EPOCH][i:i+ts, j:j+ts][np.newaxis, ...]
        return (torch.FloatTensor(np.stack(sequence, axis=0)),
                torch.FloatTensor(target))


def make_loaders(input_epochs):
    """Build train/val loaders using the shared spatial block holdout."""
    val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
    if not os.path.exists(val_idx_path):
        print("ERROR: val_tile_indices.json not found. Run train_3channel.py first.")
        sys.exit(1)
    with open(val_idx_path) as f:
        val_tiles_set = set(tuple(t) for t in json.load(f))

    ds = HorizonDataset(input_epochs)
    train_idx, val_idx = [], []
    for idx, tile in enumerate(ds.tiles):
        if tuple(tile) in val_tiles_set:
            val_idx.append(idx)
        else:
            train_idx.append(idx)

    tr = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va = DataLoader(Subset(ds, val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Split: {len(train_idx)} train tiles, {len(val_idx)} val tiles", flush=True)
    return tr, va, ds


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


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.out_channels == 1:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =====================================================
# Linear baseline per horizon
# Pixel-wise least-squares trend through all input epochs
# projected to TARGET_EPOCH. Strongest possible linear baseline.
# =====================================================
def linear_baseline_mse(input_epochs, val_tiles, years_ahead):
    """Compute MSE and FoM of pixel-wise linear regression baseline on val tiles."""
    gt          = ghsl[TARGET_EPOCH]
    last_bu_map = ghsl[input_epochs[-1]]  # last observed builtup before target
    years       = np.array(input_epochs, dtype=np.float64)
    t_target    = float(TARGET_EPOCH)

    all_gt, all_pred, all_last_bu = [], [], []
    for (i, j) in val_tiles:
        ts = TILE_SIZE
        tile_gt = gt[i:i+ts, j:j+ts]
        if tile_gt.mean() < 0.01:
            continue

        # Stack pixel time series: shape (T, H*W)
        stack = np.stack([ghsl[y][i:i+ts, j:j+ts].flatten() for y in input_epochs])  # (T, HW)
        T, HW = stack.shape

        # Fit pixel-wise linear regression: y = a + b*t
        t_mean   = years.mean()
        t_denom  = np.sum((years - t_mean) ** 2)
        b        = np.sum((stack - stack.mean(axis=0)) * (years - t_mean)[:, None], axis=0) / t_denom
        a        = stack.mean(axis=0) - b * t_mean
        pred_px  = (a + b * t_target).clip(0, 1).reshape(ts, ts)

        all_gt.append(tile_gt)
        all_pred.append(pred_px)
        all_last_bu.append(last_bu_map[i:i+ts, j:j+ts])

    all_gt      = np.concatenate([g.flatten() for g in all_gt])
    all_pred    = np.concatenate([p.flatten() for p in all_pred])
    all_last_bu = np.concatenate([b.flatten() for b in all_last_bu])

    # All-pixel MSE/MAE — consistent with ConvLSTM and CNN eval (no urban mask).
    mse  = float(np.mean((all_gt - all_pred) ** 2))
    mae  = float(np.mean(np.abs(all_gt - all_pred)))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((all_gt - all_pred) ** 2)
    ss_tot = np.sum((all_gt - all_gt.mean()) ** 2)
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    CHANGE_THRESHOLD = 0.01
    obs_ch  = (all_gt      - all_last_bu) > CHANGE_THRESHOLD
    pred_ch = (all_pred    - all_last_bu) > CHANGE_THRESHOLD
    inter   = int((obs_ch & pred_ch).sum())
    union_v = int((obs_ch | pred_ch).sum())
    fom     = inter / union_v if union_v > 0 else 0.0

    return {"mse": round(mse, 6), "mae": round(mae, 6), "rmse": round(rmse, 6),
            "r2": round(r2, 4), "fom": round(fom, 4)}


# =====================================================
# Train and evaluate one horizon
# =====================================================
def train_horizon(horizon_name, input_epochs, years_ahead):
    print(f"\n{'='*60}", flush=True)
    print(f"HORIZON: {horizon_name} ({years_ahead} years ahead)", flush=True)
    print(f"  Input epochs: {input_epochs}", flush=True)
    print(f"  Last input year: {input_epochs[-1]}  →  Target: {TARGET_EPOCH}", flush=True)
    print(f"{'='*60}", flush=True)

    tr_loader, va_loader, ds = make_loaders(input_epochs)

    model = ConvLSTM(input_channels=3, hidden_channels=64,
                     num_layers=2, mc_dropout=MC_DROPOUT)
    model.apply(init_weights)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val_loss = float('inf')
    best_state    = None
    t_start       = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for sequences, targets in tr_loader:
            optimizer.zero_grad()
            pred = predict_future_step(model, sequences)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(tr_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in va_loader:
                pred = predict_future_step(model, sequences)
                val_loss += criterion(pred, targets).item()
        val_loss /= len(va_loader)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t_start
        eta     = elapsed / epoch * (NUM_EPOCHS - epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | train={train_loss:.6f} | "
                  f"val={val_loss:.6f} | {elapsed/60:.0f}m | ETA {eta/60:.0f}m", flush=True)

    model.load_state_dict(best_state)
    save_path = str(MODELS_DIR / f"multihorizon_{horizon_name}.pth")
    torch.save(best_state, save_path)
    print(f"\n  Model saved: {save_path} (best_val_loss={best_val_loss:.6f})", flush=True)

    # Final eval on val set
    model.eval()
    CHANGE_THRESHOLD = 0.01
    all_preds, all_targets, all_last_bu = [], [], []
    with torch.no_grad():
        for sequences, targets in va_loader:
            pred = predict_future_step(model, sequences)
            all_preds.append(pred)
            all_targets.append(targets)
            all_last_bu.append(sequences[:, -1, 0:1])
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_last_bu = torch.cat(all_last_bu)

    val_mse  = float(((all_preds - all_targets) ** 2).mean().item())
    val_mae  = float((all_preds - all_targets).abs().mean().item())
    val_rmse = float(val_mse ** 0.5)

    # FoM
    obs_change  = (all_targets - all_last_bu) > CHANGE_THRESHOLD
    pred_change = (all_preds   - all_last_bu) > CHANGE_THRESHOLD
    intersection = int((obs_change & pred_change).sum().item())
    union        = int((obs_change | pred_change).sum().item())
    fom = intersection / union if union > 0 else 0.0

    # R² — all-pixel, consistent with CNN eval and linear baseline
    p_flat  = all_preds.flatten().numpy()
    g_flat  = all_targets.flatten().numpy()
    ss_res  = float(np.sum((g_flat - p_flat) ** 2))
    ss_tot  = float(np.sum((g_flat - g_flat.mean()) ** 2))
    r2      = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    total_time = time.time() - t_start
    print(f"\n  ConvLSTM {horizon_name}: MSE={val_mse:.6f}, MAE={val_mae:.6f}, "
          f"RMSE={val_rmse:.6f}, R²={r2:.4f}, FoM={fom:.4f}", flush=True)

    # Linear baseline
    val_tiles_list = [tile for tile in ds.tiles
                      if tuple(tile) in set(tuple(t) for t in
                      json.load(open(str(RESULTS_DIR / "val_tile_indices.json"))))]
    lin = linear_baseline_mse(input_epochs, val_tiles_list, years_ahead)
    print(f"  Linear baseline {horizon_name}: MSE={lin['mse']:.6f}, R²={lin['r2']:.4f}, FoM={lin.get('fom', 0):.4f}", flush=True)

    imp_vs_linear = (1 - val_mse / lin["mse"]) * 100 if lin["mse"] > 0 else 0.0
    print(f"  ConvLSTM vs Linear: {imp_vs_linear:+.1f}%", flush=True)

    return {
        "horizon": horizon_name,
        "years_ahead": years_ahead,
        "input_epochs": input_epochs,
        "last_input_year": input_epochs[-1],
        "target_year": TARGET_EPOCH,
        "n_input_epochs": len(input_epochs),
        "convlstm": {
            "val_mse":  round(val_mse,  6),
            "val_mae":  round(val_mae,  6),
            "val_rmse": round(val_rmse, 6),
            "val_fom":  round(fom,      4),
            "r2":       round(r2,       4),
            "best_val_loss": round(best_val_loss, 6),
            "params": sum(p.numel() for p in model.parameters()),
            "training_time_hours": round(total_time / 3600, 2),
        },
        "linear_regression_baseline": lin,
        "improvement_vs_linear_pct": round(imp_vs_linear, 2),
        "model_path": save_path,
    }


# =====================================================
# Run all horizons
# =====================================================
all_results = {}

for name, cfg in HORIZONS.items():
    result = train_horizon(name, cfg["epochs"], cfg["years_ahead"])
    all_results[name] = result
    gc.collect()

# =====================================================
# Summary
# =====================================================
print(f"\n{'='*60}", flush=True)
print("MULTI-HORIZON SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Horizon':<10} {'ConvLSTM MSE':>14} {'Linear MSE':>12} {'Improvement':>12} {'R²':>8}", flush=True)
print("-" * 60, flush=True)
for name, r in all_results.items():
    print(f"{name:<10} {r['convlstm']['val_mse']:>14.6f} "
          f"{r['linear_regression_baseline']['mse']:>12.6f} "
          f"{r['improvement_vs_linear_pct']:>+11.1f}% "
          f"{r['convlstm']['r2']:>8.4f}", flush=True)

# =====================================================
# Save
# =====================================================
output = {
    "experiment": "multihorizon_forecasting",
    "description": (
        "ConvLSTM trained separately at 5/10/20-year horizons, all targeting GHSL 2015. "
        "Linear baseline: pixel-wise least-squares regression projected to 2015."
    ),
    "target_epoch": TARGET_EPOCH,
    "split_method": "spatial_block_holdout (val_tile_indices.json)",
    "architecture": "ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1)",
    "horizons": all_results,
    "timestamp": datetime.datetime.now().isoformat(),
}

out_path = str(RESULTS_DIR / "multihorizon_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)

# Auto-consolidate
from scripts.consolidate_results import consolidate
consolidate()

print(f"\n{'='*60}", flush=True)
print("MULTI-HORIZON TRAINING COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
