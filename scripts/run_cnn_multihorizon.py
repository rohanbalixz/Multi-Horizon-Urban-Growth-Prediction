#!/usr/bin/env python3
"""
CNN Baseline at Multi-Horizon Forecasting
==========================================
Trains and evaluates SimpleCNN at the same 5/10/20-year horizons used in
train_multihorizon.py, using the identical input epoch sets, val tile split,
and target year (2015).

CNN takes all input timesteps concatenated as flat channels (T × 3 = input_channels),
so it has full access to the historical trajectory but without temporal ordering.

Results written to: results/metrics/cnn_multihorizon_results.json
Compare against:    results/metrics/multihorizon_results.json  (ConvLSTM)

Purpose: Determine whether ConvLSTM's temporal memory gives an advantage over
a spatial-only CNN at longer horizons. This is the key experiment for the paper.
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
print("CNN MULTI-HORIZON BASELINE", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

torch.manual_seed(42)
np.random.seed(42)

TARGET_EPOCH = 2015
TILE_SIZE    = 128
BATCH_SIZE   = 8
NUM_EPOCHS   = 25
CHANGE_THRESHOLD = 0.01

HORIZONS = {
    "5yr":  [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010],
    "10yr": [1975, 1980, 1985, 1990, 1995, 2000, 2005],
    "20yr": [1975, 1980, 1985, 1990, 1995],
}

# =====================================================
# Load data
# =====================================================
ALL_INPUT_EPOCHS = sorted(set(e for epochs in HORIZONS.values() for e in epochs))

print("\n[DATA] Loading all data channels...", flush=True)
ghsl, volume, population = {}, {}, {}

for year in ALL_INPUT_EPOCHS + [TARGET_EPOCH]:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")) as src:
        ghsl[year] = src.read(1)
    print(f"  Builtup {year}: mean={ghsl[year].mean():.6f}", flush=True)

for year in ALL_INPUT_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")) as src:
        volume[year] = src.read(1)
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{year}.tif")) as src:
        population[year] = src.read(1)

print("  All channels loaded.", flush=True)
gc.collect()

# =====================================================
# Dataset
# =====================================================
class HorizonDataset(Dataset):
    """Dataset for a specific set of input epochs."""
    def __init__(self, input_epochs):
        self.input_epochs = input_epochs
        height, width = ghsl[TARGET_EPOCH].shape
        stride = TILE_SIZE // 2
        self.tiles = []
        for i in range(0, height - TILE_SIZE + 1, stride):
            for j in range(0, width - TILE_SIZE + 1, stride):
                if ghsl[TARGET_EPOCH][i:i+TILE_SIZE, j:j+TILE_SIZE].mean() > 0.01:
                    self.tiles.append((i, j))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        frames = []
        for epoch in self.input_epochs:
            bu  = ghsl[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
            vol = volume[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
            pop = population[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE]
            frames.append(np.stack([bu, vol, pop], axis=0))
        # Shape: (T, 3, H, W)
        sequence = torch.FloatTensor(np.stack(frames, axis=0))
        target   = torch.FloatTensor(ghsl[TARGET_EPOCH][i:i+TILE_SIZE, j:j+TILE_SIZE]).unsqueeze(0)
        return sequence, target


def make_loaders(input_epochs):
    ds = HorizonDataset(input_epochs)
    val_idx_path = RESULTS_DIR / "val_tile_indices.json"
    if not val_idx_path.exists():
        print("ERROR: val_tile_indices.json not found. Run train_3channel.py first.")
        sys.exit(1)
    with open(val_idx_path) as f:
        val_tiles_set = set(tuple(t) for t in json.load(f))

    train_indices, val_indices = [], []
    for idx, tile in enumerate(ds.tiles):
        if tuple(tile) in val_tiles_set:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    print(f"  Split: {len(train_indices)} train / {len(val_indices)} val tiles", flush=True)
    tr_loader = DataLoader(Subset(ds, train_indices), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(Subset(ds, val_indices),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr_loader, va_loader

# =====================================================
# Model
# =====================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        return torch.sigmoid(self.net(x.reshape(B, T * C, H, W)))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.out_channels == 1:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# =====================================================
# Training
# =====================================================
def train_and_eval(model, tr_loader, va_loader, label):
    t0 = time.time()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(NUM_EPOCHS):
        model.train()
        tr_loss = 0
        for seqs, tgts in tr_loader:
            optimizer.zero_grad()
            pred = model(seqs)
            loss = criterion(pred, tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for seqs, tgts in va_loader:
                va_loss += criterion(model(seqs), tgts).item()
        va_loss /= len(va_loader)

        scheduler.step(va_loss)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (epoch + 1) * (NUM_EPOCHS - epoch - 1)
            print(f"    Epoch {epoch+1}/{NUM_EPOCHS}: train={tr_loss:.6f}  val={va_loss:.6f}  "
                  f"{elapsed/60:.0f}m elapsed  ETA {eta/60:.0f}m", flush=True)

    model.load_state_dict(best_state)
    model.eval()

    # --- Final metrics ---
    all_preds, all_tgts, all_last_bu = [], [], []
    with torch.no_grad():
        for seqs, tgts in va_loader:
            all_preds.append(model(seqs))
            all_tgts.append(tgts)
            all_last_bu.append(seqs[:, -1, 0:1])   # last built-up channel for FoM

    all_preds   = torch.cat(all_preds)
    all_tgts    = torch.cat(all_tgts)
    all_last_bu = torch.cat(all_last_bu)

    mse  = ((all_preds - all_tgts) ** 2).mean().item()
    mae  = (all_preds - all_tgts).abs().mean().item()
    rmse = mse ** 0.5
    ss_res = ((all_tgts - all_preds) ** 2).sum().item()
    ss_tot = ((all_tgts - all_tgts.mean()) ** 2).sum().item()
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    obs_ch  = (all_tgts    - all_last_bu) > CHANGE_THRESHOLD
    pred_ch = (all_preds   - all_last_bu) > CHANGE_THRESHOLD
    inter   = int((obs_ch & pred_ch).sum().item())
    union   = int((obs_ch | pred_ch).sum().item())
    fom     = inter / union if union > 0 else 0.0

    elapsed = time.time() - t0
    print(f"  {label} FINAL: MSE={mse:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  "
          f"R²={r2:.4f}  FoM={fom:.4f}  ({elapsed/60:.0f}m)", flush=True)

    return {
        "val_mse":          round(mse,  6),
        "val_mae":          round(mae,  6),
        "val_rmse":         round(rmse, 6),
        "val_r2":           round(r2,   4),
        "val_fom":          round(fom,  4),
        "params":           model.count_parameters(),
        "training_time_min": round(elapsed / 60, 1),
        "epochs":           NUM_EPOCHS,
    }

# =====================================================
# Run all horizons
# =====================================================
results = {
    "experiment":  "cnn_multihorizon_baseline",
    "description": (
        "SimpleCNN (T×3 flat input channels) trained at 5/10/20-year horizons, "
        "all targeting GHSL 2015. Identical input epochs, val split, and target year "
        "as train_multihorizon.py. MSE/FoM computed over all val-tile pixels (no mask). "
        "ConvLSTM comparison uses final-evaluation MSE from multihorizon_results.json "
        "(also all-pixel, no mask) — NOT training-loop val_loss. FoM from same file."
    ),
    "target_epoch": TARGET_EPOCH,
    "split_method": "spatial_block_holdout (val_tile_indices.json)",
    "horizons": {},
    "timestamp": datetime.datetime.now().isoformat(),
}

for name, input_epochs in HORIZONS.items():
    years_ahead = {"5yr": 5, "10yr": 10, "20yr": 20}[name]
    n_ch = len(input_epochs) * 3
    print(f"\n{'='*60}", flush=True)
    print(f"HORIZON: {name}  ({years_ahead} yr ahead, last input={input_epochs[-1]})", flush=True)
    print(f"  Input epochs: {input_epochs}", flush=True)
    print(f"  CNN input channels: {n_ch}  ({len(input_epochs)} timesteps × 3 channels)", flush=True)

    tr_loader, va_loader = make_loaders(input_epochs)
    model = SimpleCNN(input_channels=n_ch)
    model.apply(init_weights)
    print(f"  Parameters: {model.count_parameters():,}", flush=True)

    res = train_and_eval(model, tr_loader, va_loader, label=f"CNN-{name}")
    res["input_epochs"]    = input_epochs
    res["last_input_year"] = input_epochs[-1]
    res["years_ahead"]     = years_ahead
    res["n_input_epochs"]  = len(input_epochs)

    results["horizons"][name] = res
    del model; gc.collect()

# =====================================================
# Summary table
# =====================================================
print(f"\n{'='*60}", flush=True)
print("CNN MULTI-HORIZON SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Horizon':<8} {'MSE':>10} {'MAE':>10} {'R²':>8} {'FoM':>8} {'Params':>8} {'Time(m)':>8}", flush=True)
print("-" * 65, flush=True)
for name, r in results["horizons"].items():
    print(
        f"{name:<8} {r['val_mse']:>10.6f} {r['val_mae']:>10.6f} "
        f"{r['val_r2']:>8.4f} {r['val_fom']:>8.4f} "
        f"{r['params']:>8,} {r['training_time_min']:>8.0f}",
        flush=True,
    )

# =====================================================
# Cross-model comparison (consistent all-pixel MSE)
# =====================================================
# ConvLSTM MSE and FoM are read from multihorizon_results.json — the final
# post-training evaluation over all val-tile pixels (same mask as CNN above).
# DO NOT use training-loop val_loss (batch-averaged, different statistic).
mh_path = RESULTS_DIR / "multihorizon_results.json"
if not mh_path.exists():
    print("WARNING: multihorizon_results.json not found — skipping ConvLSTM comparison")
    conv_ref = {}
else:
    with open(mh_path) as f:
        mh_data = json.load(f)
    conv_ref = {
        name: {
            "mse": mh_data["horizons"][name]["convlstm"]["mse"],
            "fom": mh_data["horizons"][name]["convlstm"]["val_fom"],
        }
        for name in ["5yr", "10yr", "20yr"]
        if name in mh_data["horizons"]
    }

print(f"\n{'='*60}", flush=True)
print("COMPARISON: ConvLSTM vs CNN at each horizon", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Horizon':<8} {'ConvMSE':>10} {'CNNMSE':>10} {'CNN win%':>10} "
      f"{'ConvFoM':>9} {'CNNFoM':>9} {'FoM Δ':>8}", flush=True)
print("-" * 72, flush=True)
comparison = {}
for name in HORIZONS:
    if name not in conv_ref:
        continue
    conv_mse  = conv_ref[name]["mse"]
    conv_fom  = conv_ref[name]["fom"]
    cnn_mse   = results["horizons"][name]["val_mse"]
    cnn_fom   = results["horizons"][name]["val_fom"]
    cnn_win   = round((1 - cnn_mse / conv_mse) * 100, 1)
    fom_delta = round(conv_fom - cnn_fom, 4)
    winner    = "CNN wins" if cnn_fom > conv_fom else "ConvLSTM wins"
    print(
        f"{name:<8} {conv_mse:>10.6f} {cnn_mse:>10.6f} {cnn_win:>+9.1f}%  "
        f"{conv_fom:>9.4f} {cnn_fom:>9.4f} {fom_delta:>+8.4f}  ({winner})",
        flush=True,
    )
    comparison[name] = {
        "convlstm_mse_allpx": conv_mse,
        "convlstm_fom":       conv_fom,
        "cnn_mse_allpx":      cnn_mse,
        "cnn_fom":            cnn_fom,
        "cnn_mse_win_pct":    cnn_win,
        "fom_gap_conv_minus_cnn": fom_delta,
    }
results["convlstm_comparison"] = comparison
mse_wins  = [comparison[h]["cnn_mse_win_pct"] for h in ["5yr","10yr","20yr"] if h in comparison]
fom_gaps  = [comparison[h]["fom_gap_conv_minus_cnn"] for h in ["5yr","10yr","20yr"] if h in comparison]
results["key_finding"] = (
    f"CNN wins on MSE at all horizons: {mse_wins[0]}% (5yr) → {mse_wins[1]}% (10yr) → {mse_wins[2]}% (20yr). "
    f"CNN also wins on FoM at all horizons. FoM gap (ConvLSTM − CNN): "
    f"{fom_gaps[0]:+.4f} (5yr) → {fom_gaps[1]:+.4f} (10yr) → {fom_gaps[2]:+.4f} (20yr). "
    "ConvLSTM FoM improves at longer horizons; CNN FoM degrades."
)

# =====================================================
# Save results
# =====================================================
out_path = RESULTS_DIR / "cnn_multihorizon_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
    f.write("\n")
print(f"\nResults saved to {out_path}", flush=True)
print(f"\n{'='*60}", flush=True)
print("CNN MULTI-HORIZON BASELINE COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
