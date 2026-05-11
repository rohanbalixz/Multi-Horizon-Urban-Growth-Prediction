#!/usr/bin/env python3
"""
3-Channel Ablation & Baseline Experiments
==========================================
Re-runs all ablations and baselines with the 3-channel input:
  CH0: Built-up density, CH1: Built-up volume, CH2: Population density

Experiments:
  1. Ablation -- built-up only (1 channel)
  2. Ablation -- volume only (1 channel)
  3. Ablation -- population only (1 channel)
  4. Ablation -- built-up + volume (2ch, no population)
  5. Ablation -- 1-layer ConvLSTM (3ch)
  6. CNN baseline (3ch)
  7. U-Net baseline (3ch)
  8. Linear extrapolation (unchanged)

Saves to: results/metrics/ablation_3ch_results.json
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("3-CHANNEL ABLATION & BASELINE EXPERIMENTS", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

print("  Imports complete.", flush=True)

from src.models.convlstm import ConvLSTM

# Global seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

NUM_EPOCHS = 25
BATCH_SIZE = 8
MC_DROPOUT = 0.1

TRAIN_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH = 2015

# =====================================================
# Load Data from pre-processed GeoTIFFs
# =====================================================

print("\n[DATA] Loading all data channels...", flush=True)

ALL_EPOCHS = sorted(set(TRAIN_EPOCHS + [TARGET_EPOCH]))

ghsl = {}
for year in ALL_EPOCHS:
    print(f"  Loading built-up {year}...", flush=True)
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")) as src:
        ghsl[year] = src.read(1)

ref_shape = ghsl[TARGET_EPOCH].shape

# Built-up volume (training epochs only — 2015 has no volume data used in training)
print("  Loading built-up volume...", flush=True)
volume = {}
for year in TRAIN_EPOCHS:
    vpath = str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")
    with rasterio.open(vpath) as src:
        volume[year] = src.read(1)
    print(f"  Volume {year}: shape={volume[year].shape}, mean={volume[year].mean():.4f}", flush=True)

# Population density (training epochs only)
print("  Loading population density...", flush=True)
population = {}
for year in TRAIN_EPOCHS:
    fpath = str(OUTPUT_DIR / f"CONUS_population_{year}.tif")
    for attempt in range(1, 4):
        try:
            with rasterio.open(fpath) as src:
                population[year] = src.read(1)
                print(f"    Population {year}: shape={population[year].shape}, mean={population[year].mean():.4f}", flush=True)
            break
        except rasterio.errors.RasterioIOError as e:
            if attempt < 3:
                print(f"    Transient GDAL error for {year} (attempt {attempt}/3), retrying...", flush=True)
                gc.collect()
                time.sleep(2)
            else:
                print(f"    ERROR loading {fpath}: {e}", flush=True)
gc.collect()
print("  All data loaded.", flush=True)


# =====================================================
# Dataset classes
# =====================================================
class FlexibleDataset(Dataset):
    """Dataset supporting arbitrary channel combinations."""
    def __init__(self, ghsl, volume, population, tile_size=128,
                 channels='all'):
        """
        channels: 'all' (3ch), 'builtup' (1ch), 'volume' (1ch),
                  'population' (1ch), 'builtup_volume' (2ch)
        """
        self.ghsl = ghsl
        self.volume = volume
        self.population = population
        self.tile_size = tile_size
        self.channels = channels
        self.epochs = sorted(ghsl.keys())

        height, width = list(ghsl.values())[0].shape
        self.tiles = []
        stride = tile_size // 2
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                tile = ghsl[self.epochs[-1]][i:i+tile_size, j:j+tile_size]
                if tile.mean() > 0.01:
                    self.tiles.append((i, j))

    def _get_num_channels(self):
        if self.channels in ('builtup', 'volume', 'population'):
            return 1
        elif self.channels == 'builtup_volume':
            return 2
        else:  # 'all'
            return 3

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts = self.tile_size
        sequence = []
        # Use all epochs except target as input sequence
        input_epochs = [e for e in self.epochs if e != TARGET_EPOCH]
        for epoch in input_epochs:
            bu = self.ghsl[epoch][i:i+ts, j:j+ts]
            vol = self.volume[epoch][i:i+ts, j:j+ts]
            pop = self.population[epoch][i:i+ts, j:j+ts]

            if self.channels == 'builtup':
                frame = bu[np.newaxis, ...]
            elif self.channels == 'volume':
                frame = vol[np.newaxis, ...]
            elif self.channels == 'population':
                frame = pop[np.newaxis, ...]
            elif self.channels == 'builtup_volume':
                frame = np.stack([bu, vol], axis=0)
            else:  # 'all'
                frame = np.stack([bu, vol, pop], axis=0)
            sequence.append(frame)

        target = self.ghsl[TARGET_EPOCH][i:i+ts, j:j+ts]

        return (torch.FloatTensor(np.stack(sequence, axis=0)),
                torch.FloatTensor(target).unsqueeze(0))


# =====================================================
# Training infrastructure
# =====================================================
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.out_channels == 1:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def predict_future_generic(model, sequences):
    """Autoregressive future-step prediction."""
    if not hasattr(model, 'convlstm_layers'):
        out, _ = model(sequences)
        return out[:, 0]

    _, hidden_states = model(sequences)
    last_input = sequences[:, -1]
    x_t = last_input
    layer_hiddens = []
    for layer_idx, layer in enumerate(model.convlstm_layers):
        h, c = hidden_states[layer_idx]
        h, c = layer(x_t, (h, c))
        if hasattr(model, 'mc_dropouts'):
            h = model.mc_dropouts[layer_idx](h)
        hidden_states[layer_idx] = (h, c)
        x_t = h
        layer_hiddens.append(h)
    return model.decode(layer_hiddens)


def train_and_eval(model, train_loader, val_loader, num_epochs=NUM_EPOCHS,
                   lr=5e-4, label="model", compute_fom=True):
    print(f"  Training {label} ({num_epochs} epochs)...", flush=True)
    t0 = time.time()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            pred = predict_future_generic(model, sequences)
            loss = criterion(pred, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                pred = predict_future_generic(model, sequences)
                val_loss += criterion(pred, targets).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            eta = elapsed / (epoch + 1) * (num_epochs - epoch - 1)
            print(f"    Epoch {epoch+1}/{num_epochs}: train={train_loss:.6f}, "
                  f"val={val_loss:.6f}, {elapsed/60:.0f}m elapsed, ETA {eta/60:.0f}m", flush=True)

    model.load_state_dict(best_state)

    # Final metrics
    model.eval()
    CHANGE_THRESHOLD = 0.01
    all_preds, all_targets, all_last_bu = [], [], []
    with torch.no_grad():
        for sequences, targets in val_loader:
            pred = predict_future_generic(model, sequences)
            all_preds.append(pred)
            all_targets.append(targets)
            if compute_fom:
                all_last_bu.append(sequences[:, -1, 0:1])
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = ((all_preds - all_targets) ** 2).mean().item()
    mae = (all_preds - all_targets).abs().mean().item()
    rmse = mse ** 0.5
    total_time = time.time() - t0

    fom = None
    if compute_fom and all_last_bu:
        all_last_bu = torch.cat(all_last_bu)
        obs_change  = (all_targets - all_last_bu) > CHANGE_THRESHOLD
        pred_change = (all_preds   - all_last_bu) > CHANGE_THRESHOLD
        intersection = int((obs_change & pred_change).sum().item())
        union        = int((obs_change | pred_change).sum().item())
        fom = intersection / union if union > 0 else 0.0

    fom_str = f", FoM={fom:.4f}" if fom is not None else ""
    print(f"  {label} FINAL: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}{fom_str} ({total_time/60:.0f}m)", flush=True)
    result = {
        "val_mse": round(mse, 6), "val_mae": round(mae, 6), "val_rmse": round(rmse, 6),
        "training_time_min": round(total_time / 60, 1), "epochs": num_epochs
    }
    if fom is not None:
        result["val_fom"] = round(fom, 4)
    return result


def make_loaders(channels_cfg):
    """Create train/val loaders using the same spatial block holdout as train_3channel.py."""
    torch.manual_seed(42)
    ds = FlexibleDataset(ghsl, volume, population, channels=channels_cfg)

    # Load val tile indices saved by train_3channel.py
    val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
    if not os.path.exists(val_idx_path):
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

    tr_loader = DataLoader(Subset(ds, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(Subset(ds, val_indices),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr_loader, va_loader, ds._get_num_channels()


def make_loaders_seqlen(n_steps):
    """Create loaders for sequence length ablation using the last N training epochs.
    Uses same spatial block holdout split as the main model.
    n_steps: number of input timesteps (uses TRAIN_EPOCHS[-n_steps:]).
    """
    epochs_subset = TRAIN_EPOCHS[-n_steps:]  # last N epochs → most recent context
    ghsl_sub = {e: ghsl[e] for e in epochs_subset + [TARGET_EPOCH]}
    vol_sub  = {e: volume[e] for e in epochs_subset}
    pop_sub  = {e: population[e] for e in epochs_subset}

    torch.manual_seed(42)
    ds = FlexibleDataset(ghsl_sub, vol_sub, pop_sub, channels='all')

    val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
    with open(val_idx_path) as f:
        val_tiles_set = set(tuple(t) for t in json.load(f))

    train_indices, val_indices = [], []
    for idx, tile in enumerate(ds.tiles):
        if tuple(tile) in val_tiles_set:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    tr_loader = DataLoader(Subset(ds, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(Subset(ds, val_indices),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr_loader, va_loader


# =====================================================
# Simple baseline architectures
# =====================================================
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )
    def forward(self, x, hidden_states=None):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T * C, H, W)
        out = torch.sigmoid(self.net(x_flat))
        return out.unsqueeze(1), None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleUNet(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x, hidden_states=None):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T * C, H, W)
        e1 = self.enc1(x_flat)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1)).unsqueeze(1), None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =====================================================
# Run all experiments
# =====================================================
results_path = str(RESULTS_DIR / "ablation_3ch_results.json")
results = {"date": datetime.datetime.now().isoformat()}

# --- Experiment 1: Built-up only (1ch) ---
print("\n" + "=" * 60, flush=True)
print("[1] Ablation: Built-up only (1 channel)", flush=True)
print("=" * 60, flush=True)
tr_l, va_l, nch = make_loaders('builtup')
m = ConvLSTM(input_channels=1, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
m.apply(init_weights)
res = train_and_eval(m, tr_l, va_l, label="builtup-only")
res["params"] = m.count_parameters()
results["ablation_builtup_only"] = res
del m; gc.collect()

# --- Experiment 2: Volume only (1ch) ---
print("\n" + "=" * 60, flush=True)
print("[2] Ablation: Volume only (1 channel)", flush=True)
print("=" * 60, flush=True)
tr_l, va_l, nch = make_loaders('volume')
m = ConvLSTM(input_channels=1, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
m.apply(init_weights)
res = train_and_eval(m, tr_l, va_l, label="volume-only", compute_fom=False)
res["params"] = m.count_parameters()
results["ablation_volume_only"] = res
del m; gc.collect()

# --- Experiment 3: Population only (1ch) ---
print("\n" + "=" * 60, flush=True)
print("[3] Ablation: Population only (1 channel)", flush=True)
print("=" * 60, flush=True)
tr_l, va_l, nch = make_loaders('population')
m = ConvLSTM(input_channels=1, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
m.apply(init_weights)
res = train_and_eval(m, tr_l, va_l, label="population-only", compute_fom=False)
res["params"] = m.count_parameters()
results["ablation_population_only"] = res
del m; gc.collect()

# --- Experiment 4: Built-up + Volume (2ch, no population) ---
print("\n" + "=" * 60, flush=True)
print("[4] Ablation: Built-up + Volume (2ch, no population)", flush=True)
print("=" * 60, flush=True)
tr_l, va_l, nch = make_loaders('builtup_volume')
m = ConvLSTM(input_channels=2, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
m.apply(init_weights)
res = train_and_eval(m, tr_l, va_l, label="builtup+volume (2ch)")
res["params"] = m.count_parameters()
results["ablation_builtup_volume_2ch"] = res
del m; gc.collect()

# --- Experiment 5: 1-layer ConvLSTM (3ch) ---
print("\n" + "=" * 60, flush=True)
print("[5] Ablation: 1-layer ConvLSTM (3 channels)", flush=True)
print("=" * 60, flush=True)
tr_l, va_l, nch = make_loaders('all')
m = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=1, mc_dropout=MC_DROPOUT)
m.apply(init_weights)
res = train_and_eval(m, tr_l, va_l, label="1-layer 3ch")
res["params"] = m.count_parameters()
results["ablation_1layer_3ch"] = res
del m; gc.collect()

# --- Experiment 6: CNN baseline (3ch) ---
print("\n" + "=" * 60, flush=True)
print("[6] CNN Baseline (3 channels)", flush=True)
print("=" * 60, flush=True)
tr_l, va_l, _ = make_loaders('all')
n_input_timesteps = len(TRAIN_EPOCHS)
m = SimpleCNN(input_channels=n_input_timesteps * 3)  # timesteps * 3 channels
m.apply(init_weights)
res = train_and_eval(m, tr_l, va_l, label="CNN-3ch")
res["params"] = m.count_parameters()
results["baseline_cnn_3ch"] = res
del m; gc.collect()

# --- Experiment 7: U-Net baseline (3ch) ---
print("\n" + "=" * 60, flush=True)
print("[7] U-Net Baseline (3 channels)", flush=True)
print("=" * 60, flush=True)
tr_l, va_l, _ = make_loaders('all')
m = SimpleUNet(input_channels=n_input_timesteps * 3)  # timesteps * 3 channels
m.apply(init_weights)
res = train_and_eval(m, tr_l, va_l, label="UNet-3ch")
res["params"] = m.count_parameters()
results["baseline_unet_3ch"] = res
del m; gc.collect()

# --- Experiment 8: Linear extrapolation (unchanged) ---
print("\n" + "=" * 60, flush=True)
print("[8] Linear Extrapolation Baseline", flush=True)
print("=" * 60, flush=True)
_, va_l, _ = make_loaders('all')
lin_preds, lin_targets, lin_last_bu = [], [], []
for sequences, targets in va_l:
    # Linear extrapolation from last two input timesteps → 2015
    x_prev = sequences[:, -2, 0:1, :, :]  # second-to-last timestep builtup
    x_last = sequences[:, -1, 0:1, :, :]  # last timestep builtup
    pred = (x_last + (x_last - x_prev)).clamp(0, 1)  # same 5yr interval
    lin_preds.append(pred)
    lin_targets.append(targets)
    lin_last_bu.append(x_last)
lin_preds = torch.cat(lin_preds)
lin_targets = torch.cat(lin_targets)
lin_last_bu = torch.cat(lin_last_bu)
lin_mse = ((lin_preds - lin_targets) ** 2).mean().item()
lin_mae = (lin_preds - lin_targets).abs().mean().item()
CHANGE_THRESHOLD = 0.01
obs_ch_lin  = (lin_targets - lin_last_bu) > CHANGE_THRESHOLD
pred_ch_lin = (lin_preds   - lin_last_bu) > CHANGE_THRESHOLD
lin_inter = int((obs_ch_lin & pred_ch_lin).sum().item())
lin_union = int((obs_ch_lin | pred_ch_lin).sum().item())
lin_fom = lin_inter / lin_union if lin_union > 0 else 0.0
results["linear_extrapolation"] = {
    "val_mse": round(lin_mse, 6),
    "val_mae": round(lin_mae, 6),
    "val_rmse": round(lin_mse ** 0.5, 6),
    "val_fom": round(lin_fom, 4)
}
print(f"  Linear: MSE={lin_mse:.6f}, MAE={lin_mae:.6f}, FoM={lin_fom:.4f}", flush=True)

# =====================================================
# Sequence length ablation: 4 / 6 / 8 timesteps (3ch)
# Uses the last N epochs of TRAIN_EPOCHS as input.
# Shows whether longer historical context improves prediction.
# 8-step result comes from training_3ch_history.json (main model).
# =====================================================
print("\n" + "=" * 60, flush=True)
print("[9] Sequence Length Ablation (3ch, last N timesteps)", flush=True)
print("=" * 60, flush=True)

for n_steps in [4, 6]:
    epochs_used = TRAIN_EPOCHS[-n_steps:]
    label = f"seqlen_{n_steps}_3ch"
    print(f"\n  Sequence length {n_steps}: epochs {epochs_used}", flush=True)
    tr_l, va_l = make_loaders_seqlen(n_steps)
    m = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
    m.apply(init_weights)
    res = train_and_eval(m, tr_l, va_l, label=f"SeqLen-{n_steps}")
    res["params"] = m.count_parameters()
    res["n_timesteps"] = n_steps
    res["epochs_used"] = epochs_used
    results[label] = res
    del m; gc.collect()

# 8-step result = main model (already trained)
hist_path = str(RESULTS_DIR / "training_3ch_history.json")
if os.path.exists(hist_path):
    with open(hist_path) as f:
        hist = json.load(f)
    results["seqlen_8_3ch"] = {
        "val_mse": hist["val_mse"], "val_mae": hist["val_mae"],
        "val_rmse": hist["val_rmse"], "params": hist["params"],
        "n_timesteps": 8, "epochs_used": TRAIN_EPOCHS,
        "note": "Main trained model (train_3channel.py)"
    }
print(f"\n  Sequence length 8: loaded from training_3ch_history.json", flush=True)

# =====================================================
# Compute comparisons
# =====================================================
print("\n" + "=" * 60, flush=True)
print("[10] Computing comparisons", flush=True)
print("=" * 60, flush=True)

# Load 3ch ConvLSTM result if available
hist_path = str(RESULTS_DIR / "training_3ch_history.json")
if os.path.exists(hist_path):
    with open(hist_path) as f:
        hist = json.load(f)
    convlstm_3ch_mse = hist["val_mse"]
    results["convlstm_3ch"] = {
        "val_mse": hist["val_mse"],
        "val_mae": hist["val_mae"],
        "val_rmse": hist["val_rmse"],
        "params": hist["params"]
    }
else:
    convlstm_3ch_mse = None
    print("  WARNING: 3ch ConvLSTM results not found. Run train_3channel.py first.", flush=True)

if convlstm_3ch_mse:
    for key in results:
        if key in ("date", "convlstm_3ch"):
            continue
        if isinstance(results[key], dict) and "val_mse" in results[key]:
            other_mse = results[key]["val_mse"]
            if other_mse > 0:
                pct = ((other_mse - convlstm_3ch_mse) / other_mse) * 100
                results[key]["vs_convlstm_3ch_error_reduction_pct"] = round(pct, 1)
                print(f"  {key}: MSE={other_mse:.6f}, ConvLSTM-3ch error reduction: {pct:.1f}%", flush=True)

# Save
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_path}", flush=True)
print("=" * 60, flush=True)
print("ALL ABLATION EXPERIMENTS COMPLETE", flush=True)
print(f"Ended: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

# Auto-consolidate into all_results.json
from scripts.consolidate_results import consolidate
consolidate()
