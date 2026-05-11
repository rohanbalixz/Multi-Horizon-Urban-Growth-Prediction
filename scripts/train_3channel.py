#!/usr/bin/env python3
"""
3-Channel ConvLSTM Training with MC Dropout + Skip-Connection Decoder
=====================================================================
Trains the ConvLSTM with 3 input channels:
    CH0: GHSL built-up surface density
    CH1: GHS-BUILT-V built-up volume
    CH2: GHS-POP population density (log-normalized)

8-timestep sequence: {1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010} (all channels)
Target: GHSL builtup 2015 (1 channel, never seen during training).

Spatial block holdout (train/val split):
    CONUS raster is divided into 1280×1280px geographic blocks.
    Every 5th block (block_id % 5 == 0) is held out for validation (~20%).
    This ensures val tiles are from entirely different geographic regions than
    train tiles, eliminating spatial autocorrelation leakage from tile overlap.
    Val tile indices are saved to results/metrics/val_tile_indices.json for
    use in post-training evaluation.

Skip-connection decoder fuses hidden states from all ConvLSTM layers.
MC Dropout (p=0.1) is applied after each ConvLSTM layer.

Input shape: [batch, time=8, channels=3, 128, 128]

25 epochs, ReduceLROnPlateau, best-model checkpointing.

Outputs:
    - models/best_3ch_mc_model.pth
    - results/metrics/training_3ch_history.json
    - results/metrics/val_tile_indices.json
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

# Define epochs before any use
TRAIN_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH = 2015

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("3-CHANNEL ConvLSTM + MC DROPOUT TRAINING", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

print("  Imports complete.", flush=True)

from src.models.convlstm import ConvLSTM

NUM_EPOCHS = 25
BATCH_SIZE = 8
LEARNING_RATE = 5e-4
MC_DROPOUT = 0.1

# =====================================================
# PHASE 0: Load Data from pre-processed GeoTIFFs
# =====================================================

# Load target builtup for tile selection; train channels use TRAIN_EPOCHS only
builtup_files = {year: str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif") for year in TRAIN_EPOCHS + [TARGET_EPOCH]}
population_files = {year: str(OUTPUT_DIR / f"CONUS_population_{year}.tif") for year in TRAIN_EPOCHS}
volume_files = {year: str(OUTPUT_DIR / f"CONUS_volume_{year}.tif") for year in TRAIN_EPOCHS}

eval_files = {
    'builtup': str(OUTPUT_DIR / f"CONUS_builtup_{TARGET_EPOCH}.tif"),
    'population': str(OUTPUT_DIR / f"CONUS_population_{TARGET_EPOCH}.tif"),
    'volume': str(OUTPUT_DIR / f"CONUS_volume_{TARGET_EPOCH}.tif")
}

def load_geotiff(fpath, max_retries=3):
    """Load a GeoTIFF with retries for transient GDAL errors."""
    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.open(fpath) as src:
                return src.read(1)
        except rasterio.errors.RasterioIOError:
            if attempt < max_retries:
                print(f"    Transient GDAL error (attempt {attempt}/{max_retries}), retrying...", flush=True)
                gc.collect()
                time.sleep(2)
            else:
                raise

# Load GHSL built-up (already normalized [0,1])
print("\n[DATA] Loading pre-processed GHSL built-up...", flush=True)
ghsl = {}
for epoch, fpath in builtup_files.items():
    print(f"  Loading {epoch}...", flush=True)
    ghsl[epoch] = load_geotiff(fpath)
    print(f"  {epoch}: shape={ghsl[epoch].shape}, mean={ghsl[epoch].mean():.4f}", flush=True)

ref_shape = ghsl[TARGET_EPOCH].shape

# Load built-up volume (TRAIN_EPOCHS only)
print("\n[DATA] Loading GHS-BUILT-V built-up volume...", flush=True)
volume = {}
for year in TRAIN_EPOCHS:
    fpath = volume_files[year]
    if not os.path.exists(fpath):
        print(f"  ERROR: {fpath} not found! Run scripts/preprocess_all_data.py first.", flush=True)
        sys.exit(1)
    volume[year] = load_geotiff(fpath)
    print(f"  {year}: shape={volume[year].shape}, mean={volume[year].mean():.4f}", flush=True)

# Load population density (TRAIN_EPOCHS only)
print("\n[DATA] Loading GHS-POP population density...", flush=True)
population = {}
for year in TRAIN_EPOCHS:
    fpath = population_files[year]
    if not os.path.exists(fpath):
        print(f"  ERROR: {fpath} not found! Run scripts/preprocess_all_data.py first.", flush=True)
        sys.exit(1)
    population[year] = load_geotiff(fpath)
    print(f"  {year}: shape={population[year].shape}, mean={population[year].mean():.4f}", flush=True)

gc.collect()


# =====================================================
# 3-Channel Dataset
# =====================================================
class ThreeChannelDataset(Dataset):
    """
    Dataset with 3 input channels: built-up, volume, population.

    Input shape per frame: (3, tile_size, tile_size)
    Sequence: 8 timesteps -> predict next epoch built-up.

    Channel mapping per timestep:
      builtup, volume, population for each epoch in TRAIN_EPOCHS.
    """
    def __init__(self, temporal_stack, volume, population,
                 tile_size=128, train_epochs=None, target_epoch=None):
        self.temporal_stack = temporal_stack
        self.volume = volume
        self.population = population
        self.tile_size = tile_size
        self.train_epochs = train_epochs or TRAIN_EPOCHS
        self.target_epoch = target_epoch or TARGET_EPOCH

        height, width = list(temporal_stack.values())[0].shape
        self.tiles = []
        stride = tile_size // 2
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                tile_data = temporal_stack[self.target_epoch][i:i+tile_size, j:j+tile_size]
                if tile_data.mean() > 0.01:
                    self.tiles.append((i, j))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts = self.tile_size
        sequence = []
        for epoch in self.train_epochs:
            builtup = self.temporal_stack[epoch][i:i+ts, j:j+ts]
            vol = self.volume[epoch][i:i+ts, j:j+ts]
            pop = self.population[epoch][i:i+ts, j:j+ts]
            frame = np.stack([builtup, vol, pop], axis=0)  # (3, H, W)
            sequence.append(frame)

        # Target is builtup only for the target epoch
        builtup_target = self.temporal_stack[self.target_epoch][i:i+ts, j:j+ts]
        target = builtup_target[np.newaxis, ...]  # (1, H, W)
        return (
            torch.FloatTensor(np.stack(sequence, axis=0)),  # (len(TRAIN_EPOCHS), 3, H, W)
            torch.FloatTensor(target)  # (1, H, W)
        )


# Create dataset with spatial block holdout
# Divide CONUS into 1280×1280px geographic blocks. Every 5th block is val (~20%).
# This ensures val tiles are from entirely different geographic regions than train tiles.
print("\n[DATA] Creating 3-channel dataset with spatial block holdout...", flush=True)
torch.manual_seed(42)
np.random.seed(42)

BLOCK_SIZE = 1280  # pixels per block side (10 × 128px tiles)
dataset = ThreeChannelDataset(ghsl, volume, population, tile_size=128)

height_px, width_px = ref_shape
n_block_cols = (width_px + BLOCK_SIZE - 1) // BLOCK_SIZE

train_indices, val_indices = [], []
for idx, (i, j) in enumerate(dataset.tiles):
    block_row = i // BLOCK_SIZE
    block_col = j // BLOCK_SIZE
    block_id = block_row * n_block_cols + block_col
    if block_id % 5 == 0:
        val_indices.append(idx)
    else:
        train_indices.append(idx)

train_dataset = Subset(dataset, train_indices)
val_dataset   = Subset(dataset, val_indices)

# Save val tile indices for use in validate_2015.py
val_tiles = [list(dataset.tiles[idx]) for idx in val_indices]
with open(str(RESULTS_DIR / "val_tile_indices.json"), "w") as f:
    json.dump(val_tiles, f)

total = len(dataset)
print(f"  Total tiles: {total}, Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
print(f"  Spatial block size: {BLOCK_SIZE}px | Val blocks: every 5th block", flush=True)
print(f"  Val tile indices saved to results/metrics/val_tile_indices.json", flush=True)

# Verify shape
sample_x, sample_y = dataset[0]
print(f"  Sample input shape: {sample_x.shape}")   # (8, 3, 128, 128)
print(f"  Sample target shape: {sample_y.shape}")  # (1, 128, 128)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# =====================================================
# Model Setup
# =====================================================
print("\n[MODEL] Creating 3-channel ConvLSTM with MC Dropout...", flush=True)

model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.out_channels == 1:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        else:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


model.apply(init_weights)
print(f"  Parameters: {model.count_parameters():,}", flush=True)
print(f"  MC Dropout: p={MC_DROPOUT}", flush=True)
print(f"  Input channels: 3 (built-up + volume + population)", flush=True)


# =====================================================
# Training
# =====================================================
def predict_future_step(model, sequences):
    batch_size, seq_len, channels, height, width = sequences.size()
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

    pred = model.decode(layer_hiddens)
    return pred


print(f"\n[TRAIN] Starting training for {NUM_EPOCHS} epochs...", flush=True)
print(f"  Batch size: {BATCH_SIZE}", flush=True)
print(f"  Learning rate: {LEARNING_RATE}", flush=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

best_val_loss = float('inf')
best_state = None
train_losses = []
val_losses = []
t_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for sequences, targets in train_loader:
        optimizer.zero_grad()
        pred = predict_future_step(model, sequences)
        loss = criterion(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, targets in val_loader:
            pred = predict_future_step(model, sequences)
            loss = criterion(pred, targets)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t_start
    eta = elapsed / epoch * (NUM_EPOCHS - epoch)
    lr_now = optimizer.param_groups[0]['lr']

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f} | "
              f"lr={lr_now:.6f} | {elapsed/60:.0f}m elapsed, ETA {eta/60:.0f}m", flush=True)

# Restore best weights
model.load_state_dict(best_state)

# Save model
torch.save(best_state, str(MODELS_DIR / "best_3ch_mc_model.pth"))
print(f"\n  Best model saved: {MODELS_DIR / 'best_3ch_mc_model.pth'} (val_loss={best_val_loss:.6f})", flush=True)

# =====================================================
# Final Evaluation
# =====================================================
print("\n[EVAL] Final evaluation on validation set...", flush=True)
model.eval()
CHANGE_THRESHOLD = 0.01
all_preds, all_targets, all_last_bu = [], [], []
with torch.no_grad():
    for sequences, targets in val_loader:
        pred = predict_future_step(model, sequences)
        all_preds.append(pred)
        all_targets.append(targets)
        all_last_bu.append(sequences[:, -1, 0:1])  # last builtup input (CH0)

all_preds   = torch.cat(all_preds,   dim=0)
all_targets = torch.cat(all_targets, dim=0)
all_last_bu = torch.cat(all_last_bu, dim=0)

mse  = ((all_preds - all_targets) ** 2).mean().item()
mae  = (all_preds - all_targets).abs().mean().item()
rmse = mse ** 0.5

# Figure of Merit
obs_change  = (all_targets - all_last_bu) > CHANGE_THRESHOLD
pred_change = (all_preds   - all_last_bu) > CHANGE_THRESHOLD
intersection = int((obs_change & pred_change).sum().item())
union        = int((obs_change | pred_change).sum().item())
fom = intersection / union if union > 0 else 0.0

total_time = time.time() - t_start
print(f"  MSE  = {mse:.6f}", flush=True)
print(f"  MAE  = {mae:.6f}", flush=True)
print(f"  RMSE = {rmse:.6f}", flush=True)
print(f"  FoM  = {fom:.4f}", flush=True)
print(f"  Total training time: {total_time/3600:.1f} hours", flush=True)

# Save results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results = {
    "model": "ConvLSTM_3ch_MCDropout_SkipDecoder",
    "input_channels": 3,
    "channel_names": ["builtup", "volume", "population"],
    "sequence_length": len(TRAIN_EPOCHS),
    "train_epochs": TRAIN_EPOCHS,
    "target_epoch": TARGET_EPOCH,
    "mc_dropout": MC_DROPOUT,
    "num_epochs": NUM_EPOCHS,
    "val_mse": round(mse, 6),
    "val_mae": round(mae, 6),
    "val_rmse": round(rmse, 6),
    "val_fom": round(fom, 4),
    "params": model.count_parameters(),
    "training_time_hours": round(total_time / 3600, 2),
    "train_losses": [round(l, 6) for l in train_losses],
    "val_losses": [round(l, 6) for l in val_losses],
    "best_val_loss": round(best_val_loss, 6),
    "date": datetime.datetime.now().isoformat()
}

with open(str(RESULTS_DIR / "training_3ch_history.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results/metrics/training_3ch_history.json", flush=True)
print("=" * 60, flush=True)
print("TRAINING COMPLETE", flush=True)
print("=" * 60, flush=True)

# Auto-consolidate into all_results.json
from scripts.consolidate_results import consolidate
consolidate()
