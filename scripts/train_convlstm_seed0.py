#!/usr/bin/env python3
"""
ConvLSTM Seed-0 Training — Multi-seed Replication
===================================================
Trains the identical 3-channel ConvLSTM (best_3ch_mc_model.pth) with
seed=0 instead of seed=42, then evaluates on the same spatial block holdout.

Purpose: close the symmetry gap between SimpleCNN (2 seeds: 0 and 42)
and ConvLSTM (currently 1 seed: 42). If ConvLSTM seed variance is small
relative to the FoM gap (0.172–0.235), the architecture claim is robust.

Architecture: identical to train_3channel.py
  - ConvLSTM, 2 layers, 64 hidden ch, MC Dropout p=0.1, skip decoder
  - 3 channels/epoch: built-up + volume + population
  - Seq len = 8 epochs (1975–2010)
  - Target: 2015 built-up

Val tiles: same val_tile_indices.json as ALL other experiments.

Outputs:
  - models/best_3ch_mc_model_seed0.pth
  - results/metrics/convlstm_multiseed_results.json
    (appends seed-0 result alongside seed-42 from validation_2015_results.json)

Runtime: ~24 hrs on CPU.
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

SEED = 0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR  = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print(f"ConvLSTM SEED-{SEED} TRAINING (multi-seed replication)", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

torch.manual_seed(SEED)
np.random.seed(SEED)

from src.models.convlstm import ConvLSTM

# ── Config (identical to train_3channel.py) ───────────────────────────────
TRAIN_EPOCHS   = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH   = 2015
TILE_SIZE      = 128
BATCH_SIZE     = 8
NUM_EPOCHS     = 25
LEARNING_RATE  = 5e-4
MC_DROPOUT     = 0.1
BLOCK_SIZE     = 1280
CHANGE_THRESHOLD = 0.01


# ── Dataset (identical to train_3channel.py) ──────────────────────────────
class ThreeChannelDataset(Dataset):
    def __init__(self, ghsl, volume, population, tile_size=128):
        self.ghsl       = ghsl
        self.volume     = volume
        self.population = population
        self.tile_size  = tile_size
        h, w = ghsl[TRAIN_EPOCHS[-1]].shape
        stride = tile_size // 2
        self.tiles = [
            (i, j)
            for i in range(0, h - tile_size + 1, stride)
            for j in range(0, w - tile_size + 1, stride)
            if ghsl[TARGET_EPOCH][i:i+tile_size, j:j+tile_size].mean() > 0.01
        ]
        print(f"  Dataset tiles: {len(self.tiles):,}", flush=True)

    def __len__(self): return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts = self.tile_size
        seq = []
        for epoch in TRAIN_EPOCHS:
            bu  = self.ghsl[epoch][i:i+ts, j:j+ts].astype(np.float32)
            vol = self.volume[epoch][i:i+ts, j:j+ts].astype(np.float32)
            pop = self.population[epoch][i:i+ts, j:j+ts].astype(np.float32)
            seq.append(np.stack([bu, vol, pop], axis=0))
        seq_t = torch.FloatTensor(np.stack(seq, axis=0))
        tgt   = torch.FloatTensor(
            self.ghsl[TARGET_EPOCH][i:i+ts, j:j+ts].astype(np.float32)
        ).unsqueeze(0)
        return seq_t, tgt


# ── Load rasters ──────────────────────────────────────────────────────────
print("\n[DATA] Loading rasters...", flush=True)

def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)

ghsl = {yr: load_tif(OUTPUT_DIR / f"CONUS_builtup_{yr}.tif")
        for yr in TRAIN_EPOCHS + [TARGET_EPOCH]}
volume = {yr: load_tif(OUTPUT_DIR / f"CONUS_volume_{yr}.tif")
          for yr in TRAIN_EPOCHS}
population = {yr: load_tif(OUTPUT_DIR / f"CONUS_population_{yr}.tif")
              for yr in TRAIN_EPOCHS}

ref_shape = ghsl[TRAIN_EPOCHS[-1]].shape
print(f"  Raster shape: {ref_shape}", flush=True)
gc.collect()


# ── Spatial block holdout — SAME val_tile_indices.json ────────────────────
print("\n[SPLIT] Using same val_tile_indices.json as all other experiments...", flush=True)
val_idx_path = RESULTS_DIR / "val_tile_indices.json"
if not val_idx_path.exists():
    print("ERROR: val_tile_indices.json not found — run train_3channel.py first.")
    sys.exit(1)

with open(val_idx_path) as f:
    val_tiles_coords = [tuple(t) for t in json.load(f)]

dataset = ThreeChannelDataset(ghsl, volume, population, tile_size=TILE_SIZE)

# Match dataset tile indices to saved val_tile_indices
tile_to_idx = {tuple(t): i for i, t in enumerate(dataset.tiles)}
val_indices   = [tile_to_idx[t] for t in val_tiles_coords if tuple(t) in tile_to_idx]
train_indices = [i for i in range(len(dataset)) if i not in set(val_indices)]

train_dataset = Subset(dataset, train_indices)
val_dataset   = Subset(dataset, val_indices)
print(f"  Train: {len(train_dataset):,}  Val: {len(val_dataset):,}", flush=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ── Model ─────────────────────────────────────────────────────────────────
print("\n[MODEL] ConvLSTM (identical architecture to train_3channel.py)...", flush=True)
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
print(f"  Seed: {SEED}", flush=True)


# ── Autoregressive prediction (identical to train_3channel.py) ───────────
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
    return model.decode(layer_hiddens)


# ── Training loop ─────────────────────────────────────────────────────────
print(f"\n[TRAIN] {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, batch={BATCH_SIZE}...", flush=True)

criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

best_val_loss = float('inf')
best_state    = None
train_losses, val_losses = [], []
t_start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
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
    eta     = elapsed / epoch * (NUM_EPOCHS - epoch)
    lr_now  = optimizer.param_groups[0]['lr']

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f} | "
              f"lr={lr_now:.6f} | {elapsed/60:.0f}m elapsed, ETA {eta/60:.0f}m", flush=True)

model.load_state_dict(best_state)
save_path = MODELS_DIR / "best_3ch_mc_model_seed0.pth"
torch.save(best_state, str(save_path))
print(f"\n  Model saved: {save_path}", flush=True)


# ── Evaluation on val set ─────────────────────────────────────────────────
print("\n[EVAL] Computing FoM on val tiles...", flush=True)
model.eval()
all_preds, all_targets, all_last_bu = [], [], []
with torch.no_grad():
    for sequences, targets in val_loader:
        pred = predict_future_step(model, sequences)
        all_preds.append(pred)
        all_targets.append(targets)
        all_last_bu.append(sequences[:, -1, 0:1])

all_preds   = torch.cat(all_preds,   dim=0)
all_targets = torch.cat(all_targets, dim=0)
all_last_bu = torch.cat(all_last_bu, dim=0)

mse  = float(((all_preds - all_targets) ** 2).mean().item())
mae  = float((all_preds - all_targets).abs().mean().item())
rmse = float(mse ** 0.5)

obs_change  = (all_targets - all_last_bu) > CHANGE_THRESHOLD
pred_change = (all_preds   - all_last_bu) > CHANGE_THRESHOLD
inter = int((obs_change & pred_change).sum().item())
union = int((obs_change | pred_change).sum().item())
fom   = inter / union if union > 0 else 0.0

total_time = time.time() - t_start
print(f"  MSE = {mse:.6f}  MAE = {mae:.6f}  FoM = {fom:.4f}", flush=True)
print(f"  Total time: {total_time/3600:.1f} hrs", flush=True)


# ── Save multiseed results ────────────────────────────────────────────────
# Load seed-42 FoM from validation_2015_results.json for comparison
v15 = json.load(open(RESULTS_DIR / "validation_2015_results.json"))
fom_seed42 = v15["convlstm"]["fom"]
mse_seed42 = v15["convlstm"]["mse"]

foms = [fom, fom_seed42]
mses = [mse, mse_seed42]

multiseed = {
    "experiment": "convlstm_multiseed",
    "description": "ConvLSTM trained with seeds 0 and 42; same architecture, same val tiles",
    "architecture": "ConvLSTM_3ch_MCDropout_SkipDecoder",
    "params": model.count_parameters(),
    "per_seed": {
        "0":  {"fom_2015": round(fom, 4), "mse_2015": round(mse, 6),
               "model_path": "models/best_3ch_mc_model_seed0.pth",
               "training_time_hrs": round(total_time / 3600, 2)},
        "42": {"fom_2015": round(fom_seed42, 4), "mse_2015": round(mse_seed42, 6),
               "model_path": "models/best_3ch_mc_model.pth",
               "source": "validation_2015_results.json"},
    },
    "summary": {
        "n_seeds": 2,
        "seeds": [0, 42],
        "fom_2015_mean":  round(float(np.mean(foms)), 4),
        "fom_2015_std":   round(float(np.std(foms)),  4),
        "fom_2015_min":   round(float(np.min(foms)),  4),
        "fom_2015_max":   round(float(np.max(foms)),  4),
        "mse_2015_mean":  round(float(np.mean(mses)), 6),
        "mse_2015_std":   round(float(np.std(mses)),  6),
    },
    "interpretation": {
        "cnn_fom_2015_mean": 0.734,
        "cnn_fom_2015_std":  0.007,
        "note": (
            "If ConvLSTM std is small relative to CNN-ConvLSTM FoM gap (0.172-0.235), "
            "architecture finding is fully robust to seed choice."
        ),
    },
    "timestamp": str(datetime.datetime.now()),
}

out_path = RESULTS_DIR / "convlstm_multiseed_results.json"
with open(out_path, "w") as f:
    json.dump(multiseed, f, indent=2)
print(f"\n  Results saved: {out_path}", flush=True)

print(f"\n{'='*60}")
print(f"ConvLSTM SEED-{SEED} COMPLETE")
print(f"  FoM (seed 0):  {fom:.4f}")
print(f"  FoM (seed 42): {fom_seed42:.4f}")
print(f"  Gap: {abs(fom - fom_seed42):.4f}  (vs CNN-ConvLSTM gap: 0.172–0.235)")
print(f"{'='*60}\n")
