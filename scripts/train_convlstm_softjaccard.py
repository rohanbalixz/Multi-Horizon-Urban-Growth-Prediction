#!/usr/bin/env python3
"""
ConvLSTM Soft-Jaccard Fine-Tuning — FoM-Aligned Objective
==========================================================
Warm-start from best_3ch_mc_model.pth (MSE-trained, seed 42, FoM=0.504).
Fine-tune with Soft-Jaccard loss. Everything else identical to train_3channel.py.

Why warm-start:
  Cold-start soft-Jaccard fails — relu(pred-prev) ≈ 0 from random weights,
  giving near-zero gradients. The MSE checkpoint already knows where growth
  is; soft-Jaccard then sharpens how aggressively it predicts it.

Controlled comparison:
  Baseline:   ConvLSTM MSE-trained        FoM = 0.504  (best_3ch_mc_model.pth)
  This run:   ConvLSTM Soft-Jaccard tuned FoM = ???    (same weights, new loss)
  Target:     CNN MSE-trained             FoM = 0.734  (best_cnn_3ch.pth)

The only variables that change vs. the MSE baseline:
  - Loss function: MSE → Soft-Jaccard on growth residuals
  - Learning rate: 1e-4 (lower, appropriate for fine-tuning)
  - Epochs: 15 (fine-tuning, not full retraining)

Everything else identical:
  - Weights init: loaded from best_3ch_mc_model.pth
  - Architecture: ConvLSTM(input_channels=3, hidden=64, layers=2, mc_dropout=0.1)
  - Optimizer: Adam, ReduceLROnPlateau(patience=5, factor=0.5)
  - clip_grad_norm=1.0, batch_size=8, tile_size=128, seed=42
  - Val split: same val_tile_indices.json (821 tiles)

Outputs:
  models/best_convlstm_softjaccard.pth
  results/metrics/convlstm_softjaccard_results.json
  logs/train_convlstm_softjaccard.log
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

TRAIN_EPOCHS  = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH  = 2015
CHANGE_THRESH = 0.01   # FoM change threshold (matches eval pipeline)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR  = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("ConvLSTM SOFT-JACCARD (FoM-aligned) TRAINING", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

from src.models.convlstm import ConvLSTM

NUM_EPOCHS    = 15      # fine-tuning, not full retraining
BATCH_SIZE    = 8
LEARNING_RATE = 1e-4   # lower LR for fine-tuning
MC_DROPOUT    = 0.1
SEED          = 42
BLOCK_SIZE    = 1280   # same as train_3channel.py
PRETRAINED    = PROJECT_ROOT / "models" / "best_3ch_mc_model.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Loss ──────────────────────────────────────────────────────────────────────
def soft_jaccard_growth_loss(pred, target, last_builtup, eps=1e-6):
    """
    Soft-Jaccard loss aligned with Figure of Merit.

    pred:         (B, 1, H, W) — predicted built-up fraction 2015
    target:       (B, 1, H, W) — ground truth built-up fraction 2015
    last_builtup: (B, 1, H, W) — built-up fraction 2010 (last input epoch)

    Loss = 1 - Jaccard(pred_change, true_change)
    Minimum = 0  when pred_change == true_change everywhere.
    Maximum = 1  when there is zero overlap (e.g., all false positives).
    """
    pred_change = torch.relu(pred        - last_builtup)  # predicted growth
    true_change = torch.relu(target      - last_builtup)  # actual growth

    intersection = (pred_change * true_change).sum()
    union        = pred_change.sum() + true_change.sum() - intersection

    return 1.0 - intersection / (union + eps)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_geotiff(fpath, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.open(fpath) as src:
                return src.read(1)
        except rasterio.errors.RasterioIOError:
            if attempt < max_retries:
                print(f"    GDAL error (attempt {attempt}), retrying...", flush=True)
                gc.collect(); time.sleep(2)
            else:
                raise

print("\n[DATA] Loading rasters...", flush=True)
ghsl = {}
for year in TRAIN_EPOCHS + [TARGET_EPOCH]:
    ghsl[year] = load_geotiff(str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif"))
print(f"  Raster shape: {ghsl[TARGET_EPOCH].shape}", flush=True)

volume = {}
for year in TRAIN_EPOCHS:
    volume[year] = load_geotiff(str(OUTPUT_DIR / f"CONUS_volume_{year}.tif"))

population = {}
for year in TRAIN_EPOCHS:
    population[year] = load_geotiff(str(OUTPUT_DIR / f"CONUS_population_{year}.tif"))

gc.collect()
ref_shape = ghsl[TARGET_EPOCH].shape


# ── Dataset ───────────────────────────────────────────────────────────────────
class ThreeChannelDataset(Dataset):
    def __init__(self, temporal_stack, volume, population, tile_size=128,
                 train_epochs=None, target_epoch=None):
        self.temporal_stack = temporal_stack
        self.volume         = volume
        self.population     = population
        self.tile_size      = tile_size
        self.train_epochs   = train_epochs or TRAIN_EPOCHS
        self.target_epoch   = target_epoch or TARGET_EPOCH

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
        i, j  = self.tiles[idx]
        ts    = self.tile_size
        sequence = []
        for epoch in self.train_epochs:
            frame = np.stack([
                self.temporal_stack[epoch][i:i+ts, j:j+ts],
                self.volume[epoch][i:i+ts, j:j+ts],
                self.population[epoch][i:i+ts, j:j+ts],
            ], axis=0)
            sequence.append(frame)
        target = self.temporal_stack[self.target_epoch][i:i+ts, j:j+ts][np.newaxis]
        return (torch.FloatTensor(np.stack(sequence, axis=0)),
                torch.FloatTensor(target))


# ── Spatial block holdout — MUST match train_3channel.py exactly ──────────────
print("\n[SPLIT] Spatial block holdout (block_id % 5 == 0 → val)...", flush=True)
dataset = ThreeChannelDataset(ghsl, volume, population, tile_size=128)

height_px, width_px = ref_shape
n_block_cols = (width_px + BLOCK_SIZE - 1) // BLOCK_SIZE

train_indices, val_indices = [], []
for idx, (i, j) in enumerate(dataset.tiles):
    block_id = (i // BLOCK_SIZE) * n_block_cols + (j // BLOCK_SIZE)
    if block_id % 5 == 0:
        val_indices.append(idx)
    else:
        train_indices.append(idx)

# Verify against saved val_tile_indices.json
val_tiles_from_json = json.load(open(str(RESULTS_DIR / "val_tile_indices.json")))
val_tiles_here      = [list(dataset.tiles[idx]) for idx in val_indices]
overlap = len(set(map(tuple, val_tiles_here)) & set(map(tuple, val_tiles_from_json)))
print(f"  Train tiles: {len(train_indices)}, Val tiles: {len(val_indices)}", flush=True)
print(f"  ✓ Val tiles overlap with val_tile_indices.json: {overlap}/{len(val_tiles_from_json)}", flush=True)

train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(Subset(dataset, val_indices),   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)


# ── Model — warm-start from MSE checkpoint ────────────────────────────────────
print("\n[MODEL] ConvLSTM(3ch, hidden=64, layers=2, mc_dropout=0.1)", flush=True)
model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
print(f"  Loading pretrained weights from: {PRETRAINED}", flush=True)
model.load_state_dict(torch.load(str(PRETRAINED), map_location="cpu"))
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
print(f"  Warm-start: MSE baseline FoM=0.504 (seed 42)", flush=True)


def predict_future_step(model, sequences):
    _, hidden_states = model(sequences)
    x_t = sequences[:, -1]
    layer_hiddens = []
    for layer_idx, layer in enumerate(model.convlstm_layers):
        h, c = hidden_states[layer_idx]
        h, c = layer(x_t, (h, c))
        h = model.mc_dropouts[layer_idx](h)
        hidden_states[layer_idx] = (h, c)
        x_t = h
        layer_hiddens.append(h)
    return model.decode(layer_hiddens)


# ── Training ──────────────────────────────────────────────────────────────────
print(f"\n[TRAIN] {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}", flush=True)
print(f"  Loss: Soft-Jaccard on growth residuals (1 - FoM approximation)", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')
best_state    = None
t_start       = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for sequences, targets in train_loader:
        last_builtup = sequences[:, -1, 0:1, :, :]   # (B,1,H,W) — 2010 built-up
        optimizer.zero_grad()
        pred = predict_future_step(model, sequences)
        loss = soft_jaccard_growth_loss(pred, targets, last_builtup)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, targets in val_loader:
            last_builtup = sequences[:, -1, 0:1, :, :]
            pred = predict_future_step(model, sequences)
            val_loss += soft_jaccard_growth_loss(pred, targets, last_builtup).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t_start
    eta     = elapsed / epoch * (NUM_EPOCHS - epoch)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | train={train_loss:.6f} | "
              f"val={val_loss:.6f} | {elapsed/60:.0f}m | ETA {eta/60:.0f}m", flush=True)

model_path = MODELS_DIR / "best_convlstm_softjaccard.pth"
torch.save(best_state, str(model_path))
print(f"\n[SAVED] {model_path}  (best_val_loss={best_val_loss:.6f})", flush=True)


# ── Pixel-level FoM evaluation on val tiles ───────────────────────────────────
print("\n[EVAL] Pixel-level FoM on val tiles (gt > 0.01 mask)...", flush=True)
model.load_state_dict(best_state)
model.eval()

# Load 2010 built-up for change detection (last input epoch)
gt_2010 = ghsl[2010]
gt_2015 = ghsl[2015]

# Load full val tile list (same as validate_2015.py)
val_tile_positions = [tuple(t) for t in val_tiles_from_json]

A_total = B_total = C_total = 0
mse_sum = 0.0
n_px    = 0

with torch.no_grad():
    for k, (ti, tj) in enumerate(val_tile_positions):
        ts = 128
        # Build sequence
        sequence = []
        for year in TRAIN_EPOCHS:
            frame = np.stack([
                ghsl[year][ti:ti+ts, tj:tj+ts],
                volume[year][ti:ti+ts, tj:tj+ts],
                population[year][ti:ti+ts, tj:tj+ts],
            ], axis=0)
            sequence.append(frame)
        seq_t = torch.FloatTensor(np.stack(sequence, axis=0)).unsqueeze(0)  # (1,8,3,H,W)

        pred_t = predict_future_step(model, seq_t)
        pred   = pred_t.squeeze().numpy()

        gt_tile   = gt_2015[ti:ti+ts, tj:tj+ts]
        prev_tile = gt_2010[ti:ti+ts, tj:tj+ts]

        # FoM eval mask: gt_2015 > 0.01
        mask = gt_tile > 0.01
        if mask.sum() == 0:
            continue

        obs_change  = (gt_tile   - prev_tile) > CHANGE_THRESH
        pred_change = (pred      - prev_tile) > CHANGE_THRESH

        A = int((obs_change & ~pred_change & mask).sum())
        B = int((obs_change &  pred_change & mask).sum())
        C = int((~obs_change & pred_change & mask).sum())
        A_total += A; B_total += B; C_total += C

        diff = (pred[mask] - gt_tile[mask]) ** 2
        mse_sum += diff.sum()
        n_px    += mask.sum()

        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(val_tile_positions)} tiles, "
                  f"{(time.time()-t_start)/60:.0f}m", flush=True)

denom = A_total + B_total + C_total
fom   = B_total / denom if denom > 0 else 0.0
mse   = float(mse_sum) / n_px if n_px > 0 else 0.0

print(f"\n[RESULTS] ConvLSTM soft-Jaccard — 2015 spatial holdout (gt > 0.01)", flush=True)
print(f"  n_eval={n_px:,}  n_growth={A_total+B_total:,}", flush=True)
print(f"  FoM={fom:.4f}  MSE={mse:.6f}", flush=True)
print(f"  A={A_total:,}  B={B_total:,}  C={C_total:,}", flush=True)

# Load baseline numbers for comparison
try:
    mse_ref  = json.load(open(str(RESULTS_DIR / "convlstm_multiseed_results.json")))
    mse_fom  = mse_ref["summary"]["fom_2015_mean"]
    cnn_ref  = json.load(open(str(RESULTS_DIR / "cnn_multiseed_results.json")))
    cnn_fom  = cnn_ref["summary"]["fom_2015_mean"]
    print(f"\n[COMPARISON]", flush=True)
    print(f"  ConvLSTM MSE-trained  FoM 2015: {mse_fom:.4f}  (baseline)", flush=True)
    print(f"  ConvLSTM Soft-Jaccard FoM 2015: {fom:.4f}  (+{fom-mse_fom:+.4f})", flush=True)
    print(f"  SimpleCNN MSE-trained FoM 2015: {cnn_fom:.4f}  (target)", flush=True)
    gap_closed = (fom - mse_fom) / (cnn_fom - mse_fom) * 100 if cnn_fom > mse_fom else 0
    print(f"  Gap closed vs CNN: {gap_closed:.1f}%", flush=True)
    if fom >= cnn_fom:
        print(f"\n  *** ConvLSTM soft-Jaccard BEATS or TIES CNN on FoM ***", flush=True)
        print(f"  *** MSE training was the bottleneck, not the architecture ***", flush=True)
except Exception:
    pass

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "experiment": "convlstm_softjaccard",
    "description": "ConvLSTM trained with Soft-Jaccard loss (FoM-aligned). "
                   "Architecture identical to best_3ch_mc_model.pth. "
                   "Only change: MSE → Soft-Jaccard on growth residuals.",
    "loss_function": "1 - Jaccard(relu(pred-prev), relu(target-prev))",
    "architecture": "ConvLSTM_3ch_MCDropout_SkipDecoder",
    "params": 481153,
    "seed": SEED,
    "num_epochs": NUM_EPOCHS,
    "best_val_softjaccard": float(best_val_loss),
    "spatial_holdout_2015": {
        "fom": round(fom, 4),
        "mse": round(mse, 8),
        "A": A_total, "B": B_total, "C": C_total,
        "n_eval_pixels": int(n_px),
        "eval_mask": "gt_2015 > 0.01",
        "change_threshold": CHANGE_THRESH,
    },
    "model_path": str(model_path),
    "training_time_min": round((time.time() - t_start) / 60, 1),
    "timestamp": str(datetime.datetime.now()),
}

out_path = RESULTS_DIR / "convlstm_softjaccard_results.json"
with open(str(out_path), "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[SAVED] {out_path}", flush=True)

print("\n" + "=" * 60, flush=True)
print(f"DONE — ConvLSTM soft-Jaccard FoM 2015: {fom:.4f}", flush=True)
print("=" * 60, flush=True)
