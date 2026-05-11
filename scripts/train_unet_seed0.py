#!/usr/bin/env python3
"""
U-Net Seed-0 Training — Multi-seed replication for Move 2.

Trains SimpleUNet at seed=0 with identical architecture, data, and val tiles
as train_unet (seed=42). Closes the single-run uncertainty on U-Net's FoM.

Architecture: identical to unet_2020_holdout.py / run_ablations_3ch.py
  - SimpleUNet, 24-channel flat-stacked input (8 epochs × 3 ch)
  - 474K parameters
  - Training: [1975-2010] → 2015
  - Val:  block_id % 5 == 0 (821 tiles, same as all other experiments)

Outputs:
  models/best_unet_3ch_seed0.pth
  results/metrics/unet_multiseed_results.json

Runtime: ~90 minutes
"""
import sys, json, time, datetime, warnings, gc
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60, flush=True)
print(f"U-NET SEED-{SEED} TRAINING", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

TRAIN_EPOCHS     = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH     = 2015
TILE_SIZE        = 128
BATCH_SIZE       = 8
NUM_EPOCHS       = 25
LEARNING_RATE    = 5e-4
CHANGE_THRESHOLD = 0.01
BLOCK_SIZE       = 1280


# ── Architecture (identical to unet_2020_holdout.py) ─────────────────────
class SimpleUNet(nn.Module):
    def __init__(self, input_channels=24):
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
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T * C, H, W)
        e1 = self.enc1(x_flat)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ── Dataset ───────────────────────────────────────────────────────────────
class FlatStackDataset(Dataset):
    def __init__(self, ghsl, volume, population, tile_size=128):
        self.ghsl = ghsl
        self.volume = volume
        self.population = population
        self.tile_size = tile_size
        h, w = ghsl[TARGET_EPOCH].shape
        stride = tile_size // 2
        self.tiles = []
        for i in range(0, h - tile_size + 1, stride):
            for j in range(0, w - tile_size + 1, stride):
                if ghsl[TARGET_EPOCH][i:i+tile_size, j:j+tile_size].mean() > 0.01:
                    self.tiles.append((i, j))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts = self.tile_size
        frames = []
        for yr in TRAIN_EPOCHS:
            b = self.ghsl[yr][i:i+ts, j:j+ts]
            v = self.volume[yr][i:i+ts, j:j+ts]
            p = self.population[yr][i:i+ts, j:j+ts]
            frames.append(np.stack([b, v, p], axis=0))
        seq = np.stack(frames, axis=0)               # (8, 3, H, W)
        tgt = self.ghsl[TARGET_EPOCH][i:i+ts, j:j+ts][np.newaxis]  # (1, H, W)
        return torch.FloatTensor(seq), torch.FloatTensor(tgt)


# ── Load rasters ──────────────────────────────────────────────────────────
print("\n[DATA] Loading rasters...", flush=True)

def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)

ghsl       = {yr: load_tif(OUTPUT_DIR / f"CONUS_builtup_{yr}.tif")    for yr in TRAIN_EPOCHS + [TARGET_EPOCH]}
volume     = {yr: load_tif(OUTPUT_DIR / f"CONUS_volume_{yr}.tif")     for yr in TRAIN_EPOCHS}
population = {yr: load_tif(OUTPUT_DIR / f"CONUS_population_{yr}.tif") for yr in TRAIN_EPOCHS}

ref_shape = ghsl[TARGET_EPOCH].shape
print(f"  Raster shape: {ref_shape}", flush=True)
gc.collect()

# ── Split: same block_id % 5 == 0 rule as all other experiments ──────────
print("\n[SPLIT] Spatial block holdout (block_id % 5 == 0 → val)...", flush=True)
dataset = FlatStackDataset(ghsl, volume, population)
h_px, w_px = ref_shape
n_block_cols = (w_px + BLOCK_SIZE - 1) // BLOCK_SIZE

train_idx, val_idx = [], []
for idx, (i, j) in enumerate(dataset.tiles):
    block_id = (i // BLOCK_SIZE) * n_block_cols + (j // BLOCK_SIZE)
    if block_id % 5 == 0:
        val_idx.append(idx)
    else:
        train_idx.append(idx)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)
val_tiles = [list(dataset.tiles[idx]) for idx in val_idx]
print(f"  Train tiles: {len(train_ds)}, Val tiles: {len(val_ds)}", flush=True)

# Verify val tiles match existing val_tile_indices.json
existing_val = json.load(open(RESULTS_DIR / "val_tile_indices.json"))
existing_set = set(map(tuple, existing_val))
new_set      = set(map(tuple, val_tiles))
if existing_set == new_set:
    print("  ✓ Val tiles match val_tile_indices.json exactly", flush=True)
else:
    print(f"  WARNING: Val tile mismatch! existing={len(existing_set)}, new={len(new_set)}", flush=True)
    overlap = len(existing_set & new_set)
    print(f"  Overlap: {overlap}", flush=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────
model = SimpleUNet(input_channels=len(TRAIN_EPOCHS) * 3)
model.apply(init_weights)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n[MODEL] SimpleUNet | params: {n_params:,}", flush=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# ── Training ──────────────────────────────────────────────────────────────
print(f"\n[TRAIN] {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE}", flush=True)
best_val_loss = float("inf")
best_state    = None
t_start       = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(train_loader)

    model.eval()
    vl_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            vl_loss += criterion(model(x), y).item()
    vl_loss /= len(val_loader)
    scheduler.step(vl_loss)

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t_start
    eta = elapsed / epoch * (NUM_EPOCHS - epoch)
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | train={tr_loss:.6f} | "
              f"val={vl_loss:.6f} | {elapsed/60:.0f}m | ETA {eta/60:.0f}m", flush=True)

model.load_state_dict(best_state)
model.eval()
save_path = MODELS_DIR / "best_unet_3ch_seed0.pth"
torch.save(best_state, str(save_path))
print(f"\n[SAVED] {save_path}  (best_val_loss={best_val_loss:.6f})", flush=True)

# ── Pixel-level FoM eval on val tiles ─────────────────────────────────────
print("\n[EVAL] Pixel-level FoM on val tiles (gt > 0.01)...", flush=True)
gt_2015  = ghsl[TARGET_EPOCH]
prev_2010 = ghsl[2010]

prediction = np.zeros(ref_shape, dtype=np.float64)
counts     = np.zeros(ref_shape, dtype=np.float64)

t_inf = time.time()
with torch.no_grad():
    for n_done, (i, j) in enumerate(val_tiles):
        frames = []
        for yr in TRAIN_EPOCHS:
            b = ghsl[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            v = volume[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            p = population[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            frames.append(np.stack([b, v, p], axis=0))
        seq  = np.stack(frames, axis=0)
        x_in = torch.FloatTensor(seq).unsqueeze(0)
        pred = model(x_in).squeeze().numpy().clip(0, 1)
        prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred
        counts[i:i+TILE_SIZE, j:j+TILE_SIZE]     += 1
        if (n_done + 1) % 200 == 0:
            print(f"  {n_done+1}/{len(val_tiles)} tiles, {time.time()-t_inf:.0f}s", flush=True)

mask = counts > 0
prediction[mask] /= counts[mask]
prediction = prediction.astype(np.float32)

eval_mask = (counts > 0) & (gt_2015 > CHANGE_THRESHOLD)
n_eval    = int(eval_mask.sum())
gt_eval   = gt_2015[eval_mask]
pr_eval   = prediction[eval_mask]

mse = float(np.mean((gt_eval - pr_eval) ** 2))
mae = float(np.mean(np.abs(gt_eval - pr_eval)))

obs_ch  = (gt_2015   - prev_2010) > CHANGE_THRESHOLD
pred_ch = (prediction - prev_2010) > CHANGE_THRESHOLD
B = int(( obs_ch[eval_mask] &  pred_ch[eval_mask]).sum())
A = int(( obs_ch[eval_mask] & ~pred_ch[eval_mask]).sum())
C = int((~obs_ch[eval_mask] &  pred_ch[eval_mask]).sum())
fom = B / (A + B + C) if (A + B + C) > 0 else 0.0

n_growth   = int(obs_ch[eval_mask].sum())
pct_growth = 100.0 * n_growth / n_eval

print(f"\n[RESULTS] U-Net seed-0 — 2015 spatial holdout (gt > 0.01)", flush=True)
print(f"  n_eval={n_eval:,}  n_growth={n_growth:,} ({pct_growth:.2f}%)", flush=True)
print(f"  FoM={fom:.4f}  MSE={mse:.6f}  MAE={mae:.6f}", flush=True)
print(f"  A={A:,}  B={B:,}  C={C:,}", flush=True)

# ── Compare seed 0 vs seed 42 and save ───────────────────────────────────
seed42_fom = 0.6756   # from unet_2020_holdout.json["spatial_holdout_2015"]["fom"]

out = {
    "experiment": "unet_multiseed",
    "description": "SimpleUNet trained at seeds 0 and 42, same spatial block holdout",
    "architecture": "SimpleUNet_3ch_flatstack",
    "params": n_params,
    "per_seed": {
        "42": {
            "fom_2015": seed42_fom,
            "model_path": "models/best_unet_3ch.pth",
            "source": "unet_2020_holdout.json[spatial_holdout_2015]",
            "eval_mask": "counts>0 & gt_2015>0.01",
        },
        "0": {
            "fom_2015": round(fom, 4),
            "mse_2015": round(mse, 8),
            "model_path": str(save_path.relative_to(PROJECT_ROOT)),
            "n_eval_pixels": n_eval,
            "n_growth_pixels": n_growth,
            "pct_growth": round(pct_growth, 2),
            "eval_mask": "counts>0 & gt_2015>0.01",
            "A": A, "B": B, "C": C,
            "training_time_min": round((time.time() - t_start) / 60, 1),
        },
    },
    "summary": {
        "n_seeds": 2,
        "seeds": [42, 0],
        "fom_2015_mean": round((seed42_fom + fom) / 2, 4),
        "fom_2015_std":  round(abs(seed42_fom - fom) / 2, 4),
        "fom_2015_min":  round(min(seed42_fom, fom), 4),
        "fom_2015_max":  round(max(seed42_fom, fom), 4),
    },
    "timestamp": str(datetime.datetime.now()),
}

out_path = RESULTS_DIR / "unet_multiseed_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[SAVED] {out_path}", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"DONE — U-Net seed-0 FoM 2015: {fom:.4f}", flush=True)
print(f"       U-Net seed-42 FoM 2015: {seed42_fom:.4f}", flush=True)
print(f"       Seed variance (std): {abs(seed42_fom - fom)/2:.4f}", flush=True)
print(f"{'='*60}", flush=True)
