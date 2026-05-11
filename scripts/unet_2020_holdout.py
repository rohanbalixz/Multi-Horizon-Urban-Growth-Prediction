#!/usr/bin/env python3
"""
U-Net 2020 Temporal Holdout
===========================
Mirrors cnn_2020_holdout.py exactly — same data, same split, same epochs —
but uses SimpleUNet instead of SimpleCNN.

Training:  [1975-2010] (8 epochs × 3 ch = 24 channels) → 2015
Holdout:   [1980-2015] (8 epochs × 3 ch = 24 channels) → 2020

Fills the '---' cell in Table 1 of the GeoAI 2026 paper.

Outputs:
  models/best_unet_3ch.pth
  results/metrics/unet_2020_holdout.json

Runtime: ~92 minutes
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR  = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60, flush=True)
print("U-NET 2020 TEMPORAL HOLDOUT", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_EPOCHS    = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH    = 2015
HOLDOUT_EPOCHS  = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
HOLDOUT_TARGET  = 2020
TILE_SIZE       = 128
BATCH_SIZE      = 8
NUM_EPOCHS      = 25
LEARNING_RATE   = 5e-4
CHANGE_THRESHOLD = 0.01


# ── Architecture (identical to run_ablations_3ch.py) ──────────────────────
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

    def forward(self, x, hidden_states=None):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T * C, H, W)
        e1 = self.enc1(x_flat)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1)).unsqueeze(1), None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


# ── Dataset ───────────────────────────────────────────────────────────────
class UrbanDataset(Dataset):
    def __init__(self, ghsl, volume, population, input_epochs, target_epoch,
                 tile_size=128):
        self.ghsl = ghsl; self.volume = volume; self.population = population
        self.input_epochs = sorted(input_epochs)
        self.target_epoch = target_epoch
        self.tile_size = tile_size
        h, w = ghsl[self.input_epochs[-1]].shape
        stride = tile_size // 2
        self.tiles = [
            (i, j)
            for i in range(0, h - tile_size + 1, stride)
            for j in range(0, w - tile_size + 1, stride)
            if ghsl[self.input_epochs[-1]][i:i+tile_size, j:j+tile_size].mean() > 0.01
        ]

    def __len__(self): return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]; ts = self.tile_size
        seq = []
        for ep in self.input_epochs:
            bu  = self.ghsl[ep][i:i+ts, j:j+ts].astype(np.float32)
            vol = self.volume[ep][i:i+ts, j:j+ts].astype(np.float32)
            pop = self.population[ep][i:i+ts, j:j+ts].astype(np.float32)
            seq.append(np.stack([bu, vol, pop], axis=0))
        seq_t = torch.FloatTensor(np.stack(seq, axis=0))
        tgt   = torch.FloatTensor(
            self.ghsl[self.target_epoch][i:i+ts, j:j+ts].astype(np.float32)
        ).unsqueeze(0)
        return seq_t, tgt


def compute_fom(preds, targets, prev_builtup, threshold=0.01):
    obs_change  = (targets   - prev_builtup) > threshold
    pred_change = (preds     - prev_builtup) > threshold
    B = int((obs_change & pred_change).sum())
    A = int((obs_change & ~pred_change).sum())
    C = int((~obs_change & pred_change).sum())
    return B / (A + B + C) if (A + B + C) > 0 else 0.0


# ── Phase 1: Load training data ───────────────────────────────────────────
print("\n[DATA] Loading training data [1975–2015]...", flush=True)

ghsl_data = {}
for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{yr}.tif")) as s:
        ghsl_data[yr] = s.read(1)
    print(f"  builtup {yr}: mean={ghsl_data[yr].mean():.6f}", flush=True)

vol_data = {}
for yr in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{yr}.tif")) as s:
        vol_data[yr] = s.read(1)

pop_data = {}
for yr in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{yr}.tif")) as s:
        pop_data[yr] = s.read(1)

gc.collect()
print("  Training data loaded.", flush=True)

# ── Load canonical val split ──────────────────────────────────────────────
val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
if not os.path.exists(val_idx_path):
    print("ERROR: val_tile_indices.json not found. Run train_3channel.py first.")
    sys.exit(1)
with open(val_idx_path) as f:
    val_tiles_set = set(tuple(t) for t in json.load(f))
print(f"[SPLIT] Canonical val split: {len(val_tiles_set):,} tiles", flush=True)

ds_train = UrbanDataset(ghsl_data, vol_data, pop_data, TRAIN_EPOCHS, TARGET_EPOCH)
train_idx, val_idx = [], []
for idx, tile in enumerate(ds_train.tiles):
    (val_idx if tuple(tile) in val_tiles_set else train_idx).append(idx)

tr_loader = DataLoader(Subset(ds_train, train_idx), BATCH_SIZE, shuffle=True,  num_workers=0)
va_loader = DataLoader(Subset(ds_train, val_idx),   BATCH_SIZE, shuffle=False, num_workers=0)
print(f"[SPLIT] Train: {len(train_idx):,}  Val: {len(val_idx):,}", flush=True)


# ── Phase 2: Train U-Net ──────────────────────────────────────────────────
print("\n[TRAIN] Training SimpleUNet (8×3=24 input channels)...", flush=True)

model = SimpleUNet(input_channels=len(TRAIN_EPOCHS) * 3)
model.apply(init_weights)
print(f"  Parameters: {model.count_parameters():,}", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.MSELoss()

best_val_loss, best_state = float('inf'), None
t0 = time.time()
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for seqs, tgts in tr_loader:
        optimizer.zero_grad()
        preds, _ = model(seqs)
        loss = criterion(preds[:, 0], tgts)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item()
    tr_loss /= len(tr_loader)

    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for seqs, tgts in va_loader:
            preds, _ = model(seqs)
            va_loss += criterion(preds[:, 0], tgts).item()
    va_loss /= len(va_loader)
    scheduler.step(va_loss)
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if ep % 5 == 0 or ep == 1:
        el = time.time() - t0
        eta = el / ep * (NUM_EPOCHS - ep)
        print(f"  Ep {ep:2d}/{NUM_EPOCHS}: tr={tr_loss:.6f} va={va_loss:.6f} "
              f"{el/60:.0f}m elapsed ETA {eta/60:.0f}m", flush=True)

model.load_state_dict(best_state)
torch.save(best_state, str(MODELS_DIR / "best_unet_3ch.pth"))
print(f"  Model saved to models/best_unet_3ch.pth", flush=True)

# ── Phase 2b: 2015 validation FoM ────────────────────────────────────────
print("\n[EVAL] Computing 2015 spatial holdout FoM...", flush=True)
model.eval()
all_p, all_t, all_prev = [], [], []
with torch.no_grad():
    for seqs, tgts in va_loader:
        preds, _ = model(seqs)
        all_p.append(preds[:, 0]); all_t.append(tgts)
        all_prev.append(seqs[:, -1, 0:1])

all_p    = torch.cat(all_p)
all_t    = torch.cat(all_t)
all_prev = torch.cat(all_prev)
mse_2015 = ((all_p - all_t) ** 2).mean().item()
fom_2015 = compute_fom(all_p, all_t, all_prev)
print(f"  2015 FoM={fom_2015:.4f}  MSE={mse_2015:.6f}", flush=True)


# ── Phase 3: Load 2020 holdout data ──────────────────────────────────────
print("\n[DATA] Loading 2020 holdout data [1980–2020]...", flush=True)
ghsl_h = {}
for yr in HOLDOUT_EPOCHS + [HOLDOUT_TARGET]:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{yr}.tif")) as s:
        ghsl_h[yr] = s.read(1)
    print(f"  builtup {yr}: mean={ghsl_h[yr].mean():.6f}", flush=True)

vol_h = {}
for yr in HOLDOUT_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{yr}.tif")) as s:
        vol_h[yr] = s.read(1)

pop_h = {}
for yr in HOLDOUT_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{yr}.tif")) as s:
        pop_h[yr] = s.read(1)

gc.collect()

# ── Phase 4: Evaluate on 2020 ─────────────────────────────────────────────
print("\n[EVAL] Evaluating on 2020 blind temporal holdout...", flush=True)

ds_h = UrbanDataset(ghsl_h, vol_h, pop_h, HOLDOUT_EPOCHS, HOLDOUT_TARGET)
h_idx = [i for i, tile in enumerate(ds_h.tiles) if tuple(tile) in val_tiles_set]
h_loader = DataLoader(Subset(ds_h, h_idx), BATCH_SIZE, shuffle=False, num_workers=0)
print(f"  Holdout val tiles: {len(h_idx):,}", flush=True)

model.eval()
all_p2, all_t2, all_prev2 = [], [], []
with torch.no_grad():
    for seqs, tgts in h_loader:
        preds, _ = model(seqs)
        all_p2.append(preds[:, 0]); all_t2.append(tgts)
        all_prev2.append(seqs[:, -1, 0:1])

all_p2    = torch.cat(all_p2)
all_t2    = torch.cat(all_t2)
all_prev2 = torch.cat(all_prev2)

n_px     = int(all_p2.numel())
n_growth = int(((all_t2 - all_prev2) > CHANGE_THRESHOLD).sum())
mse_2020 = ((all_p2 - all_t2) ** 2).mean().item()
mae_2020 = (all_p2 - all_t2).abs().mean().item()
fom_2020 = compute_fom(all_p2, all_t2, all_prev2)
r2_2020  = 1 - ((all_p2-all_t2)**2).sum().item() / ((all_t2-all_t2.mean())**2).sum().item()

g_mask = (all_t2 - all_prev2) > CHANGE_THRESHOLD
s_mask = ~g_mask
gmse = ((all_p2[g_mask] - all_t2[g_mask])**2).mean().item() if g_mask.any() else float('nan')
smse = ((all_p2[s_mask] - all_t2[s_mask])**2).mean().item() if s_mask.any() else float('nan')

print(f"  2020 FoM={fom_2020:.4f}  MSE={mse_2020:.6f}  "
      f"growth_MSE={gmse:.6f}  stable_MSE={smse:.6f}", flush=True)
print(f"  n_eval_pixels={n_px:,}  n_growth={n_growth:,} ({100*n_growth/n_px:.1f}%)", flush=True)

total_time = time.time() - t0

# ── Save results ──────────────────────────────────────────────────────────
results = {
    "experiment": "unet_2020_temporal_holdout",
    "description": "SimpleUNet trained on [1975-2010]→2015, evaluated on [1980-2015]→2020",
    "model": "SimpleUNet_3ch_flatstack",
    "model_path": "models/best_unet_3ch.pth",
    "params": model.count_parameters(),
    "seed": SEED,
    "training_time_min": round(total_time / 60, 1),
    "spatial_holdout_2015": {
        "fom": round(fom_2015, 4),
        "mse": round(mse_2015, 8),
    },
    "holdout_2020": {
        "unet": {
            "fom": round(fom_2020, 4),
            "mse": round(mse_2020, 8),
            "mae": round(mae_2020, 8),
            "r2":  round(r2_2020, 6),
            "growth_mse": round(gmse, 8),
            "stable_mse": round(smse, 8),
        }
    },
    "n_eval_pixels": n_px,
    "n_growth_pixels": n_growth,
    "pct_growth_pixels": round(100 * n_growth / n_px, 2),
    "timestamp": datetime.datetime.now().isoformat(),
}

out_path = str(RESULTS_DIR / "unet_2020_holdout.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[DONE] Results saved to {out_path}", flush=True)
print(f"  2015 FoM={fom_2015:.4f}  MSE={mse_2015:.6f}", flush=True)
print(f"  2020 FoM={fom_2020:.4f}  MSE={mse_2020:.6f}", flush=True)
print(f"  Total time: {total_time/60:.1f} min", flush=True)
