#!/usr/bin/env python3
"""
CNN 2020 Temporal Holdout
=========================
Trains SimpleCNN (3ch, flat stacking) on [1975-2010] → 2015 — identical
to the ablation experiment in run_ablations_3ch.py — saves the model, then
evaluates it on the 2020 temporal holdout using [1980-2015] → 2020.

Why this is a clean evaluation:
  - Training task:  [1975,1980,...,2010] (8 epochs, 24 channels) → 2015
  - Holdout task:   [1980,1985,...,2015] (8 epochs, 24 channels) → 2020
  - Same temporal stride (5yr), same input channel count → same architecture
  - 2020 GHSL downloaded after ConvLSTM training — completely independent
  - Same val_tile_indices.json as ALL other experiments

Outputs:
  - models/best_cnn_3ch.pth
  - results/metrics/cnn_2020_holdout.json

Runtime: ~75 minutes total (73min train + 2min inference)
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
print("CNN 2020 TEMPORAL HOLDOUT", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

torch.manual_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────
TRAIN_EPOCHS    = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH    = 2015
HOLDOUT_EPOCHS  = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]  # 2020 input window
HOLDOUT_TARGET  = 2020
TILE_SIZE       = 128
BATCH_SIZE      = 8
NUM_EPOCHS      = 25
LEARNING_RATE   = 5e-4
CHANGE_THRESHOLD = 0.01


# ── Architecture (identical to run_ablations_3ch.py) ──────────
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x, hidden_states=None):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T * C, H, W)
        return torch.sigmoid(self.net(x_flat)).unsqueeze(1), None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ── Dataset (identical to run_ablations_3ch.py) ───────────────
class FlexibleDataset(Dataset):
    def __init__(self, ghsl, volume, population, tile_size=128):
        self.ghsl       = ghsl
        self.volume     = volume
        self.population = population
        self.tile_size  = tile_size
        self.epochs     = sorted(e for e in ghsl.keys() if e != TARGET_EPOCH)
        self.target_epoch = TARGET_EPOCH

        height, width = list(ghsl.values())[0].shape
        self.tiles = []
        stride = tile_size // 2
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                tile = ghsl[self.epochs[-1]][i:i+tile_size, j:j+tile_size]
                if tile.mean() > 0.01:
                    self.tiles.append((i, j))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts = self.tile_size
        sequence = []
        for epoch in self.epochs:
            bu  = self.ghsl[epoch][i:i+ts, j:j+ts].astype(np.float32)
            vol = self.volume[epoch][i:i+ts, j:j+ts].astype(np.float32)
            pop = self.population[epoch][i:i+ts, j:j+ts].astype(np.float32)
            sequence.append(np.stack([bu, vol, pop], axis=0))
        seq = torch.FloatTensor(np.stack(sequence, axis=0))  # (T, 3, H, W)
        tgt = torch.FloatTensor(
            self.ghsl[self.target_epoch][i:i+ts, j:j+ts].astype(np.float32)
        ).unsqueeze(0)  # (1, H, W)
        return seq, tgt


# ── Phase 1: Load training data ───────────────────────────────
print("\n[DATA] Loading training data [1975-2015]...", flush=True)

ghsl_train = {}
for year in TRAIN_EPOCHS + [TARGET_EPOCH]:
    path = str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")
    with rasterio.open(path) as src:
        ghsl_train[year] = src.read(1)
    print(f"  builtup {year}: mean={ghsl_train[year].mean():.6f}", flush=True)

volume_train = {}
for year in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")) as src:
        volume_train[year] = src.read(1)

population_train = {}
for year in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{year}.tif")) as src:
        population_train[year] = src.read(1)

gc.collect()
print("  Training data loaded.", flush=True)

# ── Load val tile indices (identical split to all experiments) ─
val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
if not os.path.exists(val_idx_path):
    print("ERROR: val_tile_indices.json not found. Run train_3channel.py first.")
    sys.exit(1)
with open(val_idx_path) as f:
    val_tiles_set = set(tuple(t) for t in json.load(f))
print(f"[SPLIT] Loaded {len(val_tiles_set):,} val tiles from spatial block holdout.", flush=True)

# ── Build dataset and loaders ─────────────────────────────────
ds = FlexibleDataset(ghsl_train, volume_train, population_train)
train_indices, val_indices = [], []
for idx, tile in enumerate(ds.tiles):
    if tuple(tile) in val_tiles_set:
        val_indices.append(idx)
    else:
        train_indices.append(idx)

tr_loader = DataLoader(Subset(ds, train_indices), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
va_loader = DataLoader(Subset(ds, val_indices),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"[SPLIT] Train: {len(train_indices):,} tiles | Val: {len(val_indices):,} tiles", flush=True)


# ── Phase 2: Train CNN ────────────────────────────────────────
print("\n[TRAIN] Training SimpleCNN (8 epochs × 3 channels = 24 input channels)...", flush=True)

n_input_ch = len(TRAIN_EPOCHS) * 3  # 24
model = SimpleCNN(input_channels=n_input_ch)
model.apply(init_weights)
print(f"  Parameters: {model.count_parameters():,}", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.MSELoss()

best_val_loss = float('inf')
train_t0 = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for seqs, tgts in tr_loader:
        optimizer.zero_grad()
        preds, _ = model(seqs)
        preds = preds[:, 0]  # (B, 1, H, W)
        loss = criterion(preds, tgts)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(tr_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seqs, tgts in va_loader:
            preds, _ = model(seqs)
            preds = preds[:, 0]
            val_loss += criterion(preds, tgts).item()
    val_loss /= len(va_loader)

    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), str(MODELS_DIR / "best_cnn_3ch.pth"))

    if epoch % 5 == 0 or epoch == 1:
        elapsed = (time.time() - train_t0) / 60
        print(f"  Epoch {epoch:2d}/{NUM_EPOCHS}: train={train_loss:.6f} val={val_loss:.6f} "
              f"best={best_val_loss:.6f} [{elapsed:.0f}m elapsed]", flush=True)

training_time_min = (time.time() - train_t0) / 60
print(f"\n[TRAIN] Done. Best val MSE: {best_val_loss:.6f} | Time: {training_time_min:.1f} min", flush=True)

# Load best checkpoint
model.load_state_dict(torch.load(str(MODELS_DIR / "best_cnn_3ch.pth"),
                                  map_location='cpu', weights_only=True))
model.eval()

# ── Also compute 2015 val FoM for the saved CNN ───────────────
print("\n[EVAL-2015] Computing CNN FoM on 2015 spatial holdout...", flush=True)
all_preds_2015, all_targets_2015, all_last_bu_2015 = [], [], []
with torch.no_grad():
    for seqs, tgts in va_loader:
        preds, _ = model(seqs)
        preds = preds[:, 0]  # (B, 1, H, W)
        all_preds_2015.append(preds)
        all_targets_2015.append(tgts)
        all_last_bu_2015.append(seqs[:, -1, 0:1])  # last input builtup channel

all_preds_2015   = torch.cat(all_preds_2015)
all_targets_2015 = torch.cat(all_targets_2015)
all_last_bu_2015 = torch.cat(all_last_bu_2015)

cnn_2015_mse = ((all_preds_2015 - all_targets_2015) ** 2).mean().item()
obs_ch_2015  = (all_targets_2015 - all_last_bu_2015) > CHANGE_THRESHOLD
pred_ch_2015 = (all_preds_2015   - all_last_bu_2015) > CHANGE_THRESHOLD
inter_2015 = int((obs_ch_2015 & pred_ch_2015).sum().item())
union_2015 = int((obs_ch_2015 | pred_ch_2015).sum().item())
cnn_2015_fom = inter_2015 / union_2015 if union_2015 > 0 else 0.0
print(f"  CNN 2015: MSE={cnn_2015_mse:.6f}, FoM={cnn_2015_fom:.4f}", flush=True)

del all_preds_2015, all_targets_2015, all_last_bu_2015, ds, tr_loader, va_loader
del ghsl_train, volume_train, population_train
gc.collect()


# ── Phase 3: Load 2020 holdout data ──────────────────────────
print("\n[DATA] Loading 2020 holdout data [1980-2015] + 2020 ground truth...", flush=True)

builtup_2020 = {}
for year in HOLDOUT_EPOCHS + [HOLDOUT_TARGET]:
    path = str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")
    with rasterio.open(path) as src:
        builtup_2020[year] = src.read(1)
        if year == HOLDOUT_TARGET:
            ref_profile = src.profile.copy()
    print(f"  builtup {year}: mean={builtup_2020[year].mean():.6f}", flush=True)

volume_2020 = {}
for year in HOLDOUT_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")) as src:
        volume_2020[year] = src.read(1)

population_2020 = {}
for year in HOLDOUT_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{year}.tif")) as src:
        population_2020[year] = src.read(1)

gc.collect()
ref_shape = builtup_2020[2015].shape
print("  2020 holdout data loaded.", flush=True)

val_tiles = sorted(val_tiles_set)  # list of (i,j) tuples


# ── Phase 4: Run CNN inference on 2020 holdout ────────────────
print(f"\n[PREDICT] Running CNN on {len(val_tiles):,} held-out val tiles for 2020...", flush=True)
prediction = np.zeros(ref_shape, dtype=np.float64)
counts      = np.zeros(ref_shape, dtype=np.float64)

infer_t0 = time.time()
model.eval()

with torch.no_grad():
    for n, (i, j) in enumerate(val_tiles):
        frames = []
        for epoch in HOLDOUT_EPOCHS:
            bu  = builtup_2020[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE].astype(np.float32)
            vol = volume_2020[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE].astype(np.float32)
            pop = population_2020[epoch][i:i+TILE_SIZE, j:j+TILE_SIZE].astype(np.float32)
            frames.append(np.stack([bu, vol, pop], axis=0))

        seq = torch.FloatTensor(np.stack(frames, axis=0)).unsqueeze(0)  # (1, 8, 3, H, W)
        pred, _ = model(seq)  # (1, 1, 1, H, W)
        pred_np = pred.squeeze().numpy().clip(0, 1)

        prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred_np
        counts[i:i+TILE_SIZE, j:j+TILE_SIZE] += 1

        if (n + 1) % 200 == 0:
            print(f"  {n+1}/{len(val_tiles)} tiles [{(time.time()-infer_t0):.0f}s]", flush=True)

mask = counts > 0
prediction[mask] /= counts[mask]
prediction = prediction.astype(np.float32)
inference_min = (time.time() - infer_t0) / 60
print(f"  Inference done in {inference_min:.1f} min.", flush=True)


# ── Phase 5: Compute metrics ──────────────────────────────────
print("\n[METRICS] Computing 2020 holdout metrics...", flush=True)

gt = builtup_2020[HOLDOUT_TARGET]
eval_mask   = (counts > 0) & (gt > 0.01)
n_eval_pixels = int(eval_mask.sum())

gt_eval   = gt[eval_mask]
pred_eval = prediction[eval_mask]

cnn_mse  = float(np.mean((gt_eval - pred_eval) ** 2))
cnn_mae  = float(np.mean(np.abs(gt_eval - pred_eval)))
cnn_rmse = float(np.sqrt(cnn_mse))
ss_res = np.sum((gt_eval - pred_eval) ** 2)
ss_tot = np.sum((gt_eval - gt_eval.mean()) ** 2)
cnn_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

# Baselines
linear_pred = np.clip(2 * builtup_2020[2015] - builtup_2020[2010], 0, 1)
persist_pred = builtup_2020[2015]

lin_eval  = linear_pred[eval_mask]
per_eval  = persist_pred[eval_mask]

lin_mse = float(np.mean((gt_eval - lin_eval) ** 2))
per_mse = float(np.mean((gt_eval - per_eval) ** 2))
lin_mae = float(np.mean(np.abs(gt_eval - lin_eval)))

# FoM
LAST_INPUT_YEAR = 2015
obs_change  = (gt           - builtup_2020[LAST_INPUT_YEAR]) > CHANGE_THRESHOLD
pred_change = (prediction   - builtup_2020[LAST_INPUT_YEAR]) > CHANGE_THRESHOLD
lin_change  = (linear_pred  - builtup_2020[LAST_INPUT_YEAR]) > CHANGE_THRESHOLD

obs_ch_val  = obs_change[eval_mask]
pred_ch_val = pred_change[eval_mask]
lin_ch_val  = lin_change[eval_mask]

def fom(obs, pred):
    inter = int((obs & pred).sum())
    union = int((obs | pred).sum())
    return inter / union if union > 0 else 0.0

cnn_fom_2020    = fom(obs_ch_val, pred_ch_val)
linear_fom_2020 = fom(obs_ch_val, lin_ch_val)
n_growth = int(obs_ch_val.sum())
pct_growth = 100.0 * n_growth / n_eval_pixels

# Growth-zone vs stable-zone MSE
growth_mask = eval_mask & obs_change
stable_mask = eval_mask & ~obs_change

cnn_growth_mse  = float(np.mean((gt[growth_mask] - prediction[growth_mask]) ** 2))
lin_growth_mse  = float(np.mean((gt[growth_mask] - linear_pred[growth_mask]) ** 2))
per_growth_mse  = float(np.mean((gt[growth_mask] - persist_pred[growth_mask]) ** 2))
cnn_stable_mse  = float(np.mean((gt[stable_mask] - prediction[stable_mask]) ** 2))

print(f"\n  === CNN 2020 TEMPORAL HOLDOUT RESULTS ===")
print(f"  Eval pixels:        {n_eval_pixels:,}")
print(f"  Growth pixels:      {n_growth:,} ({pct_growth:.2f}%)")
print(f"")
print(f"  CNN    FoM={cnn_fom_2020:.4f}  MSE={cnn_mse:.6f}  MAE={cnn_mae:.6f}  R²={cnn_r2:.4f}")
print(f"  Linear FoM={linear_fom_2020:.4f}  MSE={lin_mse:.6f}  MAE={lin_mae:.6f}")
print(f"  Persist             MSE={per_mse:.6f}  FoM=0.0000")
print(f"")
print(f"  CNN vs Linear FoM:  {cnn_fom_2020/linear_fom_2020:.2f}× better" if linear_fom_2020 > 0 else "  CNN vs Linear FoM: N/A")
print(f"  CNN growth MSE:     {cnn_growth_mse:.6f} vs Linear {lin_growth_mse:.6f} vs Persist {per_growth_mse:.6f}")
print(f"  CNN growth MSE improvement vs linear:  {(1-cnn_growth_mse/lin_growth_mse)*100:.1f}%")
print(f"  CNN stable MSE:     {cnn_stable_mse:.6f}", flush=True)


# ── Save results ──────────────────────────────────────────────
results = {
    "experiment": "cnn_2020_temporal_holdout",
    "description": (
        "SimpleCNN trained on [1975-2010]→2015, evaluated on [1980-2015]→2020. "
        "True temporal holdout: same architecture (24 input channels), same val split, "
        "2020 GHSL data completely independent of training."
    ),
    "training": {
        "model": "SimpleCNN_3ch_flat_stacking",
        "train_epochs": TRAIN_EPOCHS,
        "target_epoch": TARGET_EPOCH,
        "input_channels": n_input_ch,
        "params": model.count_parameters(),
        "num_epochs": NUM_EPOCHS,
        "best_val_mse_2015": round(best_val_loss, 6),
        "cnn_2015_fom": round(cnn_2015_fom, 4),
        "training_time_min": round(training_time_min, 1),
    },
    "holdout_2020": {
        "input_epochs": HOLDOUT_EPOCHS,
        "target_epoch": HOLDOUT_TARGET,
        "n_eval_pixels": n_eval_pixels,
        "n_growth_pixels": n_growth,
        "pct_growth_pixels": round(pct_growth, 2),
        "cnn": {
            "fom": round(cnn_fom_2020, 4),
            "mse": round(cnn_mse, 6),
            "mae": round(cnn_mae, 6),
            "r2": round(cnn_r2, 4),
            "growth_mse": round(cnn_growth_mse, 6),
            "stable_mse": round(cnn_stable_mse, 6),
        },
        "linear": {
            "fom": round(linear_fom_2020, 4),
            "mse": round(lin_mse, 6),
            "mae": round(lin_mae, 6),
            "growth_mse": round(lin_growth_mse, 6),
        },
        "persistence": {
            "fom": 0.0,
            "mse": round(per_mse, 6),
            "growth_mse": round(per_growth_mse, 6),
        },
        "improvement_vs_linear": {
            "fom_ratio": round(cnn_fom_2020 / linear_fom_2020, 3) if linear_fom_2020 > 0 else None,
            "growth_mse_improvement_pct": round((1 - cnn_growth_mse / lin_growth_mse) * 100, 1),
            "overall_mse_improvement_pct": round((1 - cnn_mse / lin_mse) * 100, 1),
        },
    },
    "timestamp": datetime.datetime.now().isoformat(),
}

out_path = str(RESULTS_DIR / "cnn_2020_holdout.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[DONE] Results saved to {out_path}", flush=True)
print(f"[DONE] Model saved to models/best_cnn_3ch.pth", flush=True)
print(f"[DONE] Total time: {(training_time_min + inference_min):.1f} min", flush=True)
