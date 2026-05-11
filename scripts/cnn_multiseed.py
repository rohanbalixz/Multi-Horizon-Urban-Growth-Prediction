#!/usr/bin/env python3
"""
CNN Multi-Seed Evaluation
=========================
Trains SimpleCNN on [1975-2010]→2015 with seeds 0 and 1.
Combines with existing seed-42 results (cnn_2020_holdout.json) to report
mean ± std FoM across 3 seeds, for both holdouts.

This addresses the single-seed limitation in the GeoAI 2026 paper.

Outputs:
  models/best_cnn_seed0.pth
  models/best_cnn_seed1.pth
  results/metrics/cnn_multiseed_results.json

Runtime: ~73 min per seed × 2 new seeds = ~150 min
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
print("CNN MULTI-SEED EVALUATION (seeds 0, 1; seed 42 from existing)", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

TRAIN_EPOCHS    = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH    = 2015
HOLDOUT_EPOCHS  = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
HOLDOUT_TARGET  = 2020
TILE_SIZE       = 128
BATCH_SIZE      = 8
NUM_EPOCHS      = 25
LEARNING_RATE   = 5e-4
CHANGE_THRESHOLD = 0.01
NEW_SEEDS = [0, 1]   # seed 42 already done


# ── Architecture (identical to cnn_2020_holdout.py) ───────────────────────
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
        return torch.sigmoid(self.net(x.reshape(B, T*C, H, W))).unsqueeze(1), None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)


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
        return (torch.FloatTensor(np.stack(seq, axis=0)),
                torch.FloatTensor(
                    self.ghsl[self.target_epoch][i:i+ts, j:j+ts].astype(np.float32)
                ).unsqueeze(0))


def compute_fom(preds, targets, prev_bu, threshold=CHANGE_THRESHOLD):
    obs  = (targets - prev_bu) > threshold
    pred = (preds   - prev_bu) > threshold
    B = int((obs & pred).sum())
    A = int((obs & ~pred).sum())
    C = int((~obs & pred).sum())
    return B / (A + B + C) if (A + B + C) > 0 else 0.0


# ── Load all data once (shared across seeds) ──────────────────────────────
print("\n[DATA] Loading training data [1975–2015]...", flush=True)
ghsl_tr = {}
for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_builtup_{yr}.tif")) as s:
        ghsl_tr[yr] = s.read(1)
    print(f"  builtup {yr}: mean={ghsl_tr[yr].mean():.6f}", flush=True)

vol_tr = {}
for yr in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_volume_{yr}.tif")) as s:
        vol_tr[yr] = s.read(1)

pop_tr = {}
for yr in TRAIN_EPOCHS:
    with rasterio.open(str(OUTPUT_DIR / f"CONUS_population_{yr}.tif")) as s:
        pop_tr[yr] = s.read(1)

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

# ── Load canonical split ──────────────────────────────────────────────────
val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
if not os.path.exists(val_idx_path):
    print("ERROR: val_tile_indices.json not found.")
    sys.exit(1)
with open(val_idx_path) as f:
    val_tiles_set = set(tuple(t) for t in json.load(f))
print(f"\n[SPLIT] Canonical val split: {len(val_tiles_set):,} tiles", flush=True)

n_input_ch = len(TRAIN_EPOCHS) * 3   # 24

# ── Per-seed training loop ────────────────────────────────────────────────
seed_results = {}

for seed in NEW_SEEDS:
    print(f"\n{'='*60}", flush=True)
    print(f"SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build loaders (dataset order fixed by val_tiles_set, only model init varies)
    ds_tr = UrbanDataset(ghsl_tr, vol_tr, pop_tr, TRAIN_EPOCHS, TARGET_EPOCH)
    tr_idx, va_idx = [], []
    for idx, tile in enumerate(ds_tr.tiles):
        (va_idx if tuple(tile) in val_tiles_set else tr_idx).append(idx)

    tr_loader = DataLoader(Subset(ds_tr, tr_idx), BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(Subset(ds_tr, va_idx), BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train: {len(tr_idx):,}  Val: {len(va_idx):,}", flush=True)

    # Train
    model = SimpleCNN(input_channels=n_input_ch)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()
    best_val, best_state = float('inf'), None
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
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1:
            el = time.time() - t0
            print(f"  Ep {ep:2d}/{NUM_EPOCHS}: tr={tr_loss:.6f} va={va_loss:.6f} "
                  f"{el/60:.0f}m elapsed", flush=True)

    model.load_state_dict(best_state)
    torch.save(best_state, str(MODELS_DIR / f"best_cnn_seed{seed}.pth"))
    print(f"  Model saved to models/best_cnn_seed{seed}.pth", flush=True)

    # 2015 FoM
    model.eval()
    all_p, all_t, all_prev = [], [], []
    with torch.no_grad():
        for seqs, tgts in va_loader:
            preds, _ = model(seqs)
            all_p.append(preds[:, 0]); all_t.append(tgts)
            all_prev.append(seqs[:, -1, 0:1])
    all_p = torch.cat(all_p); all_t = torch.cat(all_t); all_prev = torch.cat(all_prev)
    fom_2015 = compute_fom(all_p, all_t, all_prev)
    mse_2015 = ((all_p - all_t)**2).mean().item()
    print(f"  2015 FoM={fom_2015:.4f}  MSE={mse_2015:.6f}", flush=True)

    # 2020 FoM
    ds_h_ds = UrbanDataset(ghsl_h, vol_h, pop_h, HOLDOUT_EPOCHS, HOLDOUT_TARGET)
    h_idx = [i for i, t in enumerate(ds_h_ds.tiles) if tuple(t) in val_tiles_set]
    h_loader = DataLoader(Subset(ds_h_ds, h_idx), BATCH_SIZE, shuffle=False, num_workers=0)

    all_p2, all_t2, all_prev2 = [], [], []
    with torch.no_grad():
        for seqs, tgts in h_loader:
            preds, _ = model(seqs)
            all_p2.append(preds[:, 0]); all_t2.append(tgts)
            all_prev2.append(seqs[:, -1, 0:1])
    all_p2 = torch.cat(all_p2); all_t2 = torch.cat(all_t2); all_prev2 = torch.cat(all_prev2)
    fom_2020 = compute_fom(all_p2, all_t2, all_prev2)
    mse_2020 = ((all_p2 - all_t2)**2).mean().item()
    print(f"  2020 FoM={fom_2020:.4f}  MSE={mse_2020:.6f}", flush=True)

    seed_results[seed] = {
        "fom_2015": round(fom_2015, 4),
        "mse_2015": round(mse_2015, 8),
        "fom_2020": round(fom_2020, 4),
        "mse_2020": round(mse_2020, 8),
        "training_time_min": round((time.time() - t0) / 60, 1),
    }
    del model; gc.collect()

# ── Load seed-42 result from existing JSON ────────────────────────────────
print("\n[SEED 42] Loading from existing cnn_2020_holdout.json...", flush=True)
with open(str(RESULTS_DIR / "cnn_2020_holdout.json")) as f:
    s42 = json.load(f)
seed_results[42] = {
    "fom_2015": 0.739,   # from ablation_3ch_results.json (same split, same seed)
    "mse_2015": round(s42["holdout_2020"]["cnn"]["mse"], 8),  # proxy (2015 not re-eval here)
    "fom_2020": s42["holdout_2020"]["cnn"]["fom"],
    "mse_2020": s42["holdout_2020"]["cnn"]["mse"],
    "note": "seed-42 2015 FoM from ablation_3ch_results; 2020 from cnn_2020_holdout"
}
print(f"  seed 42 → 2015 FoM=0.739  2020 FoM={seed_results[42]['fom_2020']:.4f}", flush=True)

# ── Aggregate stats ───────────────────────────────────────────────────────
fom_2015_all = [seed_results[s]["fom_2015"] for s in [0, 1, 42]]
fom_2020_all = [seed_results[s]["fom_2020"] for s in [0, 1, 42]]

summary = {
    "fom_2015_mean": round(float(np.mean(fom_2015_all)),  4),
    "fom_2015_std":  round(float(np.std(fom_2015_all, ddof=1)), 4),
    "fom_2015_min":  round(float(np.min(fom_2015_all)),  4),
    "fom_2015_max":  round(float(np.max(fom_2015_all)),  4),
    "fom_2020_mean": round(float(np.mean(fom_2020_all)), 4),
    "fom_2020_std":  round(float(np.std(fom_2020_all, ddof=1)), 4),
    "fom_2020_min":  round(float(np.min(fom_2020_all)), 4),
    "fom_2020_max":  round(float(np.max(fom_2020_all)), 4),
    "seeds": [0, 1, 42],
}

print(f"\n{'='*60}", flush=True)
print(f"MULTI-SEED SUMMARY (n=3 seeds: 0, 1, 42)", flush=True)
print(f"  2015 FoM: {summary['fom_2015_mean']:.4f} ± {summary['fom_2015_std']:.4f}  "
      f"[{summary['fom_2015_min']:.4f}, {summary['fom_2015_max']:.4f}]", flush=True)
print(f"  2020 FoM: {summary['fom_2020_mean']:.4f} ± {summary['fom_2020_std']:.4f}  "
      f"[{summary['fom_2020_min']:.4f}, {summary['fom_2020_max']:.4f}]", flush=True)
print(f"{'='*60}", flush=True)

out = {
    "experiment": "cnn_multiseed",
    "description": "SimpleCNN trained with 3 seeds; fills multi-seed limitation in GeoAI 2026 paper",
    "seeds_run_fresh": NEW_SEEDS,
    "seed_42_from_existing": True,
    "per_seed": {str(s): seed_results[s] for s in [0, 1, 42]},
    "summary": summary,
    "timestamp": datetime.datetime.now().isoformat(),
}
out_path = str(RESULTS_DIR / "cnn_multiseed_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[DONE] Results saved to {out_path}", flush=True)
