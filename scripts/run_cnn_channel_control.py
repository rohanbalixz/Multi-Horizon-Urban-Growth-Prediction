#!/usr/bin/env python3
"""
CNN Channel Count Control Experiment
======================================
Resolves the confound in run_cnn_multihorizon.py:

  CNN at 5yr uses 8×3=24 input channels (FoM=0.739)
  CNN at 10yr uses 7×3=21 input channels (FoM=0.675)
  CNN at 20yr uses 5×3=15 input channels (FoM=0.687)

The degradation at longer horizons could mean either:
  (A) Temporal memory becomes important at longer horizons, OR
  (B) CNN degrades simply because it has fewer input features

To separate these, this script fixes the HORIZON at 5yr (last input=2010,
target=2015) and varies only the number of historical epochs given to CNN:

  CNN-5yr-5ep: input=[1990,1995,2000,2005,2010]  → 15ch  (same as 20yr CNN)
  CNN-5yr-7ep: input=[1980,1985,1990,1995,2000,2005,2010] → 21ch (same as 10yr CNN)
  CNN-5yr-8ep: already trained, FoM=0.739 (from cnn_multihorizon_results.json)

Then compares:
  CNN-5yr-5ep (15ch, 5yr)  vs  CNN-20yr (15ch, 20yr)  → same channels, different horizon
  CNN-5yr-7ep (21ch, 5yr)  vs  CNN-10yr (21ch, 10yr)  → same channels, different horizon

Interpretation:
  If CNN-5yr-Nep ≈ CNN-Xyr (same N channels): degradation is channel count (confound)
  If CNN-5yr-Nep >> CNN-Xyr (same N channels): degradation is genuine horizon effect

Results: results/metrics/cnn_channel_control_results.json
"""
import sys, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR  = PROJECT_ROOT / "geotiff_exports"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

print("=" * 60, flush=True)
print("CNN CHANNEL COUNT CONTROL EXPERIMENT", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio

torch.manual_seed(42)
np.random.seed(42)

TARGET_EPOCH     = 2015
TILE_SIZE        = 128
BATCH_SIZE       = 8
NUM_EPOCHS       = 25
CHANGE_THRESHOLD = 0.01

# All three configs share the same task: last input=2010, predict 2015 (5yr horizon)
# Only the number of historical epochs varies.
CONFIGS = {
    "5yr_5ep": [1990, 1995, 2000, 2005, 2010],           # 15ch — same as CNN-20yr
    "5yr_7ep": [1980, 1985, 1990, 1995, 2000, 2005, 2010], # 21ch — same as CNN-10yr
}

ALL_INPUT_EPOCHS = sorted(set(e for ep in CONFIGS.values() for e in ep))

# =====================================================
# Load data
# =====================================================
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
        sequence = torch.FloatTensor(np.stack(frames, axis=0))
        target   = torch.FloatTensor(ghsl[TARGET_EPOCH][i:i+TILE_SIZE, j:j+TILE_SIZE]).unsqueeze(0)
        return sequence, target


def make_loaders(input_epochs):
    ds = HorizonDataset(input_epochs)
    val_idx_path = RESULTS_DIR / "val_tile_indices.json"
    with open(val_idx_path) as f:
        val_tiles_set = set(tuple(t) for t in json.load(f))
    train_idx, val_idx = [], []
    for idx, tile in enumerate(ds.tiles):
        (val_idx if tuple(tile) in val_tiles_set else train_idx).append(idx)
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val tiles", flush=True)
    tr = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va = DataLoader(Subset(ds, val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return tr, va

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
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') \
            if m.out_channels != 1 else nn.init.normal_(m.weight, 0.0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# =====================================================
# Training + evaluation
# =====================================================
def train_and_eval(model, tr_loader, va_loader, label):
    t0 = time.time()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    best_val, best_state = float("inf"), None

    for epoch in range(NUM_EPOCHS):
        model.train()
        tr_loss = 0.0
        for s, t in tr_loader:
            optimizer.zero_grad()
            loss = criterion(model(s), t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        va_loss = 0
        with torch.no_grad():
            for s, t in va_loader:
                va_loss += criterion(model(s), t).item()
        va_loss /= len(va_loader)

        scheduler.step(va_loss)
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1}/{NUM_EPOCHS}: train={tr_loss:.6f}  "
                  f"val={va_loss:.6f}  {elapsed/60:.0f}m  ETA {elapsed/(epoch+1)*(NUM_EPOCHS-epoch-1)/60:.0f}m",
                  flush=True)

    model.load_state_dict(best_state)
    model.eval()

    all_p, all_t, all_lb = [], [], []
    with torch.no_grad():
        for s, t in va_loader:
            all_p.append(model(s))
            all_t.append(t)
            all_lb.append(s[:, -1, 0:1])

    p = torch.cat(all_p); tg = torch.cat(all_t); lb = torch.cat(all_lb)
    mse  = ((p - tg) ** 2).mean().item()
    mae  = (p - tg).abs().mean().item()
    rmse = mse ** 0.5
    ss_res = ((tg - p) ** 2).sum().item()
    ss_tot = ((tg - tg.mean()) ** 2).sum().item()
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    obs  = (tg  - lb) > CHANGE_THRESHOLD
    pred = (p   - lb) > CHANGE_THRESHOLD
    inter, union = int((obs & pred).sum()), int((obs | pred).sum())
    fom  = inter / union if union > 0 else 0.0

    elapsed = time.time() - t0
    print(f"  {label}: MSE={mse:.6f}  MAE={mae:.6f}  R²={r2:.4f}  FoM={fom:.4f}  ({elapsed/60:.0f}m)",
          flush=True)
    return {
        "val_mse": round(mse, 6), "val_mae": round(mae, 6),
        "val_rmse": round(rmse, 6), "val_r2": round(r2, 4),
        "val_fom": round(fom, 4), "params": model.count_parameters(),
        "training_time_min": round(elapsed / 60, 1), "epochs": NUM_EPOCHS,
    }

# =====================================================
# Run controlled experiments
# =====================================================
results = {
    "experiment": "cnn_channel_count_control",
    "description": (
        "Controls for input channel count in CNN multi-horizon comparison. "
        "All configs target 5yr task (last input=2010 -> 2015). Only the number "
        "of historical epochs varies. Compares FoM of CNN-5yr-Nep vs CNN-Xyr "
        "where both have identical input channel counts."
    ),
    "task": "5yr forecast (last_input=2010, target=2015)",
    "configs": {},
    "timestamp": datetime.datetime.now().isoformat(),
}

for name, input_epochs in CONFIGS.items():
    n_ch = len(input_epochs) * 3
    print(f"\n{'='*60}", flush=True)
    print(f"CONFIG: {name}  →  {len(input_epochs)} epochs × 3ch = {n_ch} input channels", flush=True)
    print(f"  Input epochs: {input_epochs}", flush=True)

    tr_loader, va_loader = make_loaders(input_epochs)
    model = SimpleCNN(input_channels=n_ch)
    model.apply(init_weights)
    print(f"  Parameters: {model.count_parameters():,}", flush=True)

    res = train_and_eval(model, tr_loader, va_loader, label=f"CNN-{name}")
    res["input_epochs"]   = input_epochs
    res["n_input_epochs"] = len(input_epochs)
    res["n_channels"]     = n_ch
    res["horizon_years"]  = 5
    res["last_input_year"] = 2010
    results["configs"][name] = res
    del model; gc.collect()

# =====================================================
# Load existing multi-horizon CNN results for comparison
# =====================================================
mh_path = RESULTS_DIR / "cnn_multihorizon_results.json"
with open(mh_path) as f:
    mh = json.load(f)

existing = {
    "5yr_8ep":  {"fom": mh["horizons"]["5yr"]["val_fom"],  "mse": mh["horizons"]["5yr"]["val_mse"],  "ch": 24, "horizon": 5},
    "10yr_7ep": {"fom": mh["horizons"]["10yr"]["val_fom"], "mse": mh["horizons"]["10yr"]["val_mse"], "ch": 21, "horizon": 10},
    "20yr_5ep": {"fom": mh["horizons"]["20yr"]["val_fom"], "mse": mh["horizons"]["20yr"]["val_mse"], "ch": 15, "horizon": 20},
}

# =====================================================
# Verdict table
# =====================================================
print(f"\n{'='*60}", flush=True)
print("CHANNEL CONTROL — VERDICT TABLE", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Config':<14} {'Ch':>4} {'Horizon':>8} {'FoM':>8} {'MSE':>10}", flush=True)
print("-" * 48, flush=True)

all_rows = {}
for name, res in results["configs"].items():
    all_rows[name] = {"ch": res["n_channels"], "horizon": res["horizon_years"],
                      "fom": res["val_fom"], "mse": res["val_mse"]}
all_rows.update(existing)

for name in ["5yr_8ep", "5yr_7ep", "10yr_7ep", "5yr_5ep", "20yr_5ep"]:
    if name not in all_rows:
        continue
    r = all_rows[name]
    print(f"  {name:<12} {r['ch']:>4} {r['horizon']:>7}yr {r['fom']:>8.4f} {r['mse']:>10.6f}", flush=True)

# Paired comparisons (same channel count, different horizon)
print(f"\n{'='*60}", flush=True)
print("PAIRED COMPARISON (same channels, different horizon)", flush=True)
print(f"{'='*60}", flush=True)

pairs = [
    ("5yr_7ep", "10yr_7ep", 21),
    ("5yr_5ep", "20yr_5ep", 15),
]
verdicts = {}
for a_key, b_key, ch in pairs:
    if a_key not in all_rows or b_key not in all_rows:
        continue
    a, b = all_rows[a_key], all_rows[b_key]
    fom_diff = round(a["fom"] - b["fom"], 4)
    mse_diff = round((b["mse"] - a["mse"]) / b["mse"] * 100, 1)
    print(f"\n  {ch}ch: {a_key} (FoM={a['fom']:.4f}) vs {b_key} (FoM={b['fom']:.4f})", flush=True)
    print(f"  FoM diff (5yr − Xyr): {fom_diff:+.4f}", flush=True)
    if abs(fom_diff) < 0.03:
        verdict = "CONFOUND — channel count explains the degradation, not horizon"
    elif fom_diff > 0.03:
        verdict = "HORIZON EFFECT — 5yr CNN is better even with same channels"
    else:
        verdict = "REVERSE — longer horizon CNN is better, unexplained"
    print(f"  Verdict: {verdict}", flush=True)
    verdicts[f"{ch}ch"] = {"fom_diff_5yr_minus_xyr": fom_diff, "verdict": verdict}

results["paired_comparisons"] = verdicts

# Save
out_path = RESULTS_DIR / "cnn_channel_control_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
    f.write("\n")
print(f"\nResults saved to {out_path}", flush=True)
print(f"\n{'='*60}", flush=True)
print("CHANNEL CONTROL EXPERIMENT COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
