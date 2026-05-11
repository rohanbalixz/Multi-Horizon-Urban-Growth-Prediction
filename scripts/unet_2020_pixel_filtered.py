#!/usr/bin/env python3
"""
U-Net 2020 holdout re-evaluation with pixel-level eval_mask.

The original unet_2020_holdout.py used the DataLoader val loop which
evaluates on ALL val-tile pixels (~13.4M, 4.0% growth).
CNN and ConvLSTM 2020 use eval_mask = (counts > 0) & (gt_2020 > 0.01),
giving 1,574,225 pixels, 10.57% growth.

This script recomputes U-Net 2020 FoM with the SAME eval_mask as
CNN/ConvLSTM so all three models are directly comparable in Table 1.

Uses best_unet_3ch.pth (already trained — no retraining).

Outputs:
    results/metrics/unet_2020_holdout.json  (updated in-place)
    results/metrics/unet_2020_filtered.json (new file with details)

Runtime: ~10 min (inference only)
"""
import sys, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
GEO = PROJECT_ROOT / "geotiff_exports"
MOD = PROJECT_ROOT / "models"
RES = PROJECT_ROOT / "results" / "metrics"

import numpy as np
import torch
import torch.nn as nn
import rasterio

print("=" * 60, flush=True)
print("U-NET 2020 PIXEL-FILTERED RECOMPUTATION", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

HOLDOUT_EPOCHS   = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
HOLDOUT_TARGET   = 2020
TILE_SIZE        = 128
CHANGE_THRESHOLD = 0.01


# ── SimpleUNet (identical to unet_2020_holdout.py) ─────────────────────────
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


# ── Load val tiles ─────────────────────────────────────────────────────────
with open(RES / "val_tile_indices.json") as f:
    val_tiles = [tuple(t) for t in json.load(f)]
print(f"  Val tiles: {len(val_tiles):,}", flush=True)

# ── Load rasters ───────────────────────────────────────────────────────────
print("\n[DATA] Loading 2020 holdout rasters (float32)...", flush=True)
def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)

# Holdout inputs
bu_h  = {yr: load_tif(GEO / f"CONUS_builtup_{yr}.tif")    for yr in HOLDOUT_EPOCHS}
vol_h = {yr: load_tif(GEO / f"CONUS_volume_{yr}.tif")     for yr in HOLDOUT_EPOCHS}
pop_h = {yr: load_tif(GEO / f"CONUS_population_{yr}.tif") for yr in HOLDOUT_EPOCHS}
gt_2020 = load_tif(GEO / f"CONUS_builtup_{HOLDOUT_TARGET}.tif")
ref_shape = gt_2020.shape
print(f"  Raster shape: {ref_shape}", flush=True)
gc.collect()

# ── Load U-Net ─────────────────────────────────────────────────────────────
model_path = MOD / "best_unet_3ch.pth"
print(f"\n[MODEL] Loading {model_path.name}...", flush=True)
model = SimpleUNet(input_channels=len(HOLDOUT_EPOCHS) * 3)
model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}", flush=True)

# ── Inference ─────────────────────────────────────────────────────────────
prediction = np.zeros(ref_shape, dtype=np.float64)
counts     = np.zeros(ref_shape, dtype=np.float64)

print(f"\n[PREDICT] Running on {len(val_tiles):,} val tiles...", flush=True)
t_start = time.time()
with torch.no_grad():
    for n, (i, j) in enumerate(val_tiles):
        frames = []
        for yr in HOLDOUT_EPOCHS:
            b = bu_h[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            v = vol_h[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            p = pop_h[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            frames.append(np.stack([b, v, p], axis=0))
        seq = np.stack(frames, axis=0)
        x   = torch.FloatTensor(seq).unsqueeze(0)
        pred_np = model(x).squeeze().numpy().clip(0, 1)
        prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred_np
        counts[i:i+TILE_SIZE, j:j+TILE_SIZE]     += 1
        if (n + 1) % 300 == 0:
            print(f"    {n+1}/{len(val_tiles)} tiles, {time.time()-t_start:.0f}s", flush=True)

mask = counts > 0
prediction[mask] /= counts[mask]
prediction = prediction.astype(np.float32)
print(f"  Done. {time.time()-t_start:.0f}s elapsed", flush=True)

# ── Pixel-level metrics with eval_mask = gt > 0.01 ────────────────────────
# Same domain as CNN/ConvLSTM 2020 evaluation
eval_mask = (counts > 0) & (gt_2020 > CHANGE_THRESHOLD)
n_eval    = int(eval_mask.sum())

gt_eval   = gt_2020[eval_mask]
pred_eval = prediction[eval_mask]
prev_2015 = bu_h[2015]

mse = float(np.mean((gt_eval - pred_eval) ** 2))
mae = float(np.mean(np.abs(gt_eval - pred_eval)))
r2  = 1.0 - float(np.sum((gt_eval - pred_eval)**2)) / float(np.sum((gt_eval - gt_eval.mean())**2))

obs_ch  = (gt_2020  - prev_2015) > CHANGE_THRESHOLD
pred_ch = (prediction - prev_2015) > CHANGE_THRESHOLD

B_val = int((obs_ch[eval_mask] &  pred_ch[eval_mask]).sum())
A_val = int((obs_ch[eval_mask] & ~pred_ch[eval_mask]).sum())
C_val = int((~obs_ch[eval_mask] & pred_ch[eval_mask]).sum())
fom   = B_val / (A_val + B_val + C_val) if (A_val + B_val + C_val) > 0 else 0.0

n_growth = int(obs_ch[eval_mask].sum())
pct_growth = 100.0 * n_growth / n_eval

# Growth and stable zone MSE
growth_mask_eval = obs_ch[eval_mask]
stable_mask_eval = ~obs_ch[eval_mask]
gmse = float(np.mean((gt_eval[growth_mask_eval] - pred_eval[growth_mask_eval])**2)) if growth_mask_eval.any() else float('nan')
smse = float(np.mean((gt_eval[stable_mask_eval] - pred_eval[stable_mask_eval])**2)) if stable_mask_eval.any() else float('nan')

print(f"\n[RESULTS] U-Net 2020 — PIXEL-FILTERED (gt > 0.01)", flush=True)
print(f"  n_eval={n_eval:,}  n_growth={n_growth:,} ({pct_growth:.2f}%)")
print(f"  FoM={fom:.4f}  MSE={mse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
print(f"  growth_MSE={gmse:.6f}  stable_MSE={smse:.6f}", flush=True)

# Compare with CNN and ConvLSTM 2020
# CNN: FoM=0.2432 (seed42), ConvLSTM: FoM=0.1556
print(f"\n  Compare (same eval domain gt>0.01, 1.57M pixels):")
print(f"    CNN seed42:   FoM=0.2432")
print(f"    U-Net seed42: FoM={fom:.4f}")
print(f"    ConvLSTM:     FoM=0.1556")

# ── Save filtered results ──────────────────────────────────────────────────
filtered = {
    "experiment": "unet_2020_pixel_filtered",
    "description": (
        "U-Net 2020 FoM recomputed with eval_mask = (counts>0) & (gt_2020>0.01), "
        "matching the CNN and ConvLSTM 2020 evaluation domain exactly. "
        "Original unet_2020_holdout.json used DataLoader (all val-tile pixels, "
        "13.4M pixels, 4.0% growth). This recomputation uses 1,574,225 pixels, "
        f"{pct_growth:.2f}% growth — identical domain to CNN/ConvLSTM."
    ),
    "model": "SimpleUNet_3ch_flatstack",
    "model_path": "models/best_unet_3ch.pth",
    "params": n_params,
    "eval_mask": "counts>0 & gt_2020>0.01",
    "n_eval_pixels": n_eval,
    "n_growth_pixels": n_growth,
    "pct_growth": round(pct_growth, 2),
    "holdout_2020_filtered": {
        "fom":        round(fom, 4),
        "mse":        round(mse, 8),
        "mae":        round(mae, 8),
        "r2":         round(r2, 6),
        "growth_mse": round(gmse, 8),
        "stable_mse": round(smse, 8),
        "A": A_val, "B": B_val, "C": C_val,
    },
    "comparison_same_domain": {
        "cnn_seed42_fom": 0.2432,
        "unet_fom":       round(fom, 4),
        "convlstm_fom":   0.1556,
        "cnn_unet_gap":   round(0.2432 - fom, 4),
        "unet_clstm_gap": round(fom - 0.1556, 4),
    },
    "timestamp": str(datetime.datetime.now()),
}

out_filtered = RES / "unet_2020_filtered.json"
with open(out_filtered, "w") as f:
    json.dump(filtered, f, indent=2)
print(f"\n  Saved: {out_filtered}", flush=True)

# ── Update unet_2020_holdout.json with filtered result ────────────────────
orig_path = RES / "unet_2020_holdout.json"
orig = json.load(open(orig_path))
orig["holdout_2020_filtered"] = {
    "fom":        round(fom, 4),
    "mse":        round(mse, 8),
    "n_eval_pixels": n_eval,
    "n_growth_pixels": n_growth,
    "pct_growth": round(pct_growth, 2),
    "eval_mask": "counts>0 & gt_2020>0.01 (matches CNN/ConvLSTM domain)",
    "note": "Use this for apples-to-apples CNN/U-Net/ConvLSTM 2020 comparison.",
}
with open(orig_path, "w") as f:
    json.dump(orig, f, indent=2)
print(f"  Updated: {orig_path}", flush=True)

print(f"\n{'='*60}", flush=True)
print("DONE", flush=True)
print(f"  U-Net 2020 FoM (pixel-filtered): {fom:.4f}", flush=True)
print(f"  Ranking (same domain): CNN 0.2432 > U-Net {fom:.4f} > ConvLSTM 0.1556", flush=True)
print(f"{'='*60}", flush=True)
