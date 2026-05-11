#!/usr/bin/env python3
"""
Pixel-level FoM/MSE evaluation for SimpleCNN (seeds 0 and 42).

Computes CNN 2015 metrics using the SAME eval_mask as validate_2015.py:
    eval_mask = (counts > 0) & (gt_2015 > 0.01)

This makes CNN and ConvLSTM directly comparable in Table 1.

Current Table 1 inconsistency:
    CNN 0.734 — batch-level, all tile pixels (~13.4M, includes rural zeros)
    ConvLSTM 0.504 — pixel-level, gt>0.01 only (1.64M pixels)

After this script:
    CNN seed42 — pixel-level, gt>0.01  (same domain as ConvLSTM)
    CNN seed0  — pixel-level, gt>0.01  (if best_cnn_seed0.pth exists)
    ConvLSTM   — pixel-level, gt>0.01  (from validate_2015.py, unchanged)

Outputs:
    results/metrics/cnn_pixellevel_2015.json
    (used to update Table 1 in main.tex)

Runtime: ~5 min per seed (inference only, no retraining).
"""
import sys, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
GEO  = PROJECT_ROOT / "geotiff_exports"
MOD  = PROJECT_ROOT / "models"
RES  = PROJECT_ROOT / "results" / "metrics"

import numpy as np
import torch
import torch.nn as nn
import rasterio

print("=" * 60, flush=True)
print("CNN PIXEL-LEVEL 2015 EVALUATION", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

TRAIN_EPOCHS     = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH     = 2015
TILE_SIZE        = 128
CHANGE_THRESHOLD = 0.01


# ── SimpleCNN (identical to cnn_2020_holdout.py) ─────────────────────────
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
    def forward(self, x):
        B, T, C, H, W = x.shape
        return torch.sigmoid(self.net(x.reshape(B, T * C, H, W)))


# ── Load val tiles ────────────────────────────────────────────────────────
with open(RES / "val_tile_indices.json") as f:
    val_tiles = [tuple(t) for t in json.load(f)]
print(f"  Val tiles: {len(val_tiles):,}", flush=True)


# ── Load rasters ──────────────────────────────────────────────────────────
print("\n[DATA] Loading rasters...", flush=True)
def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)

bu  = {yr: load_tif(GEO / f"CONUS_builtup_{yr}.tif") for yr in TRAIN_EPOCHS + [TARGET_EPOCH]}
vol = {yr: load_tif(GEO / f"CONUS_volume_{yr}.tif")     for yr in TRAIN_EPOCHS}
pop = {yr: load_tif(GEO / f"CONUS_population_{yr}.tif") for yr in TRAIN_EPOCHS}
ref_shape = bu[TRAIN_EPOCHS[-1]].shape
gc.collect()


# ── Inference helper ──────────────────────────────────────────────────────
def run_inference(model):
    prediction = np.zeros(ref_shape, dtype=np.float64)
    counts      = np.zeros(ref_shape, dtype=np.float64)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for n, (i, j) in enumerate(val_tiles):
            frames = []
            for yr in TRAIN_EPOCHS:
                b = bu[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
                v = vol[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
                p = pop[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
                frames.append(np.stack([b, v, p], axis=0))
            seq = np.stack(frames, axis=0)
            x   = torch.FloatTensor(seq).unsqueeze(0)
            pred_np = model(x).squeeze().numpy().clip(0, 1)
            prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred_np
            counts[i:i+TILE_SIZE, j:j+TILE_SIZE]     += 1
            if (n + 1) % 500 == 0:
                print(f"    {n+1}/{len(val_tiles)} tiles, {time.time()-t0:.0f}s", flush=True)
    mask = counts > 0
    prediction[mask] /= counts[mask]
    return prediction.astype(np.float32), counts


# ── Pixel-level metrics (same as validate_2015.py) ────────────────────────
def compute_metrics(prediction, counts, seed_label):
    gt        = bu[TARGET_EPOCH]
    eval_mask = (counts > 0) & (gt > CHANGE_THRESHOLD)
    n_eval    = int(eval_mask.sum())

    gt_eval   = gt[eval_mask]
    pred_eval = prediction[eval_mask]

    mse  = float(np.mean((gt_eval - pred_eval) ** 2))
    mae  = float(np.mean(np.abs(gt_eval - pred_eval)))

    obs_ch  = (gt          - bu[2010]) > CHANGE_THRESHOLD
    pred_ch = (prediction  - bu[2010]) > CHANGE_THRESHOLD
    B = int((obs_ch[eval_mask] &  pred_ch[eval_mask]).sum())
    A = int((obs_ch[eval_mask] & ~pred_ch[eval_mask]).sum())
    C = int((~obs_ch[eval_mask] & pred_ch[eval_mask]).sum())
    fom = B / (A + B + C) if (A + B + C) > 0 else 0.0

    n_growth = int(obs_ch[eval_mask].sum())
    pct      = 100 * n_growth / n_eval

    print(f"\n  [{seed_label}] n_eval={n_eval:,}  growth={n_growth:,} ({pct:.1f}%)")
    print(f"  [{seed_label}] MSE={mse:.6f}  FoM={fom:.4f}")

    return {"seed": seed_label, "fom_2015": round(fom, 4), "mse_2015": round(mse, 6),
            "mae_2015": round(mae, 6), "n_eval_pixels": n_eval, "n_growth": n_growth,
            "eval_mask": "counts>0 & gt_2015>0.01  (identical to validate_2015.py)"}


# ── Run for each available seed ───────────────────────────────────────────
results = {}

for seed, model_path in [(42, MOD / "best_cnn_3ch.pth"),
                          (0,  MOD / "best_cnn_seed0.pth")]:
    if not model_path.exists():
        print(f"\n  Skipping seed {seed} — {model_path.name} not found", flush=True)
        continue
    print(f"\n[CNN seed {seed}] Loading {model_path.name}...", flush=True)
    model = SimpleCNN(input_channels=24)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    pred, counts = run_inference(model)
    results[str(seed)] = compute_metrics(pred, counts, f"seed{seed}")
    del model, pred, counts; gc.collect()


# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("SUMMARY — pixel-level FoM (eval_mask = gt > 0.01):", flush=True)
foms, mses = [], []
for s, r in results.items():
    print(f"  seed {s}: FoM={r['fom_2015']:.4f}  MSE={r['mse_2015']:.6f}")
    foms.append(r["fom_2015"]); mses.append(r["mse_2015"])

if len(foms) > 1:
    mean_fom = float(np.mean(foms))
    std_fom  = float(np.std(foms))
    print(f"\n  CNN mean FoM: {mean_fom:.4f} ± {std_fom:.4f}")
    print(f"  ConvLSTM FoM (validate_2015.py): 0.5040")
    print(f"  Gap (apples-to-apples): {mean_fom - 0.5040:.4f}")

v15 = json.load(open(RES / "validation_2015_results.json"))
print(f"\n  ConvLSTM (pixel-level, same mask): FoM={v15['convlstm']['fom']:.4f}  n_eval={v15['n_eval_pixels']:,}")


# ── Save ──────────────────────────────────────────────────────────────────
output = {
    "experiment": "cnn_pixellevel_2015",
    "description": (
        "CNN 2015 FoM/MSE recomputed with pixel-level eval_mask = (counts>0) & (gt>0.01), "
        "identical to validate_2015.py. Makes CNN directly comparable to ConvLSTM in Table 1. "
        "Previously CNN used batch-level FoM (all tile pixels, ~13.4M) giving 0.734; "
        "ConvLSTM used pixel-level (1.64M pixels) giving 0.504."
    ),
    "eval_mask": "counts > 0 AND gt_2015 > 0.01",
    "change_threshold": CHANGE_THRESHOLD,
    "prev_builtup_year": 2010,
    "val_tiles": len(val_tiles),
    "convlstm_reference": {
        "fom_2015": round(v15["convlstm"]["fom"], 4),
        "n_eval_pixels": v15["n_eval_pixels"],
        "source": "validation_2015_results.json"
    },
    "per_seed": results,
    "summary": {
        "fom_mean": round(float(np.mean(foms)), 4) if foms else None,
        "fom_std":  round(float(np.std(foms)),  4) if len(foms) > 1 else None,
        "gap_over_convlstm": round(float(np.mean(foms)) - v15["convlstm"]["fom"], 4) if foms else None,
    },
    "timestamp": str(datetime.datetime.now()),
}

out_path = RES / "cnn_pixellevel_2015.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {out_path}", flush=True)
print(f"{'='*60}\n")
