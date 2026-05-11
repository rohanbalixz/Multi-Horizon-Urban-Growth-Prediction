#!/usr/bin/env python3
"""
Pixel-level FoM evaluation for ConvLSTM seed-0.

Uses IDENTICAL eval approach as validate_2015.py:
    eval_mask = (counts > 0) & (gt_2015 > 0.01)

Then updates convlstm_multiseed_results.json with pixel-level FoMs
so CNN and ConvLSTM are directly comparable in Table 1.
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
import rasterio

from src.models.convlstm import ConvLSTM

print("=" * 60, flush=True)
print("ConvLSTM SEED-0 PIXEL-LEVEL 2015 EVALUATION", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

TRAIN_EPOCHS     = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH     = 2015
TILE_SIZE        = 128
CHANGE_THRESHOLD = 0.01
MC_DROPOUT       = 0.1


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


# ── Load val tiles ────────────────────────────────────────────────────────
with open(RES / "val_tile_indices.json") as f:
    val_tiles = [tuple(t) for t in json.load(f)]
print(f"  Val tiles: {len(val_tiles):,}", flush=True)

# ── Load rasters ──────────────────────────────────────────────────────────
print("\n[DATA] Loading rasters...", flush=True)

def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1)

builtup    = {yr: load_tif(GEO / f"CONUS_builtup_{yr}.tif")    for yr in TRAIN_EPOCHS + [TARGET_EPOCH]}
volume     = {yr: load_tif(GEO / f"CONUS_volume_{yr}.tif")     for yr in TRAIN_EPOCHS}
population = {yr: load_tif(GEO / f"CONUS_population_{yr}.tif") for yr in TRAIN_EPOCHS}
ref_shape = builtup[2010].shape
print(f"  Raster shape: {ref_shape}", flush=True)
gc.collect()

# ── Load ConvLSTM seed-0 ──────────────────────────────────────────────────
model_path = MOD / "best_3ch_mc_model_seed0.pth"
print(f"\n[MODEL] Loading {model_path.name}...", flush=True)
model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Pixel-level inference (identical to validate_2015.py) ─────────────────
prediction = np.zeros(ref_shape, dtype=np.float64)
counts     = np.zeros(ref_shape, dtype=np.float64)

print(f"\n[PREDICT] Running on {len(val_tiles):,} held-out tiles...", flush=True)
t_start = time.time()

with torch.no_grad():
    for n, (i, j) in enumerate(val_tiles):
        frames = []
        for yr in TRAIN_EPOCHS:
            bu  = builtup[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            vol = volume[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            pop = population[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
            frames.append(np.stack([bu, vol, pop], axis=0))
        seq = np.stack(frames, axis=0)
        x   = torch.FloatTensor(seq).unsqueeze(0)
        pred = predict_future_step(model, x)
        pred_np = pred.squeeze().numpy().clip(0, 1)
        prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred_np
        counts[i:i+TILE_SIZE, j:j+TILE_SIZE]     += 1
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/{len(val_tiles)} tiles, {time.time()-t_start:.0f}s", flush=True)

mask = counts > 0
prediction[mask] /= counts[mask]
prediction = prediction.astype(np.float32)
print(f"  Done. {time.time()-t_start:.0f}s elapsed", flush=True)

# ── Pixel-level metrics (same eval_mask as validate_2015.py) ──────────────
gt        = builtup[TARGET_EPOCH]
eval_mask = (counts > 0) & (gt > CHANGE_THRESHOLD)
n_eval    = int(eval_mask.sum())

gt_eval   = gt[eval_mask]
pred_eval = prediction[eval_mask]

mse = float(np.mean((gt_eval - pred_eval) ** 2))
mae = float(np.mean(np.abs(gt_eval - pred_eval)))

obs_ch  = (gt - builtup[2010]) > CHANGE_THRESHOLD
pred_ch = (prediction - builtup[2010]) > CHANGE_THRESHOLD
B = int((obs_ch[eval_mask] &  pred_ch[eval_mask]).sum())
A = int((obs_ch[eval_mask] & ~pred_ch[eval_mask]).sum())
C = int((~obs_ch[eval_mask] & pred_ch[eval_mask]).sum())
fom = B / (A + B + C) if (A + B + C) > 0 else 0.0

n_growth = int(obs_ch[eval_mask].sum())
pct = 100 * n_growth / n_eval

print(f"\n  [ConvLSTM seed-0] n_eval={n_eval:,}  growth={n_growth:,} ({pct:.1f}%)")
print(f"  [ConvLSTM seed-0] MSE={mse:.6f}  FoM={fom:.4f}", flush=True)

# ── Update convlstm_multiseed_results.json ────────────────────────────────
ms_path = RES / "convlstm_multiseed_results.json"
ms = json.load(open(ms_path))

# Load seed-42 pixel-level FoM from validation_2015_results.json
v15 = json.load(open(RES / "validation_2015_results.json"))
fom_seed42_pixlevel = v15["convlstm"]["fom"]   # 0.5040
mse_seed42_pixlevel = v15["convlstm"]["mse"]   # pixel-level

# Update per_seed with pixel-level values
ms["per_seed"]["0"]["fom_2015_pixlevel"] = round(fom, 4)
ms["per_seed"]["0"]["mse_2015_pixlevel"] = round(mse, 6)
ms["per_seed"]["0"]["n_eval_pixels"]     = n_eval
ms["per_seed"]["0"]["eval_mask"]         = "counts>0 & gt_2015>0.01"
ms["per_seed"]["42"]["fom_2015_pixlevel"] = round(fom_seed42_pixlevel, 4)
ms["per_seed"]["42"]["mse_2015_pixlevel"] = round(mse_seed42_pixlevel, 6)
ms["per_seed"]["42"]["n_eval_pixels"]     = v15["n_eval_pixels"]
ms["per_seed"]["42"]["eval_mask"]         = "counts>0 & gt_2015>0.01"

# Add pixel-level summary
foms_pixlevel = [fom, fom_seed42_pixlevel]
mses_pixlevel = [mse, mse_seed42_pixlevel]
ms["summary_pixlevel"] = {
    "fom_2015_mean": round(float(np.mean(foms_pixlevel)), 4),
    "fom_2015_std":  round(float(np.std(foms_pixlevel)),  4),
    "mse_2015_mean": round(float(np.mean(mses_pixlevel)), 6),
    "mse_2015_std":  round(float(np.std(mses_pixlevel)),  6),
    "eval_mask":     "counts>0 & gt_2015>0.01",
    "note": "Pixel-level eval identical to validate_2015.py; directly comparable to cnn_pixellevel_2015.json",
}
ms["timestamp"] = str(datetime.datetime.now())

with open(ms_path, "w") as f:
    json.dump(ms, f, indent=2)
print(f"\n  Updated: {ms_path}", flush=True)

# ── Final summary ─────────────────────────────────────────────────────────
cnn_pl = json.load(open(RES / "cnn_pixellevel_2015.json"))
cnn_mean_fom = cnn_pl["summary"]["fom_mean"]
cnn_std_fom  = cnn_pl["summary"]["fom_std"]

convlstm_mean_fom = round(float(np.mean(foms_pixlevel)), 4)
convlstm_std_fom  = round(float(np.std(foms_pixlevel)),  4)

print(f"\n{'='*60}", flush=True)
print("APPLES-TO-APPLES PIXEL-LEVEL FoM SUMMARY", flush=True)
print(f"  eval_mask = (counts>0) & (gt_2015>0.01)", flush=True)
print(f"  ConvLSTM seed-0:  FoM={fom:.4f}", flush=True)
print(f"  ConvLSTM seed-42: FoM={fom_seed42_pixlevel:.4f}", flush=True)
print(f"  ConvLSTM mean:    FoM={convlstm_mean_fom:.4f} ± {convlstm_std_fom:.4f}", flush=True)
print(f"  CNN mean:         FoM={cnn_mean_fom:.4f} ± {cnn_std_fom:.4f}", flush=True)
print(f"  Gap (CNN-ConvLSTM): {cnn_mean_fom - convlstm_mean_fom:.4f}", flush=True)
print(f"{'='*60}\n", flush=True)
