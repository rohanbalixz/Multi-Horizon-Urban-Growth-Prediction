#!/usr/bin/env python3
"""
Re-evaluate ConvLSTM seed-42 with float32 rasters so n_eval matches
all other pixel-level evals (1,542,747 instead of 1,640,513).

Root cause: validate_2015.py loads rasters without .astype(np.float32),
so floating-point precision near the 0.01 threshold differs.
This script uses identical dtype and eval_mask as eval_convlstm_seed0.py.

Updates:
  - results/metrics/convlstm_multiseed_results.json (seed-42 pixlevel fields)
  - results/metrics/validation_2015_results.json (fom, n_eval_pixels)
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
print("ConvLSTM SEED-42 PIXEL-LEVEL RE-EVALUATION (float32 fix)", flush=True)
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

# ── Load rasters (float32 — same as all other pixel-level evals) ──────────
print("\n[DATA] Loading rasters (float32)...", flush=True)

def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)

builtup    = {yr: load_tif(GEO / f"CONUS_builtup_{yr}.tif")    for yr in TRAIN_EPOCHS + [TARGET_EPOCH]}
volume     = {yr: load_tif(GEO / f"CONUS_volume_{yr}.tif")     for yr in TRAIN_EPOCHS}
population = {yr: load_tif(GEO / f"CONUS_population_{yr}.tif") for yr in TRAIN_EPOCHS}
ref_shape = builtup[2010].shape
print(f"  Raster shape: {ref_shape}", flush=True)
gc.collect()

# ── Load ConvLSTM seed-42 ─────────────────────────────────────────────────
model_path = MOD / "best_3ch_mc_model.pth"
print(f"\n[MODEL] Loading {model_path.name}...", flush=True)
model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)
ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Inference ─────────────────────────────────────────────────────────────
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

# ── Pixel-level metrics ───────────────────────────────────────────────────
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

print(f"\n  [ConvLSTM seed-42, float32] n_eval={n_eval:,}  growth={n_growth:,} ({pct:.1f}%)")
print(f"  [ConvLSTM seed-42, float32] MSE={mse:.6f}  FoM={fom:.4f}", flush=True)

# ── Update convlstm_multiseed_results.json ────────────────────────────────
ms_path = RES / "convlstm_multiseed_results.json"
ms = json.load(open(ms_path))

ms["per_seed"]["42"]["fom_2015_pixlevel"] = round(fom, 4)
ms["per_seed"]["42"]["mse_2015_pixlevel"] = round(mse, 6)
ms["per_seed"]["42"]["n_eval_pixels"]     = n_eval
ms["per_seed"]["42"]["eval_mask"]         = "counts>0 & gt_2015>0.01 (float32)"

fom_seed0 = ms["per_seed"]["0"]["fom_2015_pixlevel"]
mse_seed0 = ms["per_seed"]["0"]["mse_2015_pixlevel"]

foms = [fom_seed0, fom]
mses = [mse_seed0, mse]
ms["summary_pixlevel"] = {
    "fom_2015_mean": round(float(np.mean(foms)), 4),
    "fom_2015_std":  round(float(np.std(foms)),  4),
    "mse_2015_mean": round(float(np.mean(mses)), 6),
    "mse_2015_std":  round(float(np.std(mses)),  6),
    "n_eval_pixels": n_eval,
    "eval_mask":     "counts>0 & gt_2015>0.01 (float32, all evals consistent)",
}
ms["timestamp"] = str(datetime.datetime.now())

with open(ms_path, "w") as f:
    json.dump(ms, f, indent=2)
print(f"  Updated: {ms_path}", flush=True)

# ── Also update validation_2015_results.json fom/n_eval ──────────────────
v15_path = RES / "validation_2015_results.json"
v15 = json.load(open(v15_path))
old_fom   = v15["convlstm"]["fom"]
old_n     = v15["n_eval_pixels"]
v15["convlstm"]["fom"]  = round(fom, 4)
v15["convlstm"]["mse"]  = round(mse, 6)
v15["convlstm"]["mae"]  = round(mae, 6)
v15["n_eval_pixels"]    = n_eval
v15["eval_note"]        = "Recomputed 2026-04-18 with float32 rasters; consistent with all other pixel-level evals."
with open(v15_path, "w") as f:
    json.dump(v15, f, indent=2)
print(f"  Updated: {v15_path}", flush=True)
print(f"  FoM: {old_fom:.4f} -> {fom:.4f}  |  n_eval: {old_n:,} -> {n_eval:,}", flush=True)

# ── Final summary ─────────────────────────────────────────────────────────
cnn_pl = json.load(open(RES / "cnn_pixellevel_2015.json"))

print(f"\n{'='*60}", flush=True)
print("FINAL PIXEL-LEVEL FoM — ALL EVALS CONSISTENT (float32)", flush=True)
print(f"  ConvLSTM seed-0:  FoM={fom_seed0:.4f}  n_eval={ms['per_seed']['0']['n_eval_pixels']:,}", flush=True)
print(f"  ConvLSTM seed-42: FoM={fom:.4f}  n_eval={n_eval:,}", flush=True)
print(f"  ConvLSTM mean:    FoM={ms['summary_pixlevel']['fom_2015_mean']:.4f} ± {ms['summary_pixlevel']['fom_2015_std']:.4f}", flush=True)
print(f"  CNN mean:         FoM={cnn_pl['summary']['fom_mean']:.4f} ± {cnn_pl['summary']['fom_std']:.4f}", flush=True)
gap = cnn_pl['summary']['fom_mean'] - ms['summary_pixlevel']['fom_2015_mean']
print(f"  Gap (CNN-ConvLSTM): {gap:.4f}", flush=True)
print(f"{'='*60}\n", flush=True)
