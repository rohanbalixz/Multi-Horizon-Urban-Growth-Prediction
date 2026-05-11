#!/usr/bin/env python3
"""
Lagos full transfer evaluation.

Adds to the original eval_lagos_transfer.py:
  - Collects raw per-pixel (pred, gt, prev_bu) arrays
  - Threshold sweep for CNN: t in {0.005, 0.01, 0.02, 0.05, 0.10}
  - Linear extrapolation baseline with the same sweep
  - Trivial predict-all-growth baseline (analytical)
  - Optimal-threshold results for each model
"""

import gc
import json
import datetime
import warnings
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from pyproj import Transformer

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
MOD_DIR  = PROJECT_ROOT / "models"
RES_DIR  = PROJECT_ROOT / "results" / "metrics"

TRAIN_EPOCHS     = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH     = 2015
TILE_SIZE        = 128
RESOLUTION       = 250
EVAL_MASK_THRESH = 0.01   # gt > this to enter eval domain
THRESHOLDS       = [0.005, 0.01, 0.02, 0.05, 0.10]

MOLLWEIDE_PROJ = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
LAGOS_CRS      = "EPSG:32631"
CONUS_CRS      = "EPSG:5070"
CONUS_BOUNDS   = {"west": -125.0, "east": -66.0, "south": 24.0, "north": 49.5}
LAGOS_BOUNDS   = {"west": 2.7, "east": 4.3, "south": 5.8, "north": 7.3}

print("=" * 60)
print("LAGOS FULL TRANSFER EVALUATION (threshold sweep + baselines)")
print(f"Started: {datetime.datetime.now()}")
print("=" * 60, flush=True)


# ── Models ─────────────────────────────────────────────────────────────────

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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=kernel_size // 2,
        )

    def forward(self, x, states):
        h, c = states
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64,
                 num_layers=2, mc_dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layers.append(ConvLSTMCell(
                input_channels if i == 0 else hidden_channels,
                hidden_channels,
            ))
        self.convlstm_layers = nn.ModuleList(layers)
        self.mc_dropouts = nn.ModuleList(
            [nn.Dropout2d(p=mc_dropout) for _ in range(num_layers)]
        )
        self.skip_proj = nn.Conv2d(hidden_channels * num_layers,
                                   hidden_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        states = [
            (torch.zeros(B, self.hidden_channels, H, W),
             torch.zeros(B, self.hidden_channels, H, W))
            for _ in range(self.num_layers)
        ]
        for t in range(T):
            x_t = x[:, t]
            hiddens = []
            for i, layer in enumerate(self.convlstm_layers):
                h, c = states[i]
                h, c = layer(x_t, (h, c))
                h = self.mc_dropouts[i](h)
                states[i] = (h, c)
                x_t = h
                hiddens.append(h)
        fused = self.skip_proj(torch.cat(hiddens, dim=1))
        return self.decoder(fused)   # (B, 1, H, W) from last timestep


# ── I/O helpers (identical to eval_lagos_transfer.py) ─────────────────────

def wgs84_to_moll(lon, lat):
    t = Transformer.from_crs("EPSG:4326", MOLLWEIDE_PROJ, always_xy=True)
    x, y = t.transform(lon, lat)
    return x, y


def read_window_4326(tif_path, bounds, dst_crs, resolution,
                     log_transform=False, is_density=False):
    with rasterio.open(str(tif_path)) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height,
            left=bounds["west"], bottom=bounds["south"],
            right=bounds["east"], top=bounds["north"],
            resolution=resolution,
        )
        dst_data = np.zeros((dst_h, dst_w), dtype=np.float32)
        win = rasterio.windows.from_bounds(
            bounds["west"], bounds["south"],
            bounds["east"], bounds["north"],
            transform=src.transform,
        )
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        win_data = src.read(1, window=win).astype(np.float32)
        win_transform = src.window_transform(win)
        reproject(
            source=win_data, destination=dst_data,
            src_transform=win_transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=0.0,
        )
    dst_data = np.nan_to_num(dst_data, nan=0.0, posinf=0.0, neginf=0.0)
    if is_density:
        cell_area = resolution * resolution
        mid_lat = (bounds["south"] + bounds["north"]) / 2.0
        src_res_deg = 3.0 / 3600.0
        src_cell_m2 = (src_res_deg * 111320 * np.cos(np.radians(mid_lat))) \
                    * (src_res_deg * 110540)
        dst_data = np.clip(dst_data / src_cell_m2, 0.0, 1.0)
    if log_transform:
        dst_data[dst_data > 1e8] = 0.0
        dst_data = np.log1p(dst_data).astype(np.float32)
    return dst_data


def _moll_window(bounds_wgs84):
    corners_lon = [bounds_wgs84["west"], bounds_wgs84["east"]] * 2
    corners_lat = [bounds_wgs84["south"]] * 2 + [bounds_wgs84["north"]] * 2
    xs, ys = [], []
    for lon, lat in zip(corners_lon, corners_lat):
        x, y = wgs84_to_moll(lon, lat)
        xs.append(x); ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def _moll_raw_window(tif_path, moll_west, moll_east, moll_south, moll_north):
    MOLL_XMIN, MOLL_YMAX = -18041000.0, 9000000.0
    MOLL_RES = 100.0
    with rasterio.open(str(tif_path)) as src:
        col_off = max(0, int((moll_west  - MOLL_XMIN) / MOLL_RES))
        row_off = max(0, int((MOLL_YMAX  - moll_north) / MOLL_RES))
        col_end = min(src.width,  int((moll_east  - MOLL_XMIN) / MOLL_RES) + 2)
        row_end = min(src.height, int((MOLL_YMAX  - moll_south) / MOLL_RES) + 2)
        win = rasterio.windows.Window(col_off, row_off,
                                      col_end - col_off, row_end - row_off)
        data = src.read(1, window=win).astype(np.float32)
    data = np.where(np.isfinite(data), data, 0.0)
    data = np.clip(data, 0.0, 1e7)
    return data


def moll_log1p_max(tif_path, bounds_wgs84):
    mw, me, ms, mn = _moll_window(bounds_wgs84)
    data = _moll_raw_window(tif_path, mw, me, ms, mn)
    return float(np.log1p(data).max())


def read_window_moll(tif_path, bounds, dst_crs, resolution, log_transform=False):
    moll_west, moll_east, moll_south, moll_north = _moll_window(bounds)
    win_data = _moll_raw_window(tif_path, moll_west, moll_east, moll_south, moll_north)
    src_transform = rasterio.transform.from_bounds(
        moll_west, moll_south, moll_east, moll_north,
        win_data.shape[1], win_data.shape[0],
    )
    src_crs = CRS.from_proj4(MOLLWEIDE_PROJ)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, win_data.shape[1], win_data.shape[0],
        left=moll_west, bottom=moll_south, right=moll_east, top=moll_north,
        resolution=resolution,
    )
    dst_data = np.zeros((dst_h, dst_w), dtype=np.float32)
    reproject(
        source=win_data, destination=dst_data,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=0.0, dst_nodata=0.0,
    )
    dst_data = np.clip(np.nan_to_num(dst_data, nan=0.0), 0.0, 1e7)
    if log_transform:
        dst_data = np.log1p(dst_data).astype(np.float32)
    return dst_data


def find_builtup_src(yr):
    p4 = DATA_DIR / f"GHS_BUILT_S_E{yr}_GLOBE_R2023A_4326_3ss_V1_0.tif"
    pm = DATA_DIR / f"GHS_BUILT_S_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
    if p4.exists(): return p4, True
    if pm.exists(): return pm, False
    raise FileNotFoundError(f"No built-up source for {yr}")


# ── FoM helper ─────────────────────────────────────────────────────────────

def fom_metrics(gt_flat, prev_flat, pred_flat, mask_flat, t):
    obs_ch  = (gt_flat   - prev_flat) > t
    pred_ch = (pred_flat - prev_flat) > t
    obs_ch  &= mask_flat
    B = int((obs_ch  &  pred_ch).sum())
    A = int((obs_ch  & ~pred_ch).sum())
    C = int((~obs_ch &  pred_ch & mask_flat).sum())
    denom = A + B + C
    fom  = B / denom if denom > 0 else 0.0
    prec = B / (B + C) if (B + C) > 0 else 0.0
    rec  = B / (A + B) if (A + B) > 0 else 0.0
    return {"fom": round(fom, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "A": A, "B": B, "C": C}


# ── Phase 1: normalization constants ───────────────────────────────────────

print("\n[NORM] Computing Lagos log1p normalization constants...")
vol_norm_max = pop_norm_max = 0.0
for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
    vol_src = DATA_DIR / f"GHS_BUILT_V_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
    pop_src = DATA_DIR / f"GHS_POP_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
    vol_norm_max = max(vol_norm_max, moll_log1p_max(vol_src, LAGOS_BOUNDS))
    pop_norm_max = max(pop_norm_max, moll_log1p_max(pop_src, LAGOS_BOUNDS))
    print(f"  {yr} done", flush=True)
print(f"  vol_norm_max={vol_norm_max:.6f}  pop_norm_max={pop_norm_max:.6f}")


# ── Phase 2: preprocess ────────────────────────────────────────────────────

print("\n[PREPROCESS] Loading Lagos GHSL data...")
lagos_bu  = {}
lagos_vol = {}
lagos_pop = {}

for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
    bu_src, is_4326 = find_builtup_src(yr)
    if is_4326:
        bu_raw = read_window_4326(bu_src, LAGOS_BOUNDS, LAGOS_CRS, RESOLUTION,
                                  is_density=True)
    else:
        bu_raw = read_window_moll(bu_src, LAGOS_BOUNDS, LAGOS_CRS, RESOLUTION)
        src_cell = 100.0 * 100.0   # Mollweide source cell area (m²)
        bu_raw = np.clip(bu_raw / src_cell, 0.0, 1.0)
    lagos_bu[yr] = bu_raw
    if yr in TRAIN_EPOCHS:
        vol_src = DATA_DIR / f"GHS_BUILT_V_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
        vol_arr = read_window_moll(vol_src, LAGOS_BOUNDS, LAGOS_CRS, RESOLUTION,
                                   log_transform=True)
        lagos_vol[yr] = np.clip(vol_arr / vol_norm_max, 0.0, None).astype(np.float32)
        pop_src = DATA_DIR / f"GHS_POP_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
        pop_arr = read_window_moll(pop_src, LAGOS_BOUNDS, LAGOS_CRS, RESOLUTION,
                                   log_transform=True)
        lagos_pop[yr] = np.clip(pop_arr / pop_norm_max, 0.0, None).astype(np.float32)
        del vol_arr, pop_arr
    gc.collect()

ref_h, ref_w = lagos_bu[TARGET_EPOCH].shape
print(f"  Grid: {ref_h}x{ref_w}  ({ref_h*RESOLUTION/1000:.0f}×{ref_w*RESOLUTION/1000:.0f} km)")


# ── Phase 3: model ─────────────────────────────────────────────────────────

print("\n[MODEL] Loading SimpleCNN (seed 42)...")
cnn = SimpleCNN(input_channels=24)
ckpt = torch.load(str(MOD_DIR / "best_cnn_3ch.pth"),
                  map_location="cpu", weights_only=True)
cnn.load_state_dict(ckpt.get("model_state_dict", ckpt))
cnn.eval()

print("[MODEL] Loading ConvLSTM (seed 42)...")
convlstm = ConvLSTMModel(input_channels=3, hidden_channels=64,
                         num_layers=2, mc_dropout=0.1)
ckpt = torch.load(str(MOD_DIR / "best_3ch_mc_model.pth"),
                  map_location="cpu", weights_only=True)
convlstm.load_state_dict(ckpt.get("model_state_dict", ckpt))
convlstm.eval()


# ── Phase 4: tile inference — collect raw arrays ───────────────────────────

print("\n[INFERENCE] Collecting raw predictions over all tiles...")

all_gt        = []
all_prev      = []
all_pred_cnn  = []
all_pred_clstm = []
all_bu_epochs = []   # shape: (n_pixels, 8) for linear extrapolation

n_tiles_run = n_tiles_skip = 0
gt_map  = lagos_bu[TARGET_EPOCH]
prev_bu = lagos_bu[TRAIN_EPOCHS[-1]]

for ti in range(0, ref_h - TILE_SIZE + 1, TILE_SIZE):
    for tj in range(0, ref_w - TILE_SIZE + 1, TILE_SIZE):
        gt_tile   = gt_map[ti:ti+TILE_SIZE, tj:tj+TILE_SIZE]
        prev_tile = prev_bu[ti:ti+TILE_SIZE, tj:tj+TILE_SIZE]
        mask      = gt_tile > EVAL_MASK_THRESH

        if mask.sum() < 100:
            n_tiles_skip += 1
            continue

        frames   = []
        bu_stack = []
        for yr in TRAIN_EPOCHS:
            ch = np.stack([
                lagos_bu[yr][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE],
                lagos_vol[yr][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE],
                lagos_pop[yr][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE],
            ], axis=0)
            frames.append(ch)
            bu_stack.append(lagos_bu[yr][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE])

        x = torch.from_numpy(np.stack(frames, axis=0)[None].astype(np.float32))
        with torch.no_grad():
            pred_cnn   = cnn(x).squeeze().numpy()
            pred_clstm = convlstm(x).squeeze().numpy()

        idx = mask.ravel()
        all_gt.append(gt_tile.ravel()[idx])
        all_prev.append(prev_tile.ravel()[idx])
        all_pred_cnn.append(pred_cnn.ravel()[idx])
        all_pred_clstm.append(pred_clstm.ravel()[idx])

        bu_mat = np.stack([b.ravel()[idx] for b in bu_stack], axis=1)
        all_bu_epochs.append(bu_mat)
        n_tiles_run += 1

print(f"  {n_tiles_run} tiles processed, {n_tiles_skip} skipped")

gt_all     = np.concatenate(all_gt)
prev_all   = np.concatenate(all_prev)
pred_cnn   = np.concatenate(all_pred_cnn)
pred_clstm = np.concatenate(all_pred_clstm)
mask_all   = np.ones(len(gt_all), dtype=bool)
bu_mat     = np.concatenate(all_bu_epochs, axis=0)  # (N, 8)

n_px     = len(gt_all)
n_growth = int(((gt_all - prev_all) > EVAL_MASK_THRESH).sum())
pct_grow = 100.0 * n_growth / n_px
print(f"  Total eval pixels: {n_px:,}  growth: {n_growth:,} ({pct_grow:.1f}%)")


# ── Phase 5: linear extrapolation baseline ────────────────────────────────

print("\n[LINEXTRAP] Computing linear extrapolation baseline...")
# t_vals: epoch indices 0..7 for years 1975..2010
t_vals = np.arange(8, dtype=np.float32)   # 0..7
# Fit y = a*t + b via normal equations for each pixel
T  = len(t_vals)
St  = t_vals.sum()
St2 = (t_vals**2).sum()
Sy  = bu_mat.sum(axis=1)
Sty = (bu_mat * t_vals[None, :]).sum(axis=1)
denom_reg = T * St2 - St ** 2
a_coef = (T * Sty - St * Sy) / denom_reg
b_coef = (Sy - a_coef * St) / T
# Project to t=8 (2015)
pred_lin = np.clip(a_coef * 8 + b_coef, 0.0, 1.0).astype(np.float32)
print(f"  Linear pred range: [{pred_lin.min():.3f}, {pred_lin.max():.3f}]")


# ── Phase 6: threshold sweep ───────────────────────────────────────────────

print("\n[SWEEP] Threshold sweep...")

cnn_sweep   = {}
clstm_sweep = {}
lin_sweep   = {}

for t in THRESHOLDS:
    cnn_sweep[t]   = fom_metrics(gt_all, prev_all, pred_cnn,   mask_all, t)
    clstm_sweep[t] = fom_metrics(gt_all, prev_all, pred_clstm, mask_all, t)
    lin_sweep[t]   = fom_metrics(gt_all, prev_all, pred_lin,   mask_all, t)
    print(f"  t={t:.3f}  CNN: FoM={cnn_sweep[t]['fom']:.3f} "
          f"P={cnn_sweep[t]['precision']:.3f} R={cnn_sweep[t]['recall']:.3f} | "
          f"CLSTM: FoM={clstm_sweep[t]['fom']:.3f} "
          f"P={clstm_sweep[t]['precision']:.3f} R={clstm_sweep[t]['recall']:.3f} | "
          f"Lin: FoM={lin_sweep[t]['fom']:.3f} "
          f"P={lin_sweep[t]['precision']:.3f} R={lin_sweep[t]['recall']:.3f}",
          flush=True)

best_t_clstm = max(THRESHOLDS, key=lambda t: clstm_sweep[t]["fom"])

# Trivial baselines (analytical, threshold-independent at these growth rates)
# predict-all-growth
triv_A = 0
triv_B = n_growth
triv_C = n_px - n_growth
triv_fom  = triv_B / (triv_A + triv_B + triv_C)
triv_prec = triv_B / (triv_B + triv_C)
# persistence: predicts no growth => B=0, A=n_growth, C=0 => FoM=0

print(f"\n  Trivial predict-all-growth: FoM={triv_fom:.3f} "
      f"P={triv_prec:.3f} R=1.000")
print(f"  Persistence (predict-none): FoM=0.000 P=undefined R=0.000")

# Best CNN threshold
best_t_cnn = max(THRESHOLDS, key=lambda t: cnn_sweep[t]["fom"])
best_t_lin = max(THRESHOLDS, key=lambda t: lin_sweep[t]["fom"])
print(f"\n  CNN best threshold in sweep: t={best_t_cnn} "
      f"-> FoM={cnn_sweep[best_t_cnn]['fom']:.3f}")
print(f"  Lin best threshold in sweep: t={best_t_lin} "
      f"-> FoM={lin_sweep[best_t_lin]['fom']:.3f}")


# ── Save ───────────────────────────────────────────────────────────────────

def sweep_to_serialisable(sw):
    return {str(k): v for k, v in sw.items()}

results = {
    "experiment": "lagos_full_transfer_eval",
    "description": (
        "Zero-shot transfer evaluation with threshold sweep and baselines. "
        "SimpleCNN seed 42, no retraining. "
        "Eval mask: gt_2015 > 0.01. "
        "Thresholds swept for change classification: " + str(THRESHOLDS)
    ),
    "model": "SimpleCNN_seed42 (best_cnn_3ch.pth, 74K params)",
    "region": "Lagos metropolitan area, Nigeria",
    "bounds_wgs84": LAGOS_BOUNDS,
    "n_tiles": n_tiles_run,
    "n_eval_pixels": n_px,
    "n_growth_pixels": n_growth,
    "pct_growth": round(pct_grow, 2),
    "normalization": {
        "volume": f"log1p / {vol_norm_max:.6f}",
        "population": f"log1p / {pop_norm_max:.6f}",
    },
    "cnn_threshold_sweep": sweep_to_serialisable(cnn_sweep),
    "convlstm_threshold_sweep": sweep_to_serialisable(clstm_sweep),
    "linear_extrap_threshold_sweep": sweep_to_serialisable(lin_sweep),
    "trivial_predict_all_growth": {
        "fom": round(float(triv_fom), 4),
        "precision": round(float(triv_prec), 4),
        "recall": 1.0,
        "A": triv_A, "B": triv_B, "C": triv_C,
    },
    "persistence": {"fom": 0.0, "note": "predicts no growth; FoM=0 by construction"},
    "conus_reference": {
        "cnn_fom_2015_spatial": 0.702,
        "cnn_fom_2020_temporal": 0.252,
    },
    "timestamp": datetime.datetime.now().isoformat(),
}

out_path = RES_DIR / "lagos_transfer_full.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[DONE] Saved -> {out_path}")
