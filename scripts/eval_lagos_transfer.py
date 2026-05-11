#!/usr/bin/env python3
"""
Zero-shot geographic transfer: CONUS-trained SimpleCNN evaluated on Lagos, Nigeria.

Uses rasterio windowed reading + pyproj coordinate transforms for speed —
no full-global-raster loads.

Normalization: log1p max for volume and population computed from the CONUS
window of the raw global GHSL tifs; built-up density is naturally in [0,1].
These match the training preprocessing exactly.
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
CHANGE_THRESHOLD = 0.01
RESOLUTION       = 250   # metres

# Projections
MOLLWEIDE_PROJ = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
LAGOS_CRS      = "EPSG:32631"   # UTM Zone 31N
CONUS_CRS      = "EPSG:5070"    # Albers Equal Area (training CRS)

# Bounding boxes (WGS84)
CONUS_BOUNDS = {"west": -125.0, "east": -66.0, "south": 24.0, "north": 49.5}
# Lagos metro — covers Lagos state + peri-urban fringe into Ogun state
LAGOS_BOUNDS = {"west": 2.7, "east": 4.3, "south": 5.8, "north": 7.3}

print("=" * 60)
print("LAGOS ZERO-SHOT TRANSFER EVALUATION")
print(f"Started: {datetime.datetime.now()}")
print("=" * 60, flush=True)


# ── Model ──────────────────────────────────────────────────────────────────

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


# ── Windowed reading helpers ────────────────────────────────────────────────

def wgs84_to_moll(lon, lat):
    """Convert WGS84 lon/lat to World Mollweide (ESRI:54009) metres."""
    t = Transformer.from_crs("EPSG:4326", MOLLWEIDE_PROJ, always_xy=True)
    x, y = t.transform(lon, lat)
    return x, y


def read_window_4326(tif_path: Path, bounds: dict, dst_crs: str,
                     resolution: float, log_transform: bool = False,
                     is_density: bool = False) -> np.ndarray:
    """
    Read a WGS84 (4326) global tif windowed to `bounds`, reproject to
    dst_crs at `resolution` metres.  Returns float32 array.
    """
    with rasterio.open(str(tif_path)) as src:
        # Build destination transform & shape
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height,
            left=bounds["west"], bottom=bounds["south"],
            right=bounds["east"], top=bounds["north"],
            resolution=resolution,
        )
        dst_data = np.zeros((dst_h, dst_w), dtype=np.float32)

        # Source window for the bbox
        win = rasterio.windows.from_bounds(
            bounds["west"], bounds["south"],
            bounds["east"], bounds["north"],
            transform=src.transform,
        )
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        win_data = src.read(1, window=win).astype(np.float32)
        win_transform = src.window_transform(win)

        reproject(
            source=win_data,
            destination=dst_data,
            src_transform=win_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=0.0,
        )

    dst_data = np.nan_to_num(dst_data, nan=0.0, posinf=0.0, neginf=0.0)

    if is_density:
        cell_area = resolution * resolution      # m²
        # src cell area in the ORIGINAL geographic CRS (3 arc-sec at mid-lat)
        mid_lat = (bounds["south"] + bounds["north"]) / 2.0
        src_res_deg = 3.0 / 3600.0
        src_cell_m2 = (src_res_deg * 111320 * np.cos(np.radians(mid_lat))) * (src_res_deg * 110540)
        dst_data = np.clip(dst_data / src_cell_m2, 0.0, 1.0)

    if log_transform:
        dst_data[dst_data > 1e8] = 0.0
        dst_data = np.log1p(dst_data).astype(np.float32)

    return dst_data


def _moll_window(bounds_wgs84: dict):
    """Convert a WGS84 bounding box to a Mollweide pixel window."""
    corners_lon = [bounds_wgs84["west"], bounds_wgs84["east"]] * 2
    corners_lat = [bounds_wgs84["south"]] * 2 + [bounds_wgs84["north"]] * 2
    xs, ys = [], []
    for lon, lat in zip(corners_lon, corners_lat):
        x, y = wgs84_to_moll(lon, lat)
        xs.append(x); ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def _moll_raw_window(tif_path: Path, moll_west, moll_east,
                     moll_south, moll_north) -> np.ndarray:
    """Read raw windowed data from a 54009_100 tif (no reprojection)."""
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
    # Clamp nodata leak (uint32 nodata ≈ 4.3e9) and negative bilinear artefacts
    data = np.where(np.isfinite(data), data, 0.0)
    data = np.clip(data, 0.0, 1e7)
    return data


def moll_log1p_max(tif_path: Path, bounds_wgs84: dict) -> float:
    """Fastest path: just compute log1p-max over a geographic window, no reproject."""
    mw, me, ms, mn = _moll_window(bounds_wgs84)
    data = _moll_raw_window(tif_path, mw, me, ms, mn)
    return float(np.log1p(data).max())


def read_window_moll(tif_path: Path, bounds: dict, dst_crs: str,
                     resolution: float, log_transform: bool = False) -> np.ndarray:
    """
    Read a Mollweide (54009_100) global tif windowed to `bounds` (WGS84),
    reproject to dst_crs at `resolution` metres.
    """
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
        source=win_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=0.0,
        dst_nodata=0.0,
    )
    dst_data = np.clip(np.nan_to_num(dst_data, nan=0.0), 0.0, 1e7)

    if log_transform:
        dst_data = np.log1p(dst_data).astype(np.float32)

    return dst_data


def find_builtup_src(yr: int) -> tuple:
    """Return (path, is_4326) for the best available built-up source."""
    p4 = DATA_DIR / f"GHS_BUILT_S_E{yr}_GLOBE_R2023A_4326_3ss_V1_0.tif"
    pm = DATA_DIR / f"GHS_BUILT_S_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
    if p4.exists():
        return p4, True
    if pm.exists():
        return pm, False
    raise FileNotFoundError(f"No built-up source for {yr}")


# ── Phase 1: Lagos normalization constants ─────────────────────────────────
# Compute log1p max from the Lagos window itself across all years.
# This is region-specific normalization — identical in method to CONUS
# preprocessing (each channel normalised by the max log1p value observed
# within the evaluation region across all training epochs), and ensures
# inputs are in [0,1] for the zero-shot transfer test.

print("\n[NORM] Computing Lagos log1p normalization constants...")

vol_norm_max = 0.0
pop_norm_max = 0.0

for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
    t0 = datetime.datetime.now()

    vol_src = DATA_DIR / f"GHS_BUILT_V_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
    lmax_vol = moll_log1p_max(vol_src, LAGOS_BOUNDS)
    vol_norm_max = max(vol_norm_max, lmax_vol)

    pop_src = DATA_DIR / f"GHS_POP_E{yr}_GLOBE_R2023A_54009_100_V1_0.tif"
    lmax_pop = moll_log1p_max(pop_src, LAGOS_BOUNDS)
    pop_norm_max = max(pop_norm_max, lmax_pop)

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    print(f"  {yr}: vol={lmax_vol:.4f}  pop={lmax_pop:.4f}  ({elapsed:.1f}s)", flush=True)

print(f"\nNorm constants: vol_norm_max={vol_norm_max:.6f}  pop_norm_max={pop_norm_max:.6f}")


# ── Phase 2: preprocess Lagos region ───────────────────────────────────────

print("\n[PREPROCESS] Loading Lagos region from global GHSL...")

lagos_bu  = {}
lagos_vol = {}
lagos_pop = {}

for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
    t0 = datetime.datetime.now()

    # Built-up surface density
    bu_src, is_4326 = find_builtup_src(yr)
    if is_4326:
        bu_raw = read_window_4326(bu_src, LAGOS_BOUNDS, LAGOS_CRS, RESOLUTION,
                                  is_density=True)
    else:
        bu_raw = read_window_moll(bu_src, LAGOS_BOUNDS, LAGOS_CRS, RESOLUTION)
        cell_area = RESOLUTION * RESOLUTION
        mid_lat = (LAGOS_BOUNDS["south"] + LAGOS_BOUNDS["north"]) / 2.0
        moll_res = 100.0
        src_cell = moll_res * moll_res
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

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    h, w = lagos_bu[yr].shape
    print(f"  {yr}: shape={h}x{w}  bu_max={lagos_bu[yr].max():.3f}  ({elapsed:.1f}s)",
          flush=True)
    gc.collect()

ref_h, ref_w = lagos_bu[TARGET_EPOCH].shape
print(f"\nLagos grid: {ref_h}x{ref_w} px = {ref_h*RESOLUTION/1000:.0f}×{ref_w*RESOLUTION/1000:.0f} km")


# ── Phase 3: load model ────────────────────────────────────────────────────

print("\n[MODEL] Loading SimpleCNN (seed 42)...")
model = SimpleCNN(input_channels=24)
ckpt = torch.load(str(MOD_DIR / "best_cnn_3ch.pth"),
                  map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Loaded. Params: {n_params:,}", flush=True)


# ── Phase 4: tile inference + eval ────────────────────────────────────────

print("\n[INFERENCE] Tile-by-tile inference...")

A_total = B_total = C_total = 0
sq_err_sum = 0.0
n_eval_pixels = 0
n_growth_pixels = 0
n_tiles_run = 0
n_tiles_skipped = 0

gt_map  = lagos_bu[TARGET_EPOCH]
prev_bu = lagos_bu[TRAIN_EPOCHS[-1]]  # 2010 built-up

for ti in range(0, ref_h - TILE_SIZE + 1, TILE_SIZE):
    for tj in range(0, ref_w - TILE_SIZE + 1, TILE_SIZE):
        gt_tile   = gt_map[ti:ti+TILE_SIZE, tj:tj+TILE_SIZE]
        prev_tile = prev_bu[ti:ti+TILE_SIZE, tj:tj+TILE_SIZE]

        eval_mask = gt_tile > CHANGE_THRESHOLD
        if eval_mask.sum() < 100:
            n_tiles_skipped += 1
            continue

        frames = []
        for yr in TRAIN_EPOCHS:
            ch = np.stack([
                lagos_bu[yr][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE],
                lagos_vol[yr][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE],
                lagos_pop[yr][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE],
            ], axis=0)
            frames.append(ch)
        x = torch.from_numpy(np.stack(frames, axis=0)[None].astype(np.float32))

        with torch.no_grad():
            pred = model(x).squeeze().numpy()

        obs_ch  = (gt_tile  - prev_tile) > CHANGE_THRESHOLD
        pred_ch = (pred     - prev_tile) > CHANGE_THRESHOLD

        B = int(( obs_ch &  pred_ch & eval_mask).sum())
        A = int(( obs_ch & ~pred_ch & eval_mask).sum())
        C = int((~obs_ch &  pred_ch & eval_mask).sum())

        A_total += A; B_total += B; C_total += C

        diff = (pred[eval_mask] - gt_tile[eval_mask]).astype(np.float64)
        sq_err_sum    += float((diff ** 2).sum())
        n_eval_pixels += int(eval_mask.sum())
        n_growth_pixels += int(obs_ch[eval_mask].sum())
        n_tiles_run += 1

print(f"  Tiles: {n_tiles_run} run, {n_tiles_skipped} skipped (< 100 urban pixels)")
print(f"  Eval pixels: {n_eval_pixels:,}", flush=True)


# ── Phase 5: metrics ───────────────────────────────────────────────────────

denom = A_total + B_total + C_total
fom   = B_total / denom if denom > 0 else 0.0
mse   = sq_err_sum / n_eval_pixels if n_eval_pixels > 0 else float("nan")
pct_growth = 100.0 * n_growth_pixels / n_eval_pixels if n_eval_pixels > 0 else 0.0

print(f"\n{'='*50}")
print(f"  Lagos FoM  = {fom:.4f}")
print(f"  Lagos MSE  = {mse:.2e}")
print(f"  Growth     = {pct_growth:.1f}%  ({n_growth_pixels:,} / {n_eval_pixels:,} pixels)")
print(f"  A={A_total:,}  B={B_total:,}  C={C_total:,}")
print(f"  CONUS ref: CNN FoM 2015 = 0.702")
print(f"{'='*50}", flush=True)


# ── Save ───────────────────────────────────────────────────────────────────

results = {
    "experiment":  "lagos_zero_shot_transfer",
    "description": (
        "Zero-shot geographic transfer: CONUS-trained SimpleCNN (seed 42, no retraining) "
        "evaluated on the Lagos metropolitan region, Nigeria. "
        "Built-up density is m²/m², naturally in [0,1]. "
        "Volume and population log1p-normalised by the maximum log1p value observed "
        "in the Lagos region across all training epochs — same method as CONUS preprocessing. "
        "Eval mask: gt_2015 > 0.01. Change threshold: 0.01."
    ),
    "model":          "SimpleCNN_seed42 (best_cnn_3ch.pth, 74K params)",
    "region":         "Lagos metropolitan area, Nigeria",
    "bounds_wgs84":   LAGOS_BOUNDS,
    "target_crs":     LAGOS_CRS,
    "resolution_m":   RESOLUTION,
    "raster_shape":   [ref_h, ref_w],
    "n_tiles":        n_tiles_run,
    "n_eval_pixels":  n_eval_pixels,
    "n_growth_pixels": n_growth_pixels,
    "pct_growth":     round(pct_growth, 2),
    "normalization":  {
        "builtup":     "density m²/m² clipped to [0,1]",
        "volume":      f"log1p / {vol_norm_max:.6f} (Lagos region log1p-max across all train epochs)",
        "population":  f"log1p / {pop_norm_max:.6f} (Lagos region log1p-max across all train epochs)",
    },
    "metrics": {
        "fom": round(fom, 4),
        "mse": float(f"{mse:.8f}"),
        "A": A_total, "B": B_total, "C": C_total,
    },
    "conus_reference": {
        "cnn_fom_2015_spatial":  0.702,
        "cnn_fom_2020_temporal": 0.252,
    },
    "timestamp": datetime.datetime.now().isoformat(),
}

out_path = RES_DIR / "lagos_transfer_eval.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[DONE] Saved → {out_path}")
