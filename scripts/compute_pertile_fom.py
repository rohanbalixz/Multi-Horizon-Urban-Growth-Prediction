#!/usr/bin/env python3
"""
Per-tile FoM computation + paired Wilcoxon test + bootstrap 95% CIs.

Runs SimpleCNN (seed42), SimpleUNet, and ConvLSTM (seed42) on the 821
canonical val tiles, computing FoM for each tile independently.

Then:
  - Paired Wilcoxon signed-rank test: CNN vs ConvLSTM, U-Net vs ConvLSTM
  - Bootstrap 95% CI on aggregate FoM for each model (B=2000 samples)
  - Bootstrap 95% CI on pairwise FoM differences

Outputs:
  results/metrics/pertile_fom_statistics.json

Runtime: ~20-40 min (three models, 821 tiles each)
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
from scipy.stats import wilcoxon
import rasterio

from src.models.convlstm import ConvLSTM

print("=" * 70, flush=True)
print("PER-TILE FoM — WILCOXON + BOOTSTRAP", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 70, flush=True)

TRAIN_EPOCHS     = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH     = 2015
TILE_SIZE        = 128
CHANGE_THRESHOLD = 0.01
N_BOOTSTRAP      = 2000
SEED_BOOT        = 0
rng = np.random.default_rng(SEED_BOOT)

# ── SimpleCNN ─────────────────────────────────────────────────────────────
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

# ── SimpleUNet ────────────────────────────────────────────────────────────
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

# ── ConvLSTM predict step ─────────────────────────────────────────────────
def predict_convlstm(model, sequences):
    model.eval()
    with torch.no_grad():
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

# ── Load val tiles ─────────────────────────────────────────────────────────
with open(RES / "val_tile_indices.json") as f:
    val_tiles = [tuple(t) for t in json.load(f)]
n_tiles = len(val_tiles)
print(f"\nVal tiles: {n_tiles:,}", flush=True)

# ── Load rasters ───────────────────────────────────────────────────────────
print("[DATA] Loading rasters (float32)...", flush=True)
def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)

bu  = {yr: load_tif(GEO / f"CONUS_builtup_{yr}.tif")    for yr in TRAIN_EPOCHS + [TARGET_EPOCH]}
vol = {yr: load_tif(GEO / f"CONUS_volume_{yr}.tif")     for yr in TRAIN_EPOCHS}
pop = {yr: load_tif(GEO / f"CONUS_population_{yr}.tif") for yr in TRAIN_EPOCHS}
gc.collect()
print(f"  Raster shape: {bu[2010].shape}", flush=True)

# ── Per-tile FoM helper ────────────────────────────────────────────────────
def tile_fom(pred_np, gt_tile, prev_tile, threshold=0.01):
    """FoM on a single tile using the eval_mask = gt > threshold."""
    eval_mask = gt_tile > threshold
    if not eval_mask.any():
        return 0.0
    obs_ch  = (gt_tile   - prev_tile) > threshold
    pred_ch = (pred_np   - prev_tile) > threshold
    B = int((obs_ch[eval_mask] &  pred_ch[eval_mask]).sum())
    A = int((obs_ch[eval_mask] & ~pred_ch[eval_mask]).sum())
    C = int((~obs_ch[eval_mask] & pred_ch[eval_mask]).sum())
    return B / (A + B + C) if (A + B + C) > 0 else 0.0

# ── Build input sequence per tile ──────────────────────────────────────────
def get_tile_seq(i, j):
    frames = []
    for yr in TRAIN_EPOCHS:
        b = bu[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
        v = vol[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
        p = pop[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
        frames.append(np.stack([b, v, p], axis=0))
    return np.stack(frames, axis=0)  # (T, C, H, W)

gt_global   = bu[TARGET_EPOCH]
prev_global = bu[2010]

# ── Run per-tile FoM for each model ───────────────────────────────────────
def run_pertile_fom(model_fn, model_name):
    """Returns array of shape (n_tiles,) with per-tile FoM."""
    foms = np.zeros(n_tiles, dtype=np.float32)
    t0 = time.time()
    for n, (i, j) in enumerate(val_tiles):
        seq  = get_tile_seq(i, j)
        x    = torch.FloatTensor(seq).unsqueeze(0)
        pred = model_fn(x).squeeze().detach().numpy().clip(0, 1)
        gt_t   = gt_global[i:i+TILE_SIZE, j:j+TILE_SIZE]
        prev_t = prev_global[i:i+TILE_SIZE, j:j+TILE_SIZE]
        foms[n] = tile_fom(pred, gt_t, prev_t)
        if (n + 1) % 300 == 0:
            print(f"  [{model_name}] {n+1}/{n_tiles} tiles  "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"  [{model_name}] Done. mean_fom={foms.mean():.4f}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return foms

# CNN
print("\n[CNN seed42] Loading model...", flush=True)
cnn = SimpleCNN(input_channels=24)
cnn.load_state_dict(torch.load(str(MOD / "best_cnn_3ch.pth"),
                                map_location="cpu", weights_only=True))
cnn.eval()
cnn_foms = run_pertile_fom(lambda x: cnn(x).squeeze(1), "CNN-seed42")
del cnn; gc.collect()

# U-Net
print("\n[U-Net seed42] Loading model...", flush=True)
unet = SimpleUNet(input_channels=24)
unet.load_state_dict(torch.load(str(MOD / "best_unet_3ch.pth"),
                                 map_location="cpu", weights_only=True))
unet.eval()
unet_foms = run_pertile_fom(lambda x: unet(x).squeeze(), "U-Net")
del unet; gc.collect()

# ConvLSTM
print("\n[ConvLSTM seed42] Loading model...", flush=True)
clstm = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1)
clstm.load_state_dict(torch.load(str(MOD / "best_3ch_mc_model.pth"),
                                  map_location="cpu", weights_only=True))
clstm.eval()
clstm_foms = run_pertile_fom(
    lambda x: predict_convlstm(clstm, x).squeeze(), "ConvLSTM-seed42"
)
del clstm; gc.collect()

# ── Aggregate FoM (pixel-level equivalent) ─────────────────────────────────
print("\n[AGGREGATE FoM — all val pixels combined]", flush=True)
def aggregate_fom_from_tiles(per_tile_foms):
    """Weighted aggregate: FoM = sum(B_i) / sum(A_i+B_i+C_i) over tiles.
    Per-tile FoM values are unweighted means — use tile counts for proper aggregate.
    Approximation: weight by non-zero tiles only, then compute mean.
    For the Wilcoxon test, per-tile values are what matter."""
    return float(np.mean(per_tile_foms))

print(f"  CNN   per-tile mean FoM: {cnn_foms.mean():.4f} ± {cnn_foms.std():.4f}")
print(f"  U-Net per-tile mean FoM: {unet_foms.mean():.4f} ± {unet_foms.std():.4f}")
print(f"  CLSTM per-tile mean FoM: {clstm_foms.mean():.4f} ± {clstm_foms.std():.4f}")

# Non-zero tiles (where at least one model detected change)
active_mask = (cnn_foms > 0) | (clstm_foms > 0) | (unet_foms > 0)
n_active = int(active_mask.sum())
print(f"\n  Active tiles (≥1 model has FoM>0): {n_active} / {n_tiles}")

# ── Paired Wilcoxon signed-rank tests ─────────────────────────────────────
print("\n[WILCOXON] Paired signed-rank tests:", flush=True)

def wilcoxon_test(a, b, label_a, label_b):
    diff = a - b
    stat, pval = wilcoxon(diff, alternative='greater')
    med_diff = float(np.median(diff))
    print(f"  {label_a} > {label_b}:")
    print(f"    statistic={stat:.1f}  p-value={pval:.2e}  median_diff={med_diff:.4f}")
    return {"statistic": float(stat), "p_value": float(pval), "median_diff": round(med_diff, 4),
            "n_pairs": int(len(diff))}

wil_cnn_vs_clstm  = wilcoxon_test(cnn_foms,  clstm_foms, "CNN",   "ConvLSTM")
wil_unet_vs_clstm = wilcoxon_test(unet_foms, clstm_foms, "U-Net", "ConvLSTM")
wil_cnn_vs_unet   = wilcoxon_test(cnn_foms,  unet_foms,  "CNN",   "U-Net")

# ── Bootstrap 95% CI on aggregate FoM ─────────────────────────────────────
print("\n[BOOTSTRAP] 95% CI on per-tile mean FoM (B=2000)...", flush=True)
def bootstrap_ci(vals, n_boot=N_BOOTSTRAP):
    boot_means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(boot_means.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def bootstrap_ci_diff(a, b, n_boot=N_BOOTSTRAP):
    boot_diffs = np.array([
        rng.choice(a - b, size=len(a), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(boot_diffs.mean()), float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))

cnn_mean_ci,   cnn_lo,   cnn_hi   = bootstrap_ci(cnn_foms)
unet_mean_ci,  unet_lo,  unet_hi  = bootstrap_ci(unet_foms)
clstm_mean_ci, clstm_lo, clstm_hi = bootstrap_ci(clstm_foms)

diff_cnn_clstm_mean, diff_cnn_clstm_lo, diff_cnn_clstm_hi = bootstrap_ci_diff(cnn_foms,  clstm_foms)
diff_unet_clstm_mean, diff_unet_clstm_lo, diff_unet_clstm_hi = bootstrap_ci_diff(unet_foms, clstm_foms)

print(f"  CNN   FoM: {cnn_mean_ci:.4f}  95%CI [{cnn_lo:.4f}, {cnn_hi:.4f}]")
print(f"  U-Net FoM: {unet_mean_ci:.4f}  95%CI [{unet_lo:.4f}, {unet_hi:.4f}]")
print(f"  CLSTM FoM: {clstm_mean_ci:.4f}  95%CI [{clstm_lo:.4f}, {clstm_hi:.4f}]")
print(f"  CNN-CLSTM diff: {diff_cnn_clstm_mean:.4f}  95%CI [{diff_cnn_clstm_lo:.4f}, {diff_cnn_clstm_hi:.4f}]")
print(f"  UNET-CLSTM diff: {diff_unet_clstm_mean:.4f}  95%CI [{diff_unet_clstm_lo:.4f}, {diff_unet_clstm_hi:.4f}]")

# ── Save ──────────────────────────────────────────────────────────────────
out = {
    "experiment": "pertile_fom_statistics",
    "description": (
        "Per-tile FoM computed for CNN/U-Net/ConvLSTM on 821 val tiles "
        "(eval_mask = gt > 0.01 per tile). Wilcoxon paired signed-rank test "
        "and bootstrap 95% CI on per-tile mean FoM."
    ),
    "n_tiles": n_tiles,
    "n_active_tiles": n_active,
    "eval_mask": "gt_2015 > 0.01 per tile",
    "change_threshold": CHANGE_THRESHOLD,
    "n_bootstrap": N_BOOTSTRAP,
    "per_tile_mean_fom": {
        "cnn_seed42":      round(float(cnn_foms.mean()),   4),
        "unet_seed42":     round(float(unet_foms.mean()),  4),
        "convlstm_seed42": round(float(clstm_foms.mean()), 4),
    },
    "per_tile_std_fom": {
        "cnn_seed42":      round(float(cnn_foms.std()),   4),
        "unet_seed42":     round(float(unet_foms.std()),  4),
        "convlstm_seed42": round(float(clstm_foms.std()), 4),
    },
    "bootstrap_ci_95": {
        "cnn_fom":   {"mean": round(cnn_mean_ci, 4),   "lo": round(cnn_lo, 4),   "hi": round(cnn_hi, 4)},
        "unet_fom":  {"mean": round(unet_mean_ci, 4),  "lo": round(unet_lo, 4),  "hi": round(unet_hi, 4)},
        "clstm_fom": {"mean": round(clstm_mean_ci, 4), "lo": round(clstm_lo, 4), "hi": round(clstm_hi, 4)},
        "diff_cnn_minus_clstm":  {
            "mean": round(diff_cnn_clstm_mean, 4),
            "lo":   round(diff_cnn_clstm_lo, 4),
            "hi":   round(diff_cnn_clstm_hi, 4)
        },
        "diff_unet_minus_clstm": {
            "mean": round(diff_unet_clstm_mean, 4),
            "lo":   round(diff_unet_clstm_lo, 4),
            "hi":   round(diff_unet_clstm_hi, 4)
        },
    },
    "wilcoxon_tests": {
        "cnn_gt_convlstm":  wil_cnn_vs_clstm,
        "unet_gt_convlstm": wil_unet_vs_clstm,
        "cnn_gt_unet":      wil_cnn_vs_unet,
    },
    "per_tile_fom_arrays": {
        "cnn_seed42":      cnn_foms.tolist(),
        "unet_seed42":     unet_foms.tolist(),
        "convlstm_seed42": clstm_foms.tolist(),
    },
    "timestamp": str(datetime.datetime.now()),
}

out_path = RES / "pertile_fom_statistics.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[SAVED] {out_path}", flush=True)

print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print(f"  CNN   FoM: {cnn_mean_ci:.4f} [{cnn_lo:.4f}, {cnn_hi:.4f}]")
print(f"  U-Net FoM: {unet_mean_ci:.4f} [{unet_lo:.4f}, {unet_hi:.4f}]")
print(f"  CLSTM FoM: {clstm_mean_ci:.4f} [{clstm_lo:.4f}, {clstm_hi:.4f}]")
print(f"  CNN>CLSTM  p={wil_cnn_vs_clstm['p_value']:.2e}  "
      f"diff={diff_cnn_clstm_mean:.4f} [{diff_cnn_clstm_lo:.4f}, {diff_cnn_clstm_hi:.4f}]")
print(f"  UNET>CLSTM p={wil_unet_vs_clstm['p_value']:.2e}  "
      f"diff={diff_unet_clstm_mean:.4f} [{diff_unet_clstm_lo:.4f}, {diff_unet_clstm_hi:.4f}]")
print("=" * 70, flush=True)
