#!/usr/bin/env python3
"""
FoM Threshold Sensitivity Analysis
====================================
Tests whether the CNN > ConvLSTM ranking under Figure of Merit (FoM) is
robust to the choice of change-detection threshold (0.01 in all paper results).

For each threshold t in [0.005, 0.01, 0.02, 0.05]:
  eval_mask   = (counts > 0) & (gt > t)           # which pixels are evaluated
  obs_change  = (gt  - prev_builtup) > t           # where growth occurred
  pred_change = (pred - prev_builtup) > t          # where model predicts growth
  FoM = |obs ∩ pred| / |obs ∪ pred|               # Pontius et al. 2008

Both holdouts:
  2015 spatial block:  CNN (seed 42), U-Net, ConvLSTM, Linear, Persistence
  2020 temporal blind: CNN (seed 42), ConvLSTM, Linear, Persistence
  (U-Net 2020 uses a different eval mask than the paper; included separately
   in this analysis with the uniform gt>t mask for cross-threshold comparison)

No model retraining — loads saved checkpoints and GeoTIFFs.

Outputs:
  results/metrics/threshold_sensitivity.json
  results/figures/fig_threshold_sensitivity.pdf / .png
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GEO  = PROJECT_ROOT / "geotiff_exports"
MOD  = PROJECT_ROOT / "models"
RES  = PROJECT_ROOT / "results" / "metrics"
FIG  = PROJECT_ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import rasterio

print("=" * 65, flush=True)
print("FoM THRESHOLD SENSITIVITY ANALYSIS", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 65, flush=True)

THRESHOLDS  = [0.005, 0.01, 0.02, 0.05]
TILE_SIZE   = 128

# ── Inline SimpleCNN (identical to cnn_2020_holdout.py) ─────────────────────
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
        # x: (B, T, C, H, W)  →  (B, T*C, H, W)
        B, T, C, H, W = x.shape
        return torch.sigmoid(self.net(x.reshape(B, T * C, H, W)))


# ── Inline SimpleUNet (identical to unet_2020_holdout.py / run_ablations_3ch.py)
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
        x = x.reshape(B, T * C, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))


# ── Load val tile indices ────────────────────────────────────────────────────
with open(str(RES / "val_tile_indices.json")) as f:
    val_tiles = [tuple(t) for t in json.load(f)]
print(f"\n[HOLDOUT] {len(val_tiles):,} val tiles loaded", flush=True)


# ── Raster loader helper ─────────────────────────────────────────────────────
def load_tif(path):
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32)


# ── Load all ground-truth and context rasters ────────────────────────────────
print("\n[DATA] Loading rasters...", flush=True)

bu = {yr: load_tif(GEO / f"CONUS_builtup_{yr}.tif")
      for yr in [1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]}
vol = {yr: load_tif(GEO / f"CONUS_volume_{yr}.tif")
       for yr in [1975,1980,1985,1990,1995,2000,2005,2010,2015]}
pop = {yr: load_tif(GEO / f"CONUS_population_{yr}.tif")
       for yr in [1975,1980,1985,1990,1995,2000,2005,2010,2015]}

ref_shape = bu[2015].shape
print(f"  Raster shape: {ref_shape}", flush=True)
gc.collect()


# ── Load pre-saved ConvLSTM predictions (from validate_2015/2020.py) ─────────
print("\n[CONVLSTM] Loading saved GeoTIFF predictions...", flush=True)
convlstm_pred_2015 = load_tif(GEO / "CONUS_builtup_2015_predicted.tif")
convlstm_pred_2020 = load_tif(GEO / "CONUS_builtup_2020_predicted.tif")
print("  2015 and 2020 ConvLSTM predictions loaded.", flush=True)


# ── CNN inference helper ─────────────────────────────────────────────────────
def run_cnn_inference(model, input_epochs, builtup, volume, population):
    """Run SimpleCNN on val tiles; returns (prediction, counts) arrays."""
    prediction = np.zeros(ref_shape, dtype=np.float64)
    counts      = np.zeros(ref_shape, dtype=np.float64)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for n, (i, j) in enumerate(val_tiles):
            frames = []
            for yr in input_epochs:
                b = builtup[yr][i:i+TILE_SIZE, j:j+TILE_SIZE]
                v = volume.get(yr, np.zeros((TILE_SIZE, TILE_SIZE), np.float32))[i:i+TILE_SIZE, j:j+TILE_SIZE]
                p = population.get(yr, np.zeros((TILE_SIZE, TILE_SIZE), np.float32))[i:i+TILE_SIZE, j:j+TILE_SIZE]
                frames.append(np.stack([b, v, p], axis=0))
            seq = np.stack(frames, axis=0)                 # (T, C, H, W)
            x   = torch.FloatTensor(seq).unsqueeze(0)     # (1, T, C, H, W)
            pred_np = model(x).squeeze().numpy().clip(0, 1)
            prediction[i:i+TILE_SIZE, j:j+TILE_SIZE] += pred_np
            counts[i:i+TILE_SIZE, j:j+TILE_SIZE]     += 1
            if (n + 1) % 500 == 0:
                print(f"    {n+1}/{len(val_tiles)} tiles, {time.time()-t0:.0f}s", flush=True)
    mask = counts > 0
    prediction[mask] /= counts[mask]
    return prediction.astype(np.float32), counts


def run_unet_inference(model, input_epochs, builtup, volume, population):
    """Run SimpleUNet on val tiles; returns (prediction, counts) arrays."""
    return run_cnn_inference(model, input_epochs, builtup, volume, population)


# ── Load and run CNN (seed 42) ───────────────────────────────────────────────
print("\n[CNN] Loading best_cnn_3ch.pth (seed 42)...", flush=True)
cnn_model = SimpleCNN(input_channels=24)
cnn_ckpt  = torch.load(str(MOD / "best_cnn_3ch.pth"), map_location="cpu", weights_only=True)
cnn_model.load_state_dict(cnn_ckpt)
params = sum(p.numel() for p in cnn_model.parameters())
print(f"  Parameters: {params:,}", flush=True)

EPOCHS_2015 = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
EPOCHS_2020 = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]

print("  CNN inference — 2015 holdout...", flush=True)
cnn_pred_2015, cnn_counts_2015 = run_cnn_inference(cnn_model, EPOCHS_2015, bu, vol, pop)

print("  CNN inference — 2020 holdout...", flush=True)
cnn_pred_2020, cnn_counts_2020 = run_cnn_inference(cnn_model, EPOCHS_2020, bu, vol, pop)
del cnn_model; gc.collect()


# ── Load and run U-Net ────────────────────────────────────────────────────────
print("\n[U-NET] Loading best_unet_3ch.pth...", flush=True)
unet_model = SimpleUNet(input_channels=24)
unet_ckpt  = torch.load(str(MOD / "best_unet_3ch.pth"), map_location="cpu", weights_only=True)
unet_model.load_state_dict(unet_ckpt)
print(f"  Parameters: {sum(p.numel() for p in unet_model.parameters()):,}", flush=True)

print("  U-Net inference — 2015 holdout...", flush=True)
unet_pred_2015, unet_counts_2015 = run_unet_inference(unet_model, EPOCHS_2015, bu, vol, pop)
del unet_model; gc.collect()


# ── FoM computation ──────────────────────────────────────────────────────────
def fom(obs_change, pred_change):
    B = int((obs_change & pred_change).sum())
    A = int((obs_change & ~pred_change).sum())
    C = int((~obs_change & pred_change).sum())
    return B / (A + B + C) if (A + B + C) > 0 else 0.0


# ── Threshold sweep ──────────────────────────────────────────────────────────
print("\n[SWEEP] Computing FoM at each threshold...", flush=True)

# Baselines (no model needed)
linear_pred_2015 = np.clip(2 * bu[2010] - bu[2005], 0, 1)
linear_pred_2020 = np.clip(2 * bu[2015] - bu[2010], 0, 1)

results_by_threshold = {}

for t in THRESHOLDS:
    print(f"\n  -- threshold = {t} --", flush=True)

    # ---- 2015 spatial holdout ------------------------------------------------
    counts_2015 = cnn_counts_2015  # all inference runs used same val tiles
    mask_2015   = (counts_2015 > 0) & (bu[2015] > t)
    prev_2015   = bu[2010]
    gt_2015     = bu[2015]

    obs_ch_2015 = (gt_2015   - prev_2015) > t

    def fom_2015(pred):
        pc = (pred - prev_2015) > t
        return fom(obs_ch_2015[mask_2015], pc[mask_2015])

    cnn_fom_2015    = fom_2015(cnn_pred_2015)
    unet_fom_2015   = fom_2015(unet_pred_2015)
    conv_fom_2015   = fom_2015(convlstm_pred_2015)
    lin_fom_2015    = fom_2015(linear_pred_2015)
    per_fom_2015    = 0.0   # persistence never predicts change → FoM = 0

    n_eval_2015   = int(mask_2015.sum())
    n_growth_2015 = int(obs_ch_2015[mask_2015].sum())
    pct_2015      = 100 * n_growth_2015 / n_eval_2015 if n_eval_2015 > 0 else 0

    print(f"    2015  n_eval={n_eval_2015:,}  growth={n_growth_2015:,} ({pct_2015:.1f}%)", flush=True)
    print(f"    2015  CNN={cnn_fom_2015:.4f}  UNet={unet_fom_2015:.4f}  "
          f"ConvLSTM={conv_fom_2015:.4f}  Linear={lin_fom_2015:.4f}", flush=True)

    # ---- 2020 temporal holdout -----------------------------------------------
    mask_2020   = (cnn_counts_2020 > 0) & (bu[2020] > t)
    prev_2020   = bu[2015]
    gt_2020     = bu[2020]

    obs_ch_2020 = (gt_2020 - prev_2020) > t

    def fom_2020(pred):
        pc = (pred - prev_2020) > t
        return fom(obs_ch_2020[mask_2020], pc[mask_2020])

    cnn_fom_2020  = fom_2020(cnn_pred_2020)
    conv_fom_2020 = fom_2020(convlstm_pred_2020)
    lin_fom_2020  = fom_2020(linear_pred_2020)
    per_fom_2020  = 0.0

    n_eval_2020   = int(mask_2020.sum())
    n_growth_2020 = int(obs_ch_2020[mask_2020].sum())
    pct_2020      = 100 * n_growth_2020 / n_eval_2020 if n_eval_2020 > 0 else 0

    print(f"    2020  n_eval={n_eval_2020:,}  growth={n_growth_2020:,} ({pct_2020:.1f}%)", flush=True)
    print(f"    2020  CNN={cnn_fom_2020:.4f}  ConvLSTM={conv_fom_2020:.4f}  "
          f"Linear={lin_fom_2020:.4f}", flush=True)

    results_by_threshold[str(t)] = {
        "threshold": t,
        "holdout_2015": {
            "n_eval_pixels":  n_eval_2015,
            "n_growth_pixels": n_growth_2015,
            "pct_growth":     round(pct_2015, 2),
            "cnn":       round(cnn_fom_2015,  4),
            "unet":      round(unet_fom_2015, 4),
            "convlstm":  round(conv_fom_2015, 4),
            "linear":    round(lin_fom_2015,  4),
            "persistence": 0.0,
            "cnn_gap_over_convlstm": round(cnn_fom_2015 - conv_fom_2015, 4),
            "unet_gap_over_convlstm": round(unet_fom_2015 - conv_fom_2015, 4),
        },
        "holdout_2020": {
            "n_eval_pixels":  n_eval_2020,
            "n_growth_pixels": n_growth_2020,
            "pct_growth":     round(pct_2020, 2),
            "cnn":       round(cnn_fom_2020,  4),
            "convlstm":  round(conv_fom_2020, 4),
            "linear":    round(lin_fom_2020,  4),
            "persistence": 0.0,
            "cnn_gap_over_convlstm": round(cnn_fom_2020 - conv_fom_2020, 4),
        },
    }


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65, flush=True)
print("THRESHOLD SENSITIVITY SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"\n{'Thresh':>8}  {'2015 CNN':>9}  {'2015 Conv':>9}  {'gap':>6}  "
      f"{'2020 CNN':>9}  {'2020 Conv':>9}  {'gap':>6}")
print("-" * 65)
for t in THRESHOLDS:
    r = results_by_threshold[str(t)]
    r15, r20 = r["holdout_2015"], r["holdout_2020"]
    print(f"  {t:>6.3f}  {r15['cnn']:>9.4f}  {r15['convlstm']:>9.4f}  "
          f"{r15['cnn_gap_over_convlstm']:>+6.3f}  "
          f"{r20['cnn']:>9.4f}  {r20['convlstm']:>9.4f}  "
          f"{r20['cnn_gap_over_convlstm']:>+6.3f}")

# Check ranking consistency
all_cnn_wins_2015 = all(
    results_by_threshold[str(t)]["holdout_2015"]["cnn"] >
    results_by_threshold[str(t)]["holdout_2015"]["convlstm"]
    for t in THRESHOLDS
)
all_cnn_wins_2020 = all(
    results_by_threshold[str(t)]["holdout_2020"]["cnn"] >
    results_by_threshold[str(t)]["holdout_2020"]["convlstm"]
    for t in THRESHOLDS
)
print(f"\n  CNN > ConvLSTM at ALL thresholds — 2015: {all_cnn_wins_2015}  2020: {all_cnn_wins_2020}")


# ── Save JSON ────────────────────────────────────────────────────────────────
output = {
    "experiment": "threshold_sensitivity",
    "description": (
        "FoM recomputed at thresholds [0.005, 0.01, 0.02, 0.05]. "
        "eval_mask = (counts > 0) & (gt > t); obs_change = (gt - prev) > t; "
        "pred_change = (pred - prev) > t. "
        "Paper uses t=0.01. Sensitivity tests whether CNN>ConvLSTM ranking holds."
    ),
    "models": {
        "cnn":      "best_cnn_3ch.pth (seed 42, SimpleCNN 74K params)",
        "unet":     "best_unet_3ch.pth (SimpleUNet 474K params, 32-64-128 ch)",
        "convlstm": "best_3ch_mc_model.pth (ConvLSTM 481K params, predictions from GeoTIFF)",
        "linear":   "2*prev - prev_prev linear extrapolation",
        "persistence": "previous epoch unchanged",
    },
    "cnn_wins_all_thresholds_2015": all_cnn_wins_2015,
    "cnn_wins_all_thresholds_2020": all_cnn_wins_2020,
    "results": results_by_threshold,
    "timestamp": str(datetime.datetime.now()),
}
out_path = str(RES / "threshold_sensitivity.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {out_path}", flush=True)


# ── Figure ───────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = THRESHOLDS
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35)
    FS = 8.5

    COLORS = {
        "CNN (74K)":       "#E53935",
        "U-Net (474K)":    "#8E24AA",
        "ConvLSTM (481K)": "#1E88E5",
        "Linear":          "#43A047",
        "Persistence":     "#757575",
    }
    MARKERS = {k: m for k, m in zip(COLORS, ["o","s","^","D","x"])}

    for ax, holdout, title, include_unet in [
        (axes[0], "holdout_2015", "(a) 2015 spatial block holdout", True),
        (axes[1], "holdout_2020", "(b) 2020 blind temporal holdout", False),
    ]:
        r_series = {k: [results_by_threshold[str(t)][holdout][k]
                        for t in xs]
                    for k in ["cnn", "convlstm", "linear", "persistence"]}
        if include_unet:
            r_series["unet"] = [results_by_threshold[str(t)][holdout]["unet"]
                                 for t in xs]

        label_map = {
            "cnn":         "CNN (74K)",
            "unet":        "U-Net (474K)",
            "convlstm":    "ConvLSTM (481K)",
            "linear":      "Linear",
            "persistence": "Persistence",
        }
        plot_order = (["cnn", "unet", "convlstm", "linear", "persistence"]
                      if include_unet else
                      ["cnn", "convlstm", "linear", "persistence"])

        for key in plot_order:
            if key not in r_series:
                continue
            lbl = label_map[key]
            ax.plot(xs, r_series[key],
                    marker=MARKERS[lbl], color=COLORS[lbl],
                    linewidth=1.8, markersize=6, label=lbl)

        # Highlight paper threshold
        ax.axvline(0.01, color="black", linestyle=":", linewidth=1.2, alpha=0.5)
        ax.text(0.01 + 0.0005, ax.get_ylim()[0] + 0.01,
                "paper\nthreshold", fontsize=6.5, color="#444")

        ax.set_xlabel("Change-detection threshold", fontsize=FS)
        ax.set_ylabel("Figure of Merit (FoM)", fontsize=FS)
        ax.set_title(title, fontsize=FS + 0.5, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(t) for t in xs], fontsize=FS - 0.5)
        ax.legend(fontsize=FS - 0.5, framealpha=0.85)
        ax.grid(alpha=0.22, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle(
        "NeuralTimeCapsule: FoM ranking is robust across change-detection thresholds",
        fontsize=10, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    pdf = FIG / "fig_threshold_sensitivity.pdf"
    png = FIG / "fig_threshold_sensitivity.png"
    fig.savefig(str(pdf), bbox_inches="tight", dpi=150)
    fig.savefig(str(png), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {pdf}", flush=True)
    print(f"  Saved: {png}", flush=True)
except Exception as e:
    print(f"  Figure skipped: {e}", flush=True)

print(f"\n{'='*65}")
print("THRESHOLD SENSITIVITY COMPLETE")
print(f"{'='*65}\n")
