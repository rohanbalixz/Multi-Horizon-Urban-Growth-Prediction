#!/usr/bin/env python3
"""
fig_title_maps — compact 3-panel teaser placed on the title page above the abstract.

Panels:
  (a) GHSL Ground Truth 2015
  (b) ConvLSTM Prediction 2015
  (c) MC Dropout Uncertainty σ̂  (20 passes)

Same tile as make_map_figure.py (Mid-Atlantic metro, FoM≈0.48).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio

ROOT = Path(__file__).resolve().parent.parent
GEO  = ROOT / "geotiff_exports"
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

R, C, SZ = 6144, 18432, 512   # best val tile: Mid-Atlantic metro

def load(name):
    with rasterio.open(GEO / name) as src:
        return src.read(1, window=rasterio.windows.Window(C, R, SZ, SZ)).astype(np.float32)

gt_2015   = load("CONUS_builtup_2015.tif")
pred_2015 = load("CONUS_builtup_2015_predicted.tif")
unc_std   = load("CONUS_mc_uncertainty_std.tif")

# ── colormaps ─────────────────────────────────────────────────────────────────
cmap_built = LinearSegmentedColormap.from_list(
    "built", ["#FFFDE7", "#FFE082", "#FF8F00", "#BF360C", "#4E0000"])
cmap_unc = LinearSegmentedColormap.from_list(
    "unc",   ["#0D1B2A", "#1565C0", "#00ACC1", "#F9A825", "#E65100"])

vmax_built = max(float(np.percentile(gt_2015[gt_2015 > 0], 98)), 0.3) \
             if (gt_2015 > 0).sum() else 0.4
vmax_unc   = float(np.percentile(unc_std[unc_std > 0], 97)) \
             if (unc_std > 0).sum() else 0.02

# ── figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
fig.patch.set_facecolor("white")
fig.subplots_adjust(wspace=0.10, left=0.01, right=0.99, top=0.87, bottom=0.04)

FS = 10.5

# ── (a) Ground Truth ──────────────────────────────────────────────────────────
ax = axes[0]
im = ax.imshow(gt_2015, cmap=cmap_built, vmin=0, vmax=vmax_built,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.88)
cb.set_label("Built-up fraction", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(a)  GHSL Ground Truth — 2015",
             fontsize=FS, fontweight="bold", color="#BF360C", pad=5)
ax.text(0.02, 0.03, "GHSL R2023A  |  250 m",
        transform=ax.transAxes, fontsize=8, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="#BF360C", alpha=0.82))

# ── (b) Prediction ────────────────────────────────────────────────────────────
ax = axes[1]
im = ax.imshow(pred_2015, cmap=cmap_built, vmin=0, vmax=vmax_built,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.88)
cb.set_label("Predicted fraction", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(b)  ConvLSTM Prediction — 2015",
             fontsize=FS, fontweight="bold", color="#1B5E20", pad=5)

# derive tile-level FoM for annotation
prev_2010 = load("CONUS_builtup_2010.tif")
t = 0.01
obs  = (gt_2015 - prev_2010) > t
pred = (pred_2015 - prev_2010) > t
B = int((obs & pred).sum())
A = int((obs & ~pred).sum())
Cm= int((~obs & pred).sum())
fom = B / (A + B + Cm + 1e-9)

ax.text(0.02, 0.03, f"FoM = {fom:.3f}  (inputs 1975–2010)",
        transform=ax.transAxes, fontsize=8, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="#1B5E20", alpha=0.82))

# ── (c) Uncertainty ───────────────────────────────────────────────────────────
ax = axes[2]
im = ax.imshow(unc_std, cmap=cmap_unc, vmin=0, vmax=vmax_unc,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.88)
cb.set_label("Predictive std  σ̂", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(c)  MC Dropout Uncertainty  (20 passes)",
             fontsize=FS, fontweight="bold", color="#E65100", pad=5)
ax.text(0.02, 0.03, "Bright = high uncertainty",
        transform=ax.transAxes, fontsize=8, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="#BF360C", alpha=0.72))

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    p = OUT / f"fig_title_maps.{ext}"
    fig.savefig(str(p), bbox_inches="tight", dpi=220, facecolor="white")
    print(f"Saved: {p}")
plt.close(fig)
print("Done.")
