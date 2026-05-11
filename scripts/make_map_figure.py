#!/usr/bin/env python3
"""
Generate fig_map — high-quality 4-panel spatial map for the paper.

Best val tile (row=6144, col=18432): Mid-Atlantic metro, FoM≈0.48, high density.

Panels:
  (a) GHSL Ground Truth 2015
  (b) ConvLSTM Prediction 2015
  (c) Growth detected: observed (gt-prev > 0.01) vs predicted
  (d) MC Dropout uncertainty σ̂

All panels share the same spatial extent. Colorbars are per-panel.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
import rasterio

ROOT = Path(__file__).resolve().parent.parent
GEO  = ROOT / "geotiff_exports"
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── tile parameters ───────────────────────────────────────────────────────────
R, C, SZ = 6144, 18432, 512   # best tile: Mid-Atlantic, FoM≈0.48

def load(name, r=R, c=C, sz=SZ):
    with rasterio.open(GEO / name) as src:
        return src.read(1, window=rasterio.windows.Window(c, r, sz, sz)).astype(np.float32)

gt_2015   = load("CONUS_builtup_2015.tif")
pred_2015 = load("CONUS_builtup_2015_predicted.tif")
prev_2010 = load("CONUS_builtup_2010.tif")
unc_std   = load("CONUS_mc_uncertainty_std.tif")

# ── derived layers ────────────────────────────────────────────────────────────
t = 0.01
obs_growth  = (gt_2015 - prev_2010) > t    # actually grew
pred_growth = (pred_2015 - prev_2010) > t  # model predicts growth
# FoM components
B   = obs_growth & pred_growth   # TP
A   = obs_growth & ~pred_growth  # missed growth
C_m = ~obs_growth & pred_growth  # false alarm
TN  = ~obs_growth & ~pred_growth

B_n, A_n, C_n = int(B.sum()), int(A.sum()), int(C_m.sum())
fom = B_n / (A_n + B_n + C_n + 1e-9)

# Growth classification map: 0=stable, 1=TP, 2=miss, 3=false-alarm
growth_map = np.zeros(gt_2015.shape, dtype=np.uint8)
growth_map[B]   = 1  # correctly predicted growth
growth_map[A]   = 2  # missed
growth_map[C_m] = 3  # false alarm

# ── colormaps ─────────────────────────────────────────────────────────────────
cmap_built = LinearSegmentedColormap.from_list(
    "built", ["#FFFDE7", "#FFE082", "#FF8F00", "#BF360C", "#4E0000"])
cmap_unc = LinearSegmentedColormap.from_list(
    "unc", ["#0D1B2A", "#1565C0", "#00ACC1", "#F9A825", "#E65100"])

# FoM class colormap: stable=light grey, TP=green, miss=red, FA=orange
growth_cmap = matplotlib.colors.ListedColormap(
    ["#E8EAF6", "#2E7D32", "#C62828", "#F57F17"])

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
fig.patch.set_facecolor("white")
fig.subplots_adjust(wspace=0.06, left=0.04, right=0.96, top=0.88, bottom=0.12)

vmax_built = max(float(np.percentile(gt_2015[gt_2015 > 0], 98)), 0.3) if (gt_2015 > 0).sum() else 0.4
vmax_unc   = float(np.percentile(unc_std[unc_std > 0], 97)) if (unc_std > 0).sum() else 0.02

FS = 11.5

# ── Panel (a): Ground Truth ───────────────────────────────────────────────────
ax = axes[0]
im = ax.imshow(gt_2015, cmap=cmap_built, vmin=0, vmax=vmax_built,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02, shrink=0.92)
cb.set_label("Built-up fraction", fontsize=9)
cb.ax.tick_params(labelsize=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(a)  GHSL Ground Truth 2015", fontsize=FS, fontweight="bold",
             color="#E65100", pad=7)
ax.text(0.02, 0.03, "Sentinel-2 composite | 250 m",
        transform=ax.transAxes, fontsize=8.5, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E65100", alpha=0.85))

# ── Panel (b): Prediction ─────────────────────────────────────────────────────
ax = axes[1]
im = ax.imshow(pred_2015, cmap=cmap_built, vmin=0, vmax=vmax_built,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02, shrink=0.92)
cb.set_label("Predicted fraction", fontsize=9)
cb.ax.tick_params(labelsize=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(b)  ConvLSTM Prediction — 2015", fontsize=FS, fontweight="bold",
             color="#1B5E20", pad=7)
ax.text(0.02, 0.03, f"FoM = {fom:.3f}  (inputs 1975–2010)",
        transform=ax.transAxes, fontsize=8.5, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1B5E20", alpha=0.85))

# ── Panel (c): Growth classification ─────────────────────────────────────────
ax = axes[2]
im = ax.imshow(growth_map, cmap=growth_cmap, vmin=0, vmax=3,
               interpolation="none", aspect="equal")
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(c)  Growth Classification (t = 0.01)", fontsize=FS, fontweight="bold",
             color="#1A237E", pad=7)
legend_els = [
    Patch(fc="#E8EAF6", ec="#90A4AE", label=f"Stable ({int(TN.sum()):,} px)"),
    Patch(fc="#2E7D32", label=f"Correctly predicted TP ({B_n:,} px)"),
    Patch(fc="#C62828", label=f"Missed growth A ({A_n:,} px)"),
    Patch(fc="#F57F17", label=f"False alarm C ({C_n:,} px)"),
]
ax.legend(handles=legend_els, loc="lower right", fontsize=7.5,
          framealpha=0.92, edgecolor="#90A4AE")
ax.text(0.02, 0.96, f"FoM = B/(A+B+C) = {fom:.3f}",
        transform=ax.transAxes, fontsize=9, color="white", va="top",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1A237E", alpha=0.88))

# ── Panel (d): Uncertainty ────────────────────────────────────────────────────
ax = axes[3]
im = ax.imshow(unc_std, cmap=cmap_unc, vmin=0, vmax=vmax_unc,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02, shrink=0.92)
cb.set_label("Predictive std  σ̂", fontsize=9)
cb.ax.tick_params(labelsize=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(d)  MC Dropout Uncertainty (20 passes)", fontsize=FS, fontweight="bold",
             color="#BF360C", pad=7)
ax.text(0.02, 0.03, "Bright = model is uncertain\nDecile calib. r = 0.983",
        transform=ax.transAxes, fontsize=8.0, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#BF360C", alpha=0.85))

# ── super title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Representative validation tile — Mid-Atlantic metro area  "
    "(128×128 km²,  held-out,  target year unseen during training)",
    fontsize=12, fontweight="bold", color="#1A237E", y=0.97)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    p = OUT / f"fig_map.{ext}"
    fig.savefig(str(p), bbox_inches="tight", dpi=250, facecolor="white")
    print(f"Saved: {p}")
plt.close(fig)
print("Done.")
