#!/usr/bin/env python3
"""
fig_overview — full-width 2-row infographic for the paper.

Row 1 (60% height): 3 large spatial maps from CONUS GeoTIFFs
  (a) GHSL Ground Truth 2015
  (b) ConvLSTM Prediction 2015
  (c) MC Dropout Uncertainty σ̂

Row 2 (40% height): 3 analytical result panels
  (d) Calibration scatter: σ̂ vs |error| per decile
      Shows rank ordering (r=0.983) AND magnitude underestimation (1.4–1.9×)
  (e) Channel-count decomposition bars for 5yr→10yr CNN drop
  (f) Multi-horizon FoM trajectories (ConvLSTM vs CNN actual vs CNN matched)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import rasterio

ROOT = Path(__file__).resolve().parent.parent
GEO  = ROOT / "geotiff_exports"
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── load tile ─────────────────────────────────────────────────────────────────
R, C, SZ = 6144, 18432, 640

def load(name, r=R, c=C, sz=SZ):
    with rasterio.open(GEO / name) as src:
        return src.read(1, window=rasterio.windows.Window(c, r, sz, sz)).astype(np.float32)

gt_2015   = load("CONUS_builtup_2015.tif")
pred_2015 = load("CONUS_builtup_2015_predicted.tif")
prev_2010 = load("CONUS_builtup_2010.tif")
unc_std   = load("CONUS_mc_uncertainty_std.tif")

obs  = (gt_2015 - prev_2010) > 0.01
pred = (pred_2015 - prev_2010) > 0.01
B = int((obs & pred).sum()); A = int((obs & ~pred).sum()); Cv = int((~obs & pred).sum())
fom = B / (A + B + Cv + 1e-9)

# ── calibration data (from uncertainty_summary.json) ──────────────────────────
# 10 decile bins: (mean_predicted_std, mean_actual_error)
cal_bins = [
    (0.001963, 0.003501),
    (0.002488, 0.003512),
    (0.002868, 0.003881),
    (0.003272, 0.004513),
    (0.003785, 0.005616),
    (0.004522, 0.007459),
    (0.005636, 0.010233),
    (0.007267, 0.013665),
    (0.009679, 0.017630),
    (0.015102, 0.022736),
]
cal_sig  = [s for s, e in cal_bins]
cal_err  = [e for s, e in cal_bins]
ratios   = [e / s for s, e in cal_bins]

# ── channel decomposition data ────────────────────────────────────────────────
clstm_improv    = 0.073   # genuine ConvLSTM improvement 5yr→10yr
cnn_ch_effect   = 0.060   # channel count effect on CNN
cnn_hor_effect  = 0.004   # true CNN horizon effect
total_comp      = clstm_improv + cnn_ch_effect + cnn_hor_effect

# ── multi-horizon data ────────────────────────────────────────────────────────
horizons        = [5, 10, 20]
fom_clstm       = [0.523, 0.596, 0.597]
fom_cnn         = [0.739, 0.675, 0.687]
fom_cnn_matched = [0.739, 0.679, 0.702]  # CNN at matched channel count

# ── colormaps ─────────────────────────────────────────────────────────────────
cmap_built = LinearSegmentedColormap.from_list(
    "built", ["#FAFAFA", "#FFF9C4", "#FFB300", "#E64A19", "#4E342E", "#1A237E"])
cmap_unc = LinearSegmentedColormap.from_list(
    "unc", ["#0A0A0A", "#1B3A5C", "#1565C0", "#00BCD4", "#F9A825", "#FF3D00"])

vmax_b = max(float(np.percentile(gt_2015[gt_2015 > 0.01], 98)), 0.35) if (gt_2015 > 0.01).sum() else 0.4
vmax_u = float(np.percentile(unc_std[unc_std > 0.0001], 96)) if (unc_std > 0.0001).sum() else 0.015

# ── figure ────────────────────────────────────────────────────────────────────
BG = "#12171E"
fig = plt.figure(figsize=(18, 12.5), facecolor=BG)

gs = GridSpec(2, 3, figure=fig,
              height_ratios=[1.6, 1.0],
              hspace=0.09, wspace=0.05,
              left=0.03, right=0.97, top=0.91, bottom=0.07)

fig.text(0.5, 0.965,
         "Metric Failure and Architecture Stability in Continental Urban Growth Prediction (CONUS · 250 m · 1975–2020)",
         ha="center", fontsize=14.5, fontweight="bold", color="#E3F2FD")
fig.text(0.5, 0.943,
         "Sealed 2020 holdout  ·  Six-model benchmark  ·  MSE misranks growth models algebraically  ·  "
         "Channel-count decomposition of multi-horizon CNN behaviour",
         ha="center", fontsize=9.0, color="#90CAF9", style="italic")

# ── Panel (a): Ground Truth ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.set_facecolor(BG)
im = ax.imshow(gt_2015, cmap=cmap_built, vmin=0, vmax=vmax_b,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.012, shrink=0.95)
cb.set_label("Built-up fraction", fontsize=9, color="#B0BEC5")
cb.ax.tick_params(labelsize=8, colors="#B0BEC5")
cb.outline.set_edgecolor("#37474F")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#FF7043"); sp.set_linewidth(2.5)
ax.set_title("(a)  GHSL Ground Truth — Built-Up 2015",
             fontsize=12.5, fontweight="bold", color="#FF7043", pad=8, loc="left")
ax.text(0.015, 0.035, "Sentinel-2 composite  |  held-out val tile",
        transform=ax.transAxes, fontsize=9, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.32", fc="#BF360C", alpha=0.88, ec="none"))
ax.text(0.98, 0.035, "Mid-Atlantic metro  |  128×128 km²",
        transform=ax.transAxes, fontsize=8.5, color="#CFD8DC", va="bottom", ha="right")

# ── Panel (b): ConvLSTM Prediction ───────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor(BG)
im = ax.imshow(pred_2015, cmap=cmap_built, vmin=0, vmax=vmax_b,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.012, shrink=0.95)
cb.set_label("Predicted fraction", fontsize=9, color="#B0BEC5")
cb.ax.tick_params(labelsize=8, colors="#B0BEC5")
cb.outline.set_edgecolor("#37474F")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#66BB6A"); sp.set_linewidth(2.5)
ax.set_title("(b)  ConvLSTM Prediction — 2015",
             fontsize=12.5, fontweight="bold", color="#66BB6A", pad=8, loc="left")
ax.text(0.015, 0.035, f"FoM = {fom:.3f}  |  inputs: 1975–2010  |  target: 2015",
        transform=ax.transAxes, fontsize=9, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.32", fc="#2E7D32", alpha=0.88, ec="none"))
ax.text(0.98, 0.035, "SimpleCNN achieves FoM 0.734 on same holdout",
        transform=ax.transAxes, fontsize=8, color="#CFD8DC", va="bottom", ha="right")

# ── Panel (c): Uncertainty ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.set_facecolor(BG)
im = ax.imshow(unc_std, cmap=cmap_unc, vmin=0, vmax=vmax_u,
               interpolation="bilinear", aspect="equal")
cb = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.012, shrink=0.95)
cb.set_label("Predictive std  σ̂", fontsize=9, color="#B0BEC5")
cb.ax.tick_params(labelsize=8, colors="#B0BEC5")
cb.outline.set_edgecolor("#37474F")
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor("#FF8F00"); sp.set_linewidth(2.5)
ax.set_title("(c)  MC Dropout Uncertainty — 20 Stochastic Passes",
             fontsize=12.5, fontweight="bold", color="#FF8F00", pad=8, loc="left")
ax.text(0.015, 0.035, "Bright = uncertain  |  rank-order r = 0.983  |  1.4–1.9× underestimated",
        transform=ax.transAxes, fontsize=9, color="white", va="bottom",
        bbox=dict(boxstyle="round,pad=0.32", fc="#E65100", alpha=0.88, ec="none"))
ax.text(0.98, 0.035, "Only model providing spatial uncertainty",
        transform=ax.transAxes, fontsize=8, color="#CFD8DC", va="bottom", ha="right")

# ── Panel (d): Calibration scatter ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.set_facecolor("#1C2733")

# colour decile points by their underestimation ratio
norm_ratios = np.array(ratios)
scatter = ax.scatter(cal_sig, cal_err, c=norm_ratios,
                     cmap="YlOrRd", s=80, zorder=4, vmin=1.3, vmax=1.95)
cb4 = plt.colorbar(scatter, ax=ax, fraction=0.042, pad=0.02, shrink=0.9)
cb4.set_label("|error| / σ̂  ratio", fontsize=8, color="#B0BEC5")
cb4.ax.tick_params(labelsize=7, colors="#B0BEC5")
cb4.outline.set_edgecolor("#37474F")

# 1:1 reference line and observed regression
all_range = np.linspace(min(cal_sig) * 0.9, max(cal_sig) * 1.1, 100)
ax.plot(all_range, all_range, "--", color="#78909C", lw=1.4, label="1:1 (perfect calibration)", zorder=3)

# mean ratio line
mean_ratio = float(np.mean(ratios))
ax.plot(all_range, mean_ratio * all_range, "-", color="#EF9A9A", lw=1.8,
        label=f"mean ratio = {mean_ratio:.2f}× (underestimated)", zorder=3)

# label each decile
for i, (s, e) in enumerate(zip(cal_sig, cal_err)):
    ax.text(s, e + 0.0003, f"D{i+1}", fontsize=6.5, color="#90CAF9",
            ha="center", va="bottom")

ax.set_xlabel("Mean predicted σ̂ per decile", fontsize=9, color="#B0BEC5")
ax.set_ylabel("Mean |actual error| per decile", fontsize=9, color="#B0BEC5")
ax.set_title("(d)  Uncertainty Calibration\nRank-ordered ✓  |  Magnitude underestimated ✗",
             fontsize=10.5, fontweight="bold", color="#FFB74D", pad=5, loc="left")
ax.legend(fontsize=7.5, facecolor="#263238", edgecolor="#37474F", labelcolor="white",
          loc="upper left")
ax.tick_params(colors="#78909C")
ax.grid(alpha=0.15, color="white", linestyle="--")
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#37474F")
ax.spines["bottom"].set_color("#37474F")

# ── Panel (e): Channel decomposition ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.set_facecolor("#1C2733")

bar_labels = ["ConvLSTM\ngenuine\nimprovement", "CNN\nchannel\nloss", "CNN\nhorizon\neffect"]
bar_values = [clstm_improv, cnn_ch_effect, cnn_hor_effect]
bar_colors = ["#42A5F5", "#EF9A9A", "#B0BEC5"]
bar_pcts   = [100 * v / total_comp for v in bar_values]
xpos = np.arange(3)

bars = ax.bar(xpos, bar_values, color=bar_colors, width=0.52, zorder=3,
              edgecolor="#263238", linewidth=0.6)
for bar, v, pct in zip(bars, bar_values, bar_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
            f"{v:.3f}\n({pct:.0f}\u202f%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="white")

ax.set_xticks(xpos)
ax.set_xticklabels(bar_labels, fontsize=8.5, color="#B0BEC5")
ax.set_ylabel("FoM contribution", fontsize=9, color="#B0BEC5")
ax.set_title("(e)  5yr\u2192 10yr gap compression\nDecomposed into 3 components",
             fontsize=10.5, fontweight="bold", color="#42A5F5", pad=5, loc="left")
ax.set_ylim(0, 0.115)
ax.tick_params(colors="#78909C")
ax.grid(axis="y", alpha=0.15, color="white", linestyle="--", zorder=0)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#37474F")
ax.spines["bottom"].set_color("#37474F")

# summary text
ax.text(0.98, 0.05,
        f"94% channel effect\n6% true horizon",
        transform=ax.transAxes, fontsize=8, color="#EF9A9A",
        ha="right", va="bottom", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#1C2733", alpha=0.9, ec="#37474F"))

# ── Panel (f): Multi-horizon ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
ax.set_facecolor("#1C2733")

ax.plot(horizons, fom_cnn, "o-", color="#EF5350", lw=2.4, ms=9,
        label="SimpleCNN — actual", zorder=5)
ax.plot(horizons, fom_cnn_matched, "o--", color="#EF9A9A", lw=1.8, ms=8,
        mfc="none", mec="#EF9A9A", mew=2.0, zorder=4,
        label="SimpleCNN — matched channels")
ax.plot(horizons, fom_clstm, "s-", color="#42A5F5", lw=2.4, ms=9,
        label="ConvLSTM", zorder=5)
ax.fill_between(horizons, fom_clstm, fom_cnn, alpha=0.10, color="#78909C")

for yr, vc, vm, vl in zip(horizons, fom_cnn, fom_cnn_matched, fom_clstm):
    ax.annotate(f"{vc:.3f}", (yr, vc), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8.5,
                color="#EF5350", fontweight="bold")
    ax.annotate(f"{vl:.3f}", (yr, vl), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=8.5,
                color="#42A5F5", fontweight="bold")
    if yr > 5:
        ax.annotate(f"({vm:.3f})", (yr, vm), xytext=(-5, 5),
                    textcoords="offset points", ha="right", fontsize=7,
                    color="#EF9A9A")

ax.set_xticks(horizons)
ax.set_xticklabels(["5-yr\n(last: 2010)", "10-yr\n(last: 2005)", "20-yr\n(last: 1995)"],
                   fontsize=9, color="#B0BEC5")
ax.set_ylabel("FoM (2015 holdout)", fontsize=9, color="#B0BEC5")
ax.set_title("(f)  Multi-Horizon FoM\nCNN leads at every horizon",
             fontsize=10.5, fontweight="bold", color="#42A5F5", pad=5, loc="left")
ax.legend(fontsize=8, loc="lower right",
          facecolor="#263238", edgecolor="#37474F", labelcolor="white")
ax.set_ylim(0.38, 0.88)
ax.tick_params(colors="#78909C")
ax.grid(alpha=0.15, color="white", linestyle="--")
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#37474F")
ax.spines["bottom"].set_color("#37474F")

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    p = OUT / f"fig_overview.{ext}"
    fig.savefig(str(p), bbox_inches="tight", dpi=220, facecolor=BG)
    print(f"Saved: {p}")
plt.close(fig)
print("Done.")
