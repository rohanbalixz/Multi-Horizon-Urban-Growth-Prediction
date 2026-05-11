#!/usr/bin/env python3
"""
fig_multihorizon — two-panel figure with corrected channel-count decomposition.

Panel (a): FoM vs horizon
  - ConvLSTM filled squares (genuine multi-horizon)
  - CNN filled circles (actual multi-horizon, different channel counts)
  - CNN open circles / dashed (5yr task, matched channel counts)
  Shows: ConvLSTM improves; channel-matched CNN is flat

Panel (b): Gap decomposition bar chart for 5yr→10yr compression
  - Stacked: ConvLSTM genuine improvement | CNN channel effect | CNN horizon effect

All data from verified JSON files.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── verified numbers ───────────────────────────────────────────────────────────
horizons   = [5, 10, 20]

# ConvLSTM: actual multi-horizon FoMs (from multihorizon_results.json)
fom_clstm  = [0.523, 0.596, 0.597]

# CNN: actual multi-horizon FoMs (from cnn_multihorizon_results.json)
# These use different channel counts: 24ch, 21ch, 15ch
fom_cnn    = [0.739, 0.675, 0.687]

# CNN: 5yr task with matched channel counts (from cnn_channel_control_results.json)
# Same 5yr task (last_input=2010, target=2015), only channel count changes
fom_cnn_matched = [
    0.739,   # 24ch — same as fom_cnn[0]
    0.679,   # 21ch — matched to 10yr experiment
    0.702,   # 15ch — matched to 20yr experiment
]

# ── decomposition for 5yr→10yr gap compression ───────────────────────────────
# Gap at 5yr: 0.739 - 0.523 = 0.216
# Gap at 10yr (observed): 0.675 - 0.596 = 0.079
# Gap compression: 0.216 - 0.079 = 0.137

clstm_improvement  = fom_clstm[1]  - fom_clstm[0]   # +0.073 (genuine)
cnn_channel_effect = fom_cnn[0]    - fom_cnn_matched[1]  # +0.060 (channel loss)
cnn_horizon_effect = fom_cnn_matched[1] - fom_cnn[1]     # +0.004 (true horizon)

total_compression = clstm_improvement + cnn_channel_effect + cnn_horizon_effect

# ── colour palette ────────────────────────────────────────────────────────────
C_CNN    = "#C62828"
C_CLSTM  = "#1565C0"
C_CNN_M  = "#E57373"    # lighter red for matched CNN
C_CH     = "#78909C"    # channel effect
C_HOR    = "#B0BEC5"    # horizon effect

FS  = 10.5
FSS = 9.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                gridspec_kw={"width_ratios": [1.5, 1.0]})
fig.subplots_adjust(wspace=0.40, left=0.08, right=0.97, top=0.88, bottom=0.14)

# ── Panel (a): FoM vs horizon ─────────────────────────────────────────────────
# ConvLSTM line
ax1.plot(horizons, fom_clstm, "s-", color=C_CLSTM, lw=2.4, ms=9, zorder=5,
         label="ConvLSTM (recurrent, 3 ch × T epochs)")
# CNN actual (varying channels)
ax1.plot(horizons, fom_cnn, "o-", color=C_CNN, lw=2.4, ms=9, zorder=5,
         label="SimpleCNN — actual (24 / 21 / 15 channels)")
# CNN channel-matched (5yr task, reduced channels)
ax1.plot(horizons, fom_cnn_matched, "o--", color=C_CNN_M, lw=1.8, ms=8,
         mfc="white", mec=C_CNN_M, mew=2.0, zorder=4,
         label="SimpleCNN — 5yr task, matched channel count")

# Decomposition brackets — placed slightly to the right of x=10 so they no
# longer sit on top of the 10yr data labels. Channel-effect bracket spans
# the gap from the matched-CNN 10yr level up to the CNN-5yr level; horizon-
# effect bracket spans from matched-CNN to actual-CNN at 10yr.
BR_X   = 11.0   # x-position for the two decomposition brackets
TXT_X  = 11.25  # text sits just to the right of the bracket

ax1.annotate("", xy=(BR_X, fom_cnn_matched[1]), xytext=(BR_X, fom_cnn[0]),
             arrowprops=dict(arrowstyle="<->", color=C_CH, lw=1.3))
ax1.text(TXT_X, (fom_cnn[0] + fom_cnn_matched[1]) / 2,
         f"channel\neffect\n−{cnn_channel_effect:.3f}",
         fontsize=7.5, color=C_CH, va="center", ha="left", style="italic")

ax1.annotate("", xy=(BR_X, fom_cnn[1]), xytext=(BR_X, fom_cnn_matched[1]),
             arrowprops=dict(arrowstyle="<->", color=C_HOR, lw=1.3))
ax1.text(TXT_X, fom_cnn_matched[1] - 0.012,
         f"horizon\neffect\n−{cnn_horizon_effect:.3f}",
         fontsize=7.5, color=C_HOR, va="top", ha="left", style="italic")

# ConvLSTM improvement annotation — sit it in the empty white space between
# the blue line (lower) and red line (upper), to the left of x=10.
ax1.annotate("", xy=(8.6, fom_clstm[1]), xytext=(8.6, fom_clstm[0]),
             arrowprops=dict(arrowstyle="<->", color=C_CLSTM, lw=1.3))
ax1.text(8.35, (fom_clstm[0] + fom_clstm[1]) / 2,
         f"+{clstm_improvement:.3f}\ngenuine", fontsize=7.5,
         color=C_CLSTM, va="center", ha="right", style="italic")

# data labels.  Use larger pixel offsets so labels do not collide with the
# matched-CNN open markers (which sit only 0.003–0.015 FoM away from the
# filled circles in some places).
for yr, vc, vm, vl in zip(horizons, fom_cnn, fom_cnn_matched, fom_clstm):
    # CNN actual (filled red): label above
    ax1.annotate(f"{vc:.3f}", (yr, vc), xytext=(0, 12),
                 textcoords="offset points", ha="center", fontsize=FSS - 1.5,
                 color=C_CNN, fontweight="bold")
    # ConvLSTM (blue): label below
    ax1.annotate(f"{vl:.3f}", (yr, vl), xytext=(0, -16),
                 textcoords="offset points", ha="center", fontsize=FSS - 1.5,
                 color=C_CLSTM, fontweight="bold")
    if yr > 5:
        # Matched CNN (open red): label further right and below to avoid the
        # CNN-actual label that sits above the matched point at 10yr/20yr.
        ax1.annotate(f"({vm:.3f})", (yr, vm), xytext=(10, -4),
                     textcoords="offset points", ha="left", fontsize=7,
                     color=C_CNN_M)

ax1.set_xticks(horizons)
ax1.set_xticklabels(["5-year\n(last obs: 2010\n24 channels)",
                     "10-year\n(last obs: 2005\n21 channels)",
                     "20-year\n(last obs: 1995\n15 channels)"],
                    fontsize=FSS - 1)
ax1.set_xlabel("Forecast horizon (years from last observation to 2015 target)", fontsize=FSS)
ax1.set_ylabel("Figure of Merit (2015 spatial holdout)", fontsize=FSS)
ax1.set_title("(a) FoM vs. forecast horizon\n"
              "Channel-matched CNN (open markers) isolates true horizon effect",
              fontsize=FS, fontweight="bold")
ax1.set_ylim(0.38, 0.88)
ax1.legend(fontsize=FSS - 2, loc="lower right",
           framealpha=0.9, edgecolor="#B0BEC5")
ax1.grid(alpha=0.20, linestyle="--")
ax1.set_axisbelow(True)
for sp in ["top", "right"]:
    ax1.spines[sp].set_visible(False)

# ── Panel (b): Gap decomposition ──────────────────────────────────────────────
labels  = ["ConvLSTM\ngenuine\nimprovement", "CNN\nchannel\ndisadvantage", "CNN\ntrue horizon\neffect"]
values  = [clstm_improvement, cnn_channel_effect, cnn_horizon_effect]
colors  = [C_CLSTM, C_CH, C_HOR]
pcts    = [100 * v / total_compression for v in values]
x_pos   = np.arange(len(labels))

bars = ax2.bar(x_pos, values, color=colors, width=0.55, zorder=3,
               edgecolor="white", linewidth=0.6)
for bar, v, pct in zip(bars, values, pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
             f"{v:.3f}\n({pct:.0f}\u202f%)",
             ha="center", va="bottom", fontsize=FSS - 1, fontweight="bold",
             color="#37474F")

ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=FSS - 1.5)
ax2.set_ylabel("FoM contribution to\n5yr\u2192 10yr gap compression", fontsize=FSS)
ax2.set_title("(b) Decomposition of 5yr\u2192 10yr gap compression\n"
              f"Total compression: 0.216 \u2192 0.079 (\u2212{total_compression:.3f})",
              fontsize=FS, fontweight="bold")
ax2.set_ylim(0, 0.115)
ax2.grid(axis="y", alpha=0.20, linestyle="--", zorder=0)
ax2.set_axisbelow(True)
for sp in ["top", "right"]:
    ax2.spines[sp].set_visible(False)

# legend for decomposition
patches = [
    mpatches.Patch(color=C_CLSTM, label="ConvLSTM genuine improvement"),
    mpatches.Patch(color=C_CH,    label="CNN information reduction (channel count)"),
    mpatches.Patch(color=C_HOR,   label="CNN true horizon effect"),
]
ax2.legend(handles=patches, fontsize=7, loc="upper right",
           framealpha=0.9, edgecolor="#B0BEC5")

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT / f"fig_multihorizon.{ext}"
    fig.savefig(str(path), bbox_inches="tight", dpi=220)
    print(f"Saved: {path}")
plt.close(fig)
print("Done.")
