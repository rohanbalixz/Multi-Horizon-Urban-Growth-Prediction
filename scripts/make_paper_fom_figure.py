#!/usr/bin/env python3
"""
fig_paper_fom_comparison — three-panel benchmark figure.

Panel (a): FoM bar chart, 2015 spatial holdout (error bars from 2 seeds)
Panel (b): FoM bar chart, 2020 blind holdout
Panel (c): Threshold sensitivity — LINE PLOT (log x-axis)
           Solid lines = 2015 holdout; dotted lines = 2020 holdout.
           Converts the unreadable grouped-bar design to a clean line plot.

All data from verified JSON / multiseed results.
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
MODELS = ["SimpleCNN", "U-Net", "ConvLSTM", "Linear", "SLEUTH", "Persistence"]
COLORS = {
    "SimpleCNN":   "#D32F2F",
    "U-Net":       "#7B1FA2",
    "ConvLSTM":    "#1565C0",
    "Linear":      "#388E3C",
    "SLEUTH":      "#F57C00",
    "Persistence": "#616161",
}

fom_2015 = {"SimpleCNN": 0.702, "U-Net": 0.680, "ConvLSTM": 0.508,
            "Linear": 0.375, "SLEUTH": 0.056, "Persistence": 0.000}
fom_2015_std = {"SimpleCNN": 0.019, "U-Net": 0.004, "ConvLSTM": 0.002,
                "Linear": None, "SLEUTH": None, "Persistence": None}

fom_2020 = {"SimpleCNN": 0.252, "U-Net": 0.229, "ConvLSTM": 0.156,
            "Linear": 0.053, "SLEUTH": 0.121, "Persistence": 0.000}
fom_2020_std = {"SimpleCNN": 0.009, "U-Net": None, "ConvLSTM": None,
                "Linear": None, "SLEUTH": None, "Persistence": None}

# Threshold sensitivity (from threshold_sensitivity.json)
thresholds = [0.005, 0.010, 0.020, 0.050]

fom_thresh_2015 = {
    "SimpleCNN": [0.6454, 0.6834, 0.6796, 0.5057],
    "U-Net":     [0.6239, 0.6714, 0.6724, 0.4621],
    "ConvLSTM":  [0.5013, 0.5078, 0.4159, 0.2418],
    "Linear":    [0.4313, 0.3751, 0.3016, 0.2779],
}
fom_thresh_2020 = {
    "SimpleCNN": [0.2917, 0.2432, 0.1702, 0.0832],
    "ConvLSTM":  [0.2347, 0.1556, 0.0838, 0.0367],
    "Linear":    [0.0641, 0.0530, 0.0377, 0.0193],
}

# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4.8))
fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.14, wspace=0.35)

ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

FS  = 10.5
FSS = 9.0

def draw_bar_panel(ax, fom_dict, std_dict, title, xlabel, xlim):
    models = list(fom_dict.keys())
    values = [fom_dict[m] for m in models]
    stds   = [std_dict.get(m) for m in models]
    clrs   = [COLORS[m] for m in models]
    ypos   = np.arange(len(models))

    bars = ax.barh(ypos, values, color=clrs, height=0.62, zorder=3,
                   edgecolor="#ECEFF1", linewidth=0.5)
    for bar, v, s in zip(bars, values, stds):
        if s is not None:
            ax.errorbar(v, bar.get_y() + bar.get_height() / 2,
                        xerr=s, fmt="none", ecolor="white", elinewidth=2,
                        capsize=4, zorder=5)
        lbl = f"{v:.3f}" + (f"±{s:.3f}" if s else "")
        ax.text(v + xlim * 0.012, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=8.5, fontweight="bold",
                color="#212121")

    ax.set_yticks(ypos)
    ax.set_yticklabels(models, fontsize=FSS)
    ax.set_xlabel(xlabel, fontsize=FSS)
    ax.set_title(title, fontsize=FS, fontweight="bold", pad=5)
    ax.set_xlim(0, xlim)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.18, linestyle="--", color="#90A4AE")
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#CFD8DC")
    ax.spines["bottom"].set_color("#CFD8DC")
    bars[0].set_edgecolor(clrs[0])
    bars[0].set_linewidth(2.0)

draw_bar_panel(ax1, fom_2015, fom_2015_std,
               "(a)  FoM \u2014 2015 Spatial Holdout\n"
               "[higher \u2192 better growth localisation]",
               "Figure of Merit  (FoM)", 0.92)
draw_bar_panel(ax2, fom_2020, fom_2020_std,
               "(b)  FoM \u2014 2020 Blind Holdout\n"
               "[sealed temporal test; 10.6\u202f% growth pixels]",
               "Figure of Merit  (FoM)", 0.40)

# ── Panel (c): threshold sensitivity line plot ────────────────────────────────
shown = {
    "SimpleCNN":  ("o", "-",   fom_thresh_2015["SimpleCNN"], fom_thresh_2020["SimpleCNN"]),
    "U-Net":      ("s", "--",  fom_thresh_2015["U-Net"],     None),
    "ConvLSTM":   ("^", "-.",  fom_thresh_2015["ConvLSTM"],  fom_thresh_2020["ConvLSTM"]),
    "Linear":     ("D", ":",   fom_thresh_2015["Linear"],    fom_thresh_2020["Linear"]),
}
for m, (mk, ls, v15, v20) in shown.items():
    c = COLORS[m]
    ax3.plot(thresholds, v15, ls=ls, color=c, lw=2.0, ms=7,
             marker=mk, label=f"{m} (2015)", zorder=4)
    if v20 is not None:
        ax3.plot(thresholds, v20, ls=ls, color=c, lw=1.6, ms=6,
                 marker=mk, alpha=0.55, zorder=4, label=f"{m} (2020)")

ax3.set_xscale("log")
ax3.set_xticks(thresholds)
ax3.set_xticklabels(["0.005", "0.01", "0.02", "0.05"], fontsize=FSS - 1)
ax3.set_xlabel("Growth threshold  $t$", fontsize=FSS)
ax3.set_ylabel("Figure of Merit  (FoM)", fontsize=FSS)
ax3.set_title("(c)  FoM vs. change threshold $t$\n"
              "Bold lines = 2015 holdout;  faint = 2020 holdout",
              fontsize=FS, fontweight="bold", pad=5)

legend_els = [mpatches.Patch(color=COLORS[m], label=m)
              for m in shown] + [
    plt.Line2D([0], [0], color="grey", lw=2.0, label="2015 holdout"),
    plt.Line2D([0], [0], color="grey", lw=1.6, alpha=0.55, label="2020 holdout"),
]
ax3.legend(handles=legend_els, fontsize=7.5, loc="upper right",
           framealpha=0.92, edgecolor="#CFD8DC")
ax3.grid(alpha=0.18, linestyle="--")
ax3.set_axisbelow(True)
for sp in ["top", "right"]:
    ax3.spines[sp].set_visible(False)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    p = OUT / f"fig_paper_fom_comparison.{ext}"
    fig.savefig(str(p), bbox_inches="tight", dpi=220)
    print(f"Saved: {p}")
plt.close(fig)
print("Done.")
