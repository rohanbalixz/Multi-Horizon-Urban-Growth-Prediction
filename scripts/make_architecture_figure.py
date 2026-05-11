#!/usr/bin/env python3
"""
Generate fig_architecture — professional ConvLSTM architecture diagram.

Layout (left → right):
  Input block → ConvLSTM L1 → ConvLSTM L2 → Decoder → Prediction (top)
                                                       → Uncertainty (bottom)
  Results panel on far right.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_INPUT  = "#E3F2FD"
C_CLSTM  = "#1565C0"
C_SKIP   = "#6A1B9A"
C_OUT    = "#1B5E20"
C_UNC    = "#BF360C"
C_RES    = "#1A237E"
C_EDGE   = "#37474F"
C_WHITE  = "#FAFAFA"
C_DARK   = "#212121"
C_LGREY  = "#ECEFF1"

def box(ax, cx, cy, w, h, fc, ec=C_EDGE, lw=1.4, rad=0.05, zorder=3, alpha=1.0):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       zorder=zorder, alpha=alpha)
    ax.add_patch(p)

def txt(ax, x, y, s, color=C_DARK, fs=8.5, bold=False, ha="center", va="center", **kw):
    ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
            fontweight="bold" if bold else "normal", **kw)

def arrow(ax, x0, y0, x1, y1, color=C_EDGE, lw=1.6, style="-|>", shrink=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=shrink, shrinkB=shrink))

# ── canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5.0))
ax.set_xlim(0, 15)
ax.set_ylim(0, 5.0)
ax.axis("off")

# ── title ────────────────────────────────────────────────────────────────────
txt(ax, 7.5, 4.75, "ConvLSTM Architecture",
    fs=12.5, bold=True, color=C_DARK)

# ──────────────────────────────────────────────────────────────────────────────
# INPUT BLOCK  (cx=1.0)
# ──────────────────────────────────────────────────────────────────────────────
cx_in = 1.05
box(ax, cx_in, 2.5, 1.70, 3.6, fc=C_INPUT, ec="#1565C0", lw=1.6, rad=0.07)
txt(ax, cx_in, 4.10, "Input", color="#0D47A1", fs=9.5, bold=True)
txt(ax, cx_in, 3.80, "T=8 epochs", color="#0D47A1", fs=8.0)
txt(ax, cx_in, 3.48, "1975–2010", color=C_DARK, fs=7.8)
txt(ax, cx_in, 3.18, "3 channels:", color=C_DARK, fs=7.4)
txt(ax, cx_in, 2.94, "• built-up density", color=C_DARK, fs=7.0)
txt(ax, cx_in, 2.73, "• building volume", color=C_DARK, fs=7.0)
txt(ax, cx_in, 2.52, "• population density", color=C_DARK, fs=7.0)
txt(ax, cx_in, 2.22, "128×128 px  |  250 m", color=C_DARK, fs=7.2)
txt(ax, cx_in, 1.90, "[T=8, C=3, H=128, W=128]",
    color="#546E7A", fs=6.5, style="italic")

# ── arrow in → L1 ────────────────────────────────────────────────────────────
arrow(ax, cx_in + 0.85, 2.50, 2.55, 2.50, lw=1.8)
txt(ax, 2.18, 2.72, "recurrent\nstep-by-step",
    color="#546E7A", fs=6.4, va="bottom")

# ──────────────────────────────────────────────────────────────────────────────
# CONVLSTM LAYER 1  (cx=3.30)
# ──────────────────────────────────────────────────────────────────────────────
cx_l1 = 3.30
box(ax, cx_l1, 2.50, 1.50, 2.20, fc=C_CLSTM, ec="#0D47A1", lw=1.6, rad=0.08)
txt(ax, cx_l1, 3.28, "ConvLSTM", color=C_WHITE, fs=10.0, bold=True)
txt(ax, cx_l1, 3.00, "Layer 1", color=C_WHITE, fs=9.0)
txt(ax, cx_l1, 2.70, "hidden = 64", color=C_WHITE, fs=7.8)
txt(ax, cx_l1, 2.48, "kernel 3×3", color=C_WHITE, fs=7.8)
txt(ax, cx_l1, 2.22, "MC Dropout  p=0.1", color="#90CAF9", fs=7.2)

# ── arrow L1 → L2 ────────────────────────────────────────────────────────────
arrow(ax, cx_l1 + 0.75, 2.50, 4.82, 2.50, lw=1.8)
txt(ax, 4.40, 2.72, "H₁, C₁", color="#546E7A", fs=6.8, style="italic")

# ──────────────────────────────────────────────────────────────────────────────
# CONVLSTM LAYER 2  (cx=5.60)
# ──────────────────────────────────────────────────────────────────────────────
cx_l2 = 5.60
box(ax, cx_l2, 2.50, 1.50, 2.20, fc=C_CLSTM, ec="#0D47A1", lw=1.6, rad=0.08)
txt(ax, cx_l2, 3.28, "ConvLSTM", color=C_WHITE, fs=10.0, bold=True)
txt(ax, cx_l2, 3.00, "Layer 2", color=C_WHITE, fs=9.0)
txt(ax, cx_l2, 2.70, "hidden = 64", color=C_WHITE, fs=7.8)
txt(ax, cx_l2, 2.48, "kernel 3×3", color=C_WHITE, fs=7.8)
txt(ax, cx_l2, 2.22, "MC Dropout  p=0.1", color="#90CAF9", fs=7.2)

# ── skip connection arc ───────────────────────────────────────────────────────
# from L1 top to decoder top
ax.annotate("", xy=(7.72, 3.85), xytext=(cx_l1, 3.62),
            arrowprops=dict(arrowstyle="-|>", color=C_SKIP, lw=1.5,
                            connectionstyle="arc3,rad=-0.30"))
txt(ax, 5.55, 4.38, "skip connection", color=C_SKIP, fs=7.5, bold=True)

# ── arrow L2 → Decoder ───────────────────────────────────────────────────────
arrow(ax, cx_l2 + 0.75, 2.50, 7.12, 2.50, lw=1.8)
txt(ax, 6.75, 2.72, "H₂, C₂\n+skip", color="#546E7A", fs=6.5, style="italic")

# ──────────────────────────────────────────────────────────────────────────────
# DECODER  (cx=8.15)
# ──────────────────────────────────────────────────────────────────────────────
cx_dec = 8.15
box(ax, cx_dec, 2.50, 1.95, 2.40, fc=C_SKIP, ec="#4A148C", lw=1.6, rad=0.08)
txt(ax, cx_dec, 3.40, "Skip-Connection", color=C_WHITE, fs=9.0, bold=True)
txt(ax, cx_dec, 3.15, "Decoder", color=C_WHITE, fs=9.0)
txt(ax, cx_dec, 2.82, "Conv(128→64) + BN + ReLU", color="#CE93D8", fs=7.0)
txt(ax, cx_dec, 2.60, "Conv(64→32)  + BN + ReLU", color="#CE93D8", fs=7.0)
txt(ax, cx_dec, 2.38, "Conv(32→16)  + BN + ReLU", color="#CE93D8", fs=7.0)
txt(ax, cx_dec, 2.16, "Conv(16→1)   + Sigmoid", color="#CE93D8", fs=7.0)

# ── arrows decoder → prediction and uncertainty ───────────────────────────────
arrow(ax, cx_dec + 0.98, 2.90, 9.68, 3.60, lw=1.8, color=C_OUT)
arrow(ax, cx_dec + 0.98, 2.10, 9.68, 1.40, lw=1.8, color=C_UNC)
txt(ax, 9.45, 3.28, "mean", color=C_OUT, fs=6.8)
txt(ax, 9.42, 1.72, "20 passes\n×MC Drop", color=C_UNC, fs=6.5, style="italic")

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION OUTPUT  (cx=10.60, cy=3.60)
# ──────────────────────────────────────────────────────────────────────────────
cx_pred = 10.60
box(ax, cx_pred, 3.60, 1.55, 1.30, fc=C_OUT, ec="#1B5E20", lw=1.6, rad=0.07)
txt(ax, cx_pred, 4.08, "Prediction", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_pred, 3.80, "1 × 128×128 px", color=C_WHITE, fs=7.8)
txt(ax, cx_pred, 3.55, "built-up density", color="#A5D6A7", fs=7.2)
txt(ax, cx_pred, 3.32, "∈ [0, 1]", color="#A5D6A7", fs=7.2)

# ──────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY OUTPUT  (cx=10.60, cy=1.40)
# ──────────────────────────────────────────────────────────────────────────────
cx_unc = 10.60
box(ax, cx_unc, 1.40, 1.55, 1.30, fc=C_UNC, ec="#7F0000", lw=1.6, rad=0.07)
txt(ax, cx_unc, 1.88, "Uncertainty", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_unc, 1.60, "MC Dropout: 20 passes", color=C_WHITE, fs=7.2)
txt(ax, cx_unc, 1.35, "per-pixel σ̂", color="#FFCCBC", fs=7.8)
txt(ax, cx_unc, 1.12, "calibration r=0.983", color="#FFCCBC", fs=7.0)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS PANEL  (cx=13.0)
# ──────────────────────────────────────────────────────────────────────────────
cx_res = 13.10
box(ax, cx_res, 2.55, 2.65, 3.80, fc="#E8EAF6", ec=C_RES, lw=1.6, rad=0.08)
txt(ax, cx_res, 4.22, "Key Results", color=C_RES, fs=10.0, bold=True)
# divider
ax.plot([cx_res - 1.25, cx_res + 1.25], [4.00, 4.00],
        color=C_RES, lw=0.8, alpha=0.5, zorder=4)
txt(ax, cx_res, 3.72, "FoM 2015:  0.508 ± 0.002", color=C_DARK, fs=8.0)
txt(ax, cx_res, 3.46, "FoM 2020:  0.156", color=C_DARK, fs=8.0)
txt(ax, cx_res, 3.20, "3× linear on FoM 2020", color=C_DARK, fs=7.5)
ax.plot([cx_res - 1.25, cx_res + 1.25], [3.00, 3.00],
        color=C_RES, lw=0.6, alpha=0.4, linestyle="--", zorder=4)
txt(ax, cx_res, 2.72, "Growth-px MSE:", color=C_DARK, fs=7.8)
txt(ax, cx_res, 2.50, "6.10×10⁻⁴  (59% < linear)", color=C_DARK, fs=7.5)
ax.plot([cx_res - 1.25, cx_res + 1.25], [2.28, 2.28],
        color=C_RES, lw=0.6, alpha=0.4, linestyle="--", zorder=4)
txt(ax, cx_res, 2.02, "Calibration r = 0.983", color=C_DARK, fs=7.8)
txt(ax, cx_res, 1.78, "(n=10 decile bins, 869K px)", color="#555555", fs=7.0)
txt(ax, cx_res, 1.52, "Only model with spatial σ̂", color=C_RES, fs=7.5)

# arrows from outputs to results panel
arrow(ax, cx_pred + 0.78, 3.60, cx_res - 1.33, 3.10, lw=1.5, color=C_DARK, shrink=2)
arrow(ax, cx_unc  + 0.78, 1.40, cx_res - 1.33, 2.00, lw=1.5, color=C_DARK, shrink=2)

# ── parameter footer ──────────────────────────────────────────────────────────
txt(ax, 7.5, 0.30,
    "481,153 total parameters  |  Training: ~24 hrs (seed 42)  |  "
    "Inference: 40 min CONUS (MC Dropout, 20 passes)",
    color="#546E7A", fs=7.8, style="italic")

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT / f"fig_architecture.{ext}"
    fig.savefig(str(path), bbox_inches="tight", dpi=220)
    print(f"Saved: {path}")
plt.close(fig)
print("Done.")
