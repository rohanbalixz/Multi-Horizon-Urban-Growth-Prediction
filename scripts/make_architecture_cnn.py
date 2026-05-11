#!/usr/bin/env python3
"""
Generate fig_architecture_cnn — SimpleCNN architecture diagram.
Same canvas / style as make_architecture_figure.py (ConvLSTM).

Layout (left → right):
  Input (flat stack) → Conv Block 1 (24→64) → Conv Block 2 (64→32→16)
  → Output Head (16→1) → Prediction → Results panel
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_INPUT = "#E3F2FD"
C_BLK1  = "#2E7D32"   # dark green
C_BLK2  = "#388E3C"   # medium green
C_HEAD  = "#1B5E20"   # deep green
C_OUT   = "#1B5E20"
C_RES   = "#1A237E"
C_EDGE  = "#37474F"
C_WHITE = "#FAFAFA"
C_DARK  = "#212121"


def box(ax, cx, cy, w, h, fc, ec=C_EDGE, lw=1.4, rad=0.05, zorder=3, alpha=1.0):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       zorder=zorder, alpha=alpha)
    ax.add_patch(p)


def txt(ax, x, y, s, color=C_DARK, fs=8.5, bold=False,
        ha="center", va="center", **kw):
    ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
            fontweight="bold" if bold else "normal", **kw)


def arrow(ax, x0, y0, x1, y1, color=C_EDGE, lw=1.6, style="-|>", shrink=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=shrink, shrinkB=shrink))


# ── canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5.0))
ax.set_xlim(0, 15)
ax.set_ylim(0, 5.0)
ax.axis("off")

# ── title ─────────────────────────────────────────────────────────────────────
txt(ax, 7.5, 4.75, "SimpleCNN Architecture",
    fs=12.5, bold=True, color=C_DARK)

# ──────────────────────────────────────────────────────────────────────────────
# INPUT BLOCK  (cx=1.05)
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
txt(ax, cx_in, 1.90, "[B, 24, 128, 128]",
    color="#546E7A", fs=6.5, style="italic")

# flat-stack annotation
txt(ax, cx_in, 1.55, "flat-stack: T×3 → 24 ch",
    color="#546E7A", fs=6.2, style="italic")

# ── arrow in → block 1 ───────────────────────────────────────────────────────
arrow(ax, cx_in + 0.85, 2.50, 2.55, 2.50, lw=1.8)

# ──────────────────────────────────────────────────────────────────────────────
# CONV BLOCK 1  (cx=3.35)   Conv(24→64) + BN + ReLU,  Conv(64→64) + BN + ReLU
# ──────────────────────────────────────────────────────────────────────────────
cx_b1 = 3.35
box(ax, cx_b1, 2.50, 1.50, 2.40, fc=C_BLK1, ec="#1B5E20", lw=1.6, rad=0.08)
txt(ax, cx_b1, 3.42, "Conv Block 1", color=C_WHITE, fs=10.0, bold=True)
txt(ax, cx_b1, 3.10, "Conv(24→64, 3×3)", color=C_WHITE, fs=7.8)
txt(ax, cx_b1, 2.86, "BatchNorm + ReLU", color="#A5D6A7", fs=7.2)
txt(ax, cx_b1, 2.62, "Conv(64→64, 3×3)", color=C_WHITE, fs=7.8)
txt(ax, cx_b1, 2.38, "BatchNorm + ReLU", color="#A5D6A7", fs=7.2)
txt(ax, cx_b1, 2.10, "[B, 64, 128, 128]", color="#C8E6C9", fs=6.5, style="italic")

# ── arrow block1 → block2 ────────────────────────────────────────────────────
arrow(ax, cx_b1 + 0.75, 2.50, 4.82, 2.50, lw=1.8)
txt(ax, 4.40, 2.72, "64 ch", color="#546E7A", fs=6.8, style="italic")

# ──────────────────────────────────────────────────────────────────────────────
# CONV BLOCK 2  (cx=5.60)   Conv(64→32) + BN + ReLU,  Conv(32→16) + BN + ReLU
# ──────────────────────────────────────────────────────────────────────────────
cx_b2 = 5.60
box(ax, cx_b2, 2.50, 1.50, 2.40, fc=C_BLK2, ec="#1B5E20", lw=1.6, rad=0.08)
txt(ax, cx_b2, 3.42, "Conv Block 2", color=C_WHITE, fs=10.0, bold=True)
txt(ax, cx_b2, 3.10, "Conv(64→32, 3×3)", color=C_WHITE, fs=7.8)
txt(ax, cx_b2, 2.86, "BatchNorm + ReLU", color="#A5D6A7", fs=7.2)
txt(ax, cx_b2, 2.62, "Conv(32→16, 3×3)", color=C_WHITE, fs=7.8)
txt(ax, cx_b2, 2.38, "BatchNorm + ReLU", color="#A5D6A7", fs=7.2)
txt(ax, cx_b2, 2.10, "[B, 16, 128, 128]", color="#C8E6C9", fs=6.5, style="italic")

# ── arrow block2 → head ──────────────────────────────────────────────────────
arrow(ax, cx_b2 + 0.75, 2.50, 7.12, 2.50, lw=1.8)
txt(ax, 6.75, 2.72, "16 ch", color="#546E7A", fs=6.8, style="italic")

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT HEAD  (cx=8.10)
# ──────────────────────────────────────────────────────────────────────────────
cx_head = 8.10
box(ax, cx_head, 2.50, 1.90, 2.30, fc=C_HEAD, ec="#004D40", lw=1.6, rad=0.08)
txt(ax, cx_head, 3.38, "Output Head", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_head, 3.08, "Conv(16→1, 1×1)", color=C_WHITE, fs=7.8)
txt(ax, cx_head, 2.80, "Sigmoid", color="#A5D6A7", fs=8.0)
txt(ax, cx_head, 2.54, "no pooling anywhere", color="#C8E6C9", fs=7.2)
txt(ax, cx_head, 2.30, "full spatial resolution", color="#C8E6C9", fs=7.2)
txt(ax, cx_head, 2.02, "[B, 1, 128, 128]",
    color="#A5D6A7", fs=6.5, style="italic")

# ── arrow head → prediction ──────────────────────────────────────────────────
arrow(ax, cx_head + 0.95, 2.50, 9.68, 2.50, lw=1.8, color=C_OUT)

# ──────────────────────────────────────────────────────────────────────────────
# PREDICTION OUTPUT  (cx=10.60)
# ──────────────────────────────────────────────────────────────────────────────
cx_pred = 10.60
box(ax, cx_pred, 2.50, 1.55, 1.50, fc=C_OUT, ec="#1B5E20", lw=1.6, rad=0.07)
txt(ax, cx_pred, 3.10, "Prediction", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_pred, 2.82, "1 × 128×128 px", color=C_WHITE, fs=7.8)
txt(ax, cx_pred, 2.57, "built-up density", color="#A5D6A7", fs=7.2)
txt(ax, cx_pred, 2.32, "∈ [0, 1]", color="#A5D6A7", fs=7.2)
txt(ax, cx_pred, 2.04, "deterministic", color="#A5D6A7", fs=7.0)

# ── arrow pred → results ─────────────────────────────────────────────────────
arrow(ax, cx_pred + 0.78, 2.50, 11.72, 2.55, lw=1.5, color=C_DARK, shrink=2)

# ──────────────────────────────────────────────────────────────────────────────
# RESULTS PANEL  (cx=13.10)
# ──────────────────────────────────────────────────────────────────────────────
cx_res = 13.10
box(ax, cx_res, 2.55, 2.65, 3.80, fc="#E8EAF6", ec=C_RES, lw=1.6, rad=0.08)
txt(ax, cx_res, 4.22, "Key Results", color=C_RES, fs=10.0, bold=True)
ax.plot([cx_res - 1.25, cx_res + 1.25], [4.00, 4.00],
        color=C_RES, lw=0.8, alpha=0.5, zorder=4)
txt(ax, cx_res, 3.72, "FoM 2015:  0.702 ± 0.019", color=C_DARK, fs=8.0)
txt(ax, cx_res, 3.46, "FoM 2020:  0.252 ± 0.009", color=C_DARK, fs=8.0)
txt(ax, cx_res, 3.20, "leads all models, all horizons", color=C_DARK, fs=7.5)
ax.plot([cx_res - 1.25, cx_res + 1.25], [3.00, 3.00],
        color=C_RES, lw=0.6, alpha=0.4, linestyle="--", zorder=4)
txt(ax, cx_res, 2.72, "Growth-px MSE:", color=C_DARK, fs=7.8)
txt(ax, cx_res, 2.50, "5.35×10⁻⁴  (64% < linear)", color=C_DARK, fs=7.5)
ax.plot([cx_res - 1.25, cx_res + 1.25], [2.28, 2.28],
        color=C_RES, lw=0.6, alpha=0.4, linestyle="--", zorder=4)
txt(ax, cx_res, 2.02, "Channel-equated flat encoder", color=C_DARK, fs=7.8)
txt(ax, cx_res, 1.78, "no uncertainty output", color="#555555", fs=7.0)
txt(ax, cx_res, 1.52, "single forward pass", color=C_RES, fs=7.5)

# ── parameter footer ──────────────────────────────────────────────────────────
txt(ax, 7.5, 0.30,
    "74,273 total parameters  |  Training: 73.6 min (seed 42)  |  "
    "Inference: single forward pass, no stochastic sampling",
    color="#546E7A", fs=7.8, style="italic")

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT / f"fig_architecture_cnn.{ext}"
    fig.savefig(str(path), bbox_inches="tight", dpi=220)
    print(f"Saved: {path}")
plt.close(fig)
print("Done.")
