#!/usr/bin/env python3
"""
Generate fig_architecture_unet — SimpleUNet architecture diagram.
Same canvas / style as make_architecture_figure.py (ConvLSTM).

Layout (left → right):
  Input → Enc1 (128px) → Enc2 (64px) → Bottleneck (32px)
        → Dec2 (64px)  → Dec1 (128px) → Output → Results panel

  Box heights vary with spatial resolution to show the U-shape visually.
  Skip connections rendered as arcs over the diagram.
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
C_ENC   = "#E65100"   # deep orange  — encoder path
C_BN    = "#BF360C"   # dark red     — bottleneck
C_DEC   = "#F57C00"   # amber        — decoder path
C_SKIP  = "#6A1B9A"   # purple       — skip connections (same as ConvLSTM)
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
txt(ax, 7.5, 4.75, "SimpleUNet Architecture",
    fs=12.5, bold=True, color=C_DARK)

# ── heights reflect spatial resolution ───────────────────────────────────────
H_128 = 2.80   # enc1, dec1
H_64  = 1.90   # enc2, dec2
H_32  = 1.20   # bottleneck
CY    = 2.50   # all blocks share same centre-y

# ── x positions ──────────────────────────────────────────────────────────────
cx_in  = 1.05
cx_e1  = 3.10   # enc1
cx_e2  = 4.90   # enc2
cx_bn  = 6.55   # bottleneck
cx_d2  = 8.20   # dec2
cx_d1  = 10.00  # dec1
cx_out = 11.60  # output head
cx_res = 13.30  # results panel

BW = 1.40   # block width

# ── INPUT ─────────────────────────────────────────────────────────────────────
box(ax, cx_in, CY, 1.70, 3.60, fc=C_INPUT, ec="#1565C0", lw=1.6, rad=0.07)
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
txt(ax, cx_in, 1.58, "flat-stack: T×3 → 24 ch",
    color="#546E7A", fs=6.2, style="italic")

arrow(ax, cx_in + 0.85, CY, cx_e1 - BW / 2 - 0.05, CY, lw=1.8)

# ── ENCODER 1  (128×128, 24→32) ───────────────────────────────────────────────
box(ax, cx_e1, CY, BW, H_128, fc=C_ENC, ec="#BF360C", lw=1.6, rad=0.08)
txt(ax, cx_e1, CY + H_128 / 2 - 0.25, "Enc 1", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_e1, CY + H_128 / 2 - 0.55, "128×128", color="#FFCCBC", fs=7.5)
txt(ax, cx_e1, CY + 0.18, "Conv(24→32)×2", color=C_WHITE, fs=7.2)
txt(ax, cx_e1, CY - 0.10, "BN + ReLU", color="#FFCCBC", fs=7.0)
txt(ax, cx_e1, CY - H_128 / 2 + 0.22, "[B,32,128,128]",
    color="#FFCCBC", fs=6.2, style="italic")

# pool1 arrow (down + right)
ax.annotate("", xy=(cx_e2 - BW / 2 - 0.05, CY),
            xytext=(cx_e1 + BW / 2, CY),
            arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.6,
                            shrinkA=3, shrinkB=3))
txt(ax, (cx_e1 + cx_e2) / 2, CY + 0.22,
    "MaxPool×2", color="#546E7A", fs=6.6, style="italic")

# ── ENCODER 2  (64×64, 32→64) ────────────────────────────────────────────────
box(ax, cx_e2, CY, BW, H_64, fc=C_ENC, ec="#BF360C", lw=1.6, rad=0.08)
txt(ax, cx_e2, CY + H_64 / 2 - 0.24, "Enc 2", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_e2, CY + H_64 / 2 - 0.52, "64×64", color="#FFCCBC", fs=7.5)
txt(ax, cx_e2, CY + 0.10, "Conv(32→64)×2", color=C_WHITE, fs=7.2)
txt(ax, cx_e2, CY - 0.18, "BN + ReLU", color="#FFCCBC", fs=7.0)
txt(ax, cx_e2, CY - H_64 / 2 + 0.22, "[B,64,64,64]",
    color="#FFCCBC", fs=6.2, style="italic")

arrow(ax, cx_e2 + BW / 2, CY, cx_bn - BW / 2 - 0.05, CY, lw=1.6)
txt(ax, (cx_e2 + cx_bn) / 2, CY + 0.22,
    "MaxPool×2", color="#546E7A", fs=6.6, style="italic")

# ── BOTTLENECK  (32×32, 64→128) ──────────────────────────────────────────────
box(ax, cx_bn, CY, BW, H_32, fc=C_BN, ec="#7F0000", lw=1.6, rad=0.08)
txt(ax, cx_bn, CY + H_32 / 2 - 0.24, "Bottleneck", color=C_WHITE, fs=8.5, bold=True)
txt(ax, cx_bn, CY + 0.04, "Conv(64→128)×2", color=C_WHITE, fs=7.0)
txt(ax, cx_bn, CY - 0.22, "BN + ReLU  32×32", color="#FFCCBC", fs=6.8)
txt(ax, cx_bn, CY - H_32 / 2 + 0.20, "[B,128,32,32]",
    color="#FFCCBC", fs=6.2, style="italic")

arrow(ax, cx_bn + BW / 2, CY, cx_d2 - BW / 2 - 0.05, CY, lw=1.6)
txt(ax, (cx_bn + cx_d2) / 2, CY + 0.22,
    "ConvT×2", color="#546E7A", fs=6.6, style="italic")

# ── DECODER 2  (64×64, 128+64→64) ────────────────────────────────────────────
box(ax, cx_d2, CY, BW, H_64, fc=C_DEC, ec="#E65100", lw=1.6, rad=0.08)
txt(ax, cx_d2, CY + H_64 / 2 - 0.24, "Dec 2", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_d2, CY + H_64 / 2 - 0.52, "64×64", color="#FFF9C4", fs=7.5)
txt(ax, cx_d2, CY + 0.10, "Conv(128→64)×2", color=C_WHITE, fs=7.2)
txt(ax, cx_d2, CY - 0.18, "BN + ReLU", color="#FFF9C4", fs=7.0)
txt(ax, cx_d2, CY - H_64 / 2 + 0.22, "[B,64,64,64]",
    color="#FFF9C4", fs=6.2, style="italic")

arrow(ax, cx_d2 + BW / 2, CY, cx_d1 - BW / 2 - 0.05, CY, lw=1.6)
txt(ax, (cx_d2 + cx_d1) / 2, CY + 0.22,
    "ConvT×2", color="#546E7A", fs=6.6, style="italic")

# ── DECODER 1  (128×128, 64+32→32) ───────────────────────────────────────────
box(ax, cx_d1, CY, BW, H_128, fc=C_DEC, ec="#E65100", lw=1.6, rad=0.08)
txt(ax, cx_d1, CY + H_128 / 2 - 0.25, "Dec 1", color=C_WHITE, fs=9.5, bold=True)
txt(ax, cx_d1, CY + H_128 / 2 - 0.55, "128×128", color="#FFF9C4", fs=7.5)
txt(ax, cx_d1, CY + 0.18, "Conv(64→32)×2", color=C_WHITE, fs=7.2)
txt(ax, cx_d1, CY - 0.10, "BN + ReLU", color="#FFF9C4", fs=7.0)
txt(ax, cx_d1, CY - H_128 / 2 + 0.22, "[B,32,128,128]",
    color="#FFF9C4", fs=6.2, style="italic")

# ── OUTPUT HEAD ───────────────────────────────────────────────────────────────
arrow(ax, cx_d1 + BW / 2, CY, cx_out - 0.72, CY, lw=1.8, color=C_OUT)
box(ax, cx_out, CY, 1.35, 1.60, fc=C_OUT, ec="#1B5E20", lw=1.6, rad=0.07)
txt(ax, cx_out, CY + 0.55, "Output", color=C_WHITE, fs=9.0, bold=True)
txt(ax, cx_out, CY + 0.25, "Conv(32→1, 1×1)", color=C_WHITE, fs=7.2)
txt(ax, cx_out, CY - 0.02, "Sigmoid", color="#A5D6A7", fs=7.8)
txt(ax, cx_out, CY - 0.30, "∈ [0, 1]", color="#A5D6A7", fs=7.2)
txt(ax, cx_out, CY - 0.58, "[B, 1, 128×128]",
    color="#A5D6A7", fs=6.2, style="italic")

arrow(ax, cx_out + 0.68, CY, cx_res - 1.33, CY + 0.10,
      lw=1.5, color=C_DARK, shrink=2)

# ── SKIP CONNECTION: Enc1 → Dec1 (large arc over diagram) ───────────────────
top_e1 = CY + H_128 / 2
top_d1 = CY + H_128 / 2
ax.annotate("", xy=(cx_d1, top_d1 + 0.05), xytext=(cx_e1, top_e1 + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=C_SKIP, lw=1.6,
                            connectionstyle="arc3,rad=-0.40"))
txt(ax, (cx_e1 + cx_d1) / 2, top_e1 + 0.68,
    "skip 1  (32 ch)", color=C_SKIP, fs=7.2, bold=True)

# ── SKIP CONNECTION: Enc2 → Dec2 (smaller arc) ───────────────────────────────
top_e2 = CY + H_64 / 2
top_d2 = CY + H_64 / 2
ax.annotate("", xy=(cx_d2, top_d2 + 0.05), xytext=(cx_e2, top_e2 + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=C_SKIP, lw=1.5,
                            connectionstyle="arc3,rad=-0.35"))
txt(ax, (cx_e2 + cx_d2) / 2, top_e2 + 0.50,
    "skip 2  (64 ch)", color=C_SKIP, fs=7.2, bold=True)

# ── RESULTS PANEL ─────────────────────────────────────────────────────────────
box(ax, cx_res, 2.55, 2.65, 3.80, fc="#E8EAF6", ec=C_RES, lw=1.6, rad=0.08)
txt(ax, cx_res, 4.22, "Key Results", color=C_RES, fs=10.0, bold=True)
ax.plot([cx_res - 1.25, cx_res + 1.25], [4.00, 4.00],
        color=C_RES, lw=0.8, alpha=0.5, zorder=4)
txt(ax, cx_res, 3.72, "FoM 2015:  0.680 ± 0.004", color=C_DARK, fs=8.0)
txt(ax, cx_res, 3.46, "FoM 2020:  0.229", color=C_DARK, fs=8.0)
txt(ax, cx_res, 3.20, "encoder-decoder skip baseline", color=C_DARK, fs=7.5)
ax.plot([cx_res - 1.25, cx_res + 1.25], [3.00, 3.00],
        color=C_RES, lw=0.6, alpha=0.4, linestyle="--", zorder=4)
txt(ax, cx_res, 2.72, "Growth-px MSE:", color=C_DARK, fs=7.8)
txt(ax, cx_res, 2.50, "5.70×10⁻⁴  (62% < linear)", color=C_DARK, fs=7.5)
ax.plot([cx_res - 1.25, cx_res + 1.25], [2.28, 2.28],
        color=C_RES, lw=0.6, alpha=0.4, linestyle="--", zorder=4)
txt(ax, cx_res, 2.02, "Spatial skip-connection encoder", color=C_DARK, fs=7.8)
txt(ax, cx_res, 1.78, "no uncertainty output", color="#555555", fs=7.0)
txt(ax, cx_res, 1.52, "473,857 parameters", color=C_RES, fs=7.5)

# ── parameter footer ──────────────────────────────────────────────────────────
txt(ax, 7.5, 0.30,
    "473,857 total parameters  |  Training: ~374 min (seed 42)  |  "
    "Inference: single forward pass, no stochastic sampling",
    color="#546E7A", fs=7.8, style="italic")

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT / f"fig_architecture_unet.{ext}"
    fig.savefig(str(path), bbox_inches="tight", dpi=220)
    print(f"Saved: {path}")
plt.close(fig)
print("Done.")
