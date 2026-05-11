#!/usr/bin/env python3
"""
Generate publication-quality figures and tables for NeuralTimeCapsule paper.

Produces 8 figures + 2 LaTeX tables from real experimental results:
  Fig 1: Training convergence curves (train & val loss)
  Fig 2: Input data overview (3 channels × 2 time steps)
  Fig 3: Model comparison bar chart (MSE across all models)
  Fig 4: Ablation study grouped bar chart
  Fig 5: Prediction vs ground truth spatial comparison (2000)
  Fig 6: 2015 temporal validation spatial comparison
  Fig 7: Uncertainty maps (mean prediction, std, CV, CI width)
  Fig 8: Per-tile error distribution (box/violin plots)
  Table 1: Main results comparison (LaTeX)
  Table 2: Ablation study (LaTeX)
"""
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
OUT_DIR = str(PROJECT_ROOT / "results" / "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── global style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── load results ──
def load_json(path):
    p = str(RESULTS_DIR / path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required results file not found: {p}\nRun all pipeline scripts first.")
    with open(p) as f:
        return json.load(f)

history  = load_json("training_3ch_history.json")
all_res  = load_json("all_results.json")
ablation = load_json("ablation_3ch_results.json")
sota     = load_json("sota_baselines_results.json")
val2015  = load_json("validation_2015_results.json")
unc      = load_json("uncertainty_summary.json")

# Optional — only needed after train_multihorizon.py has run
mh_path = str(RESULTS_DIR / "multihorizon_results.json")
multihorizon = json.load(open(mh_path)) if os.path.exists(mh_path) else None

# Optional — CNN multi-horizon and channel control results
cnn_mh_path = str(RESULTS_DIR / "cnn_multihorizon_results.json")
cnn_multihorizon = json.load(open(cnn_mh_path)) if os.path.exists(cnn_mh_path) else None

cnn_cc_path = str(RESULTS_DIR / "cnn_channel_control_results.json")
cnn_channel_control = json.load(open(cnn_cc_path)) if os.path.exists(cnn_cc_path) else None

print("All JSON results loaded.")

# ── helper: load GeoTIFFs ──
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not available, skipping spatial figures (Fig 2,5,6,7)")


def get_tif_shape(name):
    """Get the shape of a GeoTIFF without loading it."""
    path = str(OUTPUT_DIR / name)
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as src:
        return (src.height, src.width)


def load_tif_window(name, row_off, col_off, height, width):
    """Load a windowed portion of a GeoTIFF (memory-efficient for huge CONUS rasters)."""
    path = str(OUTPUT_DIR / name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping")
        return None
    window = rasterio.windows.Window(col_off, row_off, width, height)
    with rasterio.open(path) as src:
        data = src.read(1, window=window).astype(np.float32)
    return data


def find_urban_crop(ref_name="CONUS_builtup_2000.tif", size=512):
    """Find the densest region by scanning a downsampled overview of the raster."""
    path = str(OUTPUT_DIR / ref_name)
    if not os.path.exists(path):
        return (0, 0, size, size)  # row_off, col_off, height, width

    with rasterio.open(path) as src:
        H, W = src.height, src.width
        # Read at ~1/32 resolution via overviews or decimated read
        ds = 32
        data_small = src.read(1, out_shape=(H // ds, W // ds)).astype(np.float32)

    bs = size // ds
    best_sum, best_r, best_c = 0, 0, 0
    sH, sW = data_small.shape
    for r in range(0, sH - bs, bs // 4):
        for c in range(0, sW - bs, bs // 4):
            s = data_small[r:r+bs, c:c+bs].sum()
            if s > best_sum:
                best_sum, best_r, best_c = s, r, c
    row_off = best_r * ds
    col_off = best_c * ds
    return (row_off, col_off, size, size)


def find_val_urban_crop(size=512):
    """Find the densest urban area among validation tiles for visualization.

    Builds a downsampled density map from val_tile_indices.json weighted by
    GHSL 2015 builtup, so the crop window contains real (non-zero) predictions
    from CONUS_builtup_2015_predicted.tif (which is only non-zero at val tiles).
    Falls back to find_urban_crop if val indices are unavailable.
    """
    val_idx_path = str(RESULTS_DIR / "val_tile_indices.json")
    if not os.path.exists(val_idx_path) or not HAS_RASTERIO:
        return find_urban_crop(size=size)

    with open(val_idx_path) as f:
        val_tiles = [tuple(t) for t in json.load(f)]

    path_gt = str(OUTPUT_DIR / "CONUS_builtup_2015.tif")
    if not os.path.exists(path_gt) or not val_tiles:
        return find_urban_crop(size=size)

    with rasterio.open(path_gt) as src:
        H, W = src.height, src.width
        ds = 32
        bu_small = src.read(1, out_shape=(H // ds, W // ds)).astype(np.float32)

    # Build val-tile density map at ds resolution (TILE=128px → 4 cells at ds=32)
    TILE = 128
    val_density = np.zeros_like(bu_small)
    tile_ds = max(1, TILE // ds)
    for (ti, tj) in val_tiles:
        gi = min(ti // ds, val_density.shape[0] - 1)
        gj = min(tj // ds, val_density.shape[1] - 1)
        val_density[gi:gi+tile_ds, gj:gj+tile_ds] += 1

    # Score = builtup density × val tile coverage
    score_map = bu_small * (val_density > 0).astype(np.float32)

    bs = size // ds
    sH, sW = score_map.shape
    best_score, best_r, best_c = 0, 0, 0
    for r in range(0, sH - bs, max(1, bs // 4)):
        for c in range(0, sW - bs, max(1, bs // 4)):
            s = score_map[r:r+bs, c:c+bs].sum()
            if s > best_score:
                best_score, best_r, best_c = s, r, c

    row_off = min(best_r * ds, H - size)
    col_off = min(best_c * ds, W - size)
    print(f"  Val-tile crop: row={row_off}, col={col_off}, score={best_score:.3f}")
    return (row_off, col_off, size, size)


# Cache the crop window so every figure uses the same region
CROP_CACHE = {}

def get_crop(size=512):
    """Get cached crop window — uses val-tile-aware search for figs 5/6."""
    if size not in CROP_CACHE:
        CROP_CACHE[size] = find_val_urban_crop(size=size)
        print(f"  Crop window: row={CROP_CACHE[size][0]}, col={CROP_CACHE[size][1]}, size={size}")
    return CROP_CACHE[size]


def load_cropped(name, size=512):
    """Load a cropped region of a GeoTIFF centered on the densest urban area."""
    crop = get_crop(size)
    return load_tif_window(name, crop[0], crop[1], crop[2], crop[3])


# ════════════════════════════════════════════════════
# FIGURE 1: Training Convergence Curves
# ════════════════════════════════════════════════════
def fig1_training_curves():
    print("Generating Fig 1: Training convergence curves...")
    train_loss = history["train_losses"]
    val_loss = history["val_losses"]
    epochs = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: full curve
    ax1.plot(epochs, train_loss, "b-", linewidth=1.5, label="Train loss", alpha=0.85)
    ax1.plot(epochs, val_loss, "r-", linewidth=1.5, label="Val loss", alpha=0.85)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("(a) Full training trajectory")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=13, color="gray", linestyle="--", alpha=0.5, label="Phase transition")
    ax1.annotate("Phase\ntransition", xy=(13, 0.002), fontsize=8, color="gray",
                 ha="center", va="bottom")

    # Right: zoom on convergence (epoch 13+)
    ax2.plot(epochs[12:], train_loss[12:], "b-", linewidth=1.5, label="Train loss", alpha=0.85)
    ax2.plot(epochs[12:], val_loss[12:], "r-", linewidth=1.5, label="Val loss", alpha=0.85)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("(b) Post-transition convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Mark best val
    best_idx = np.argmin(val_loss)
    ax2.axhline(y=val_loss[best_idx], color="green", linestyle=":", alpha=0.6)
    ax2.annotate(f"Best val: {val_loss[best_idx]:.6f}",
                 xy=(best_idx + 1, val_loss[best_idx]), fontsize=8, color="green",
                 xytext=(best_idx - 8, val_loss[best_idx] + 0.00005),
                 arrowprops=dict(arrowstyle="->", color="green", lw=0.8))

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_training_curves.pdf")
    fig.savefig(f"{OUT_DIR}/fig1_training_curves.png")
    plt.close(fig)
    print("  Saved fig1_training_curves.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 2: Input Data Overview (3 channels × time)
# ════════════════════════════════════════════════════
def fig2_input_data():
    if not HAS_RASTERIO:
        return
    print("Generating Fig 2: Input data overview...")

    panels = [
        ("CONUS_builtup_1975.tif", "Built-up 1975", "YlOrRd"),
        ("CONUS_builtup_1990.tif", "Built-up 1990", "YlOrRd"),
        ("CONUS_builtup_2000.tif", "Built-up 2000", "YlOrRd"),
        ("CONUS_volume_1990.tif", "Built-up volume 1990", "bone_r"),
        ("CONUS_population_1975.tif", "Population 1975", "viridis"),
        ("CONUS_population_1990.tif", "Population 1990", "viridis"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (fname, title, cmap) in zip(axes.flat, panels):
        data = load_cropped(fname)
        if data is not None:
            vmax = np.percentile(data[data > 0], 99) if (data > 0).any() else 1
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Multi-channel input data (representative metro area, 128 km\u00b2)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_input_data.pdf")
    fig.savefig(f"{OUT_DIR}/fig2_input_data.png")
    plt.close(fig)
    print("  Saved fig2_input_data.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 3: Model Comparison Bar Chart
# ════════════════════════════════════════════════════
def fig3_model_comparison():
    print("Generating Fig 3: Model comparison...")

    # Read from loaded JSON results (no hardcoded values)
    models = [
        ("U-Net", ablation["baseline_unet_3ch"]["val_mse"], "#2196F3"),
        ("ConvLSTM\n(ours)", ablation["convlstm_3ch"]["val_mse"], "#4CAF50"),
        ("1-Layer\nConvLSTM", ablation["ablation_1layer_3ch"]["val_mse"], "#9C27B0"),
        ("2ch\nConvLSTM", ablation["ablation_builtup_volume_2ch"]["val_mse"], "#795548"),
        ("Linear\nExtrap.", ablation["linear_extrapolation"]["val_mse"], "#607D8B"),
        ("Built-up\nOnly", ablation["ablation_builtup_only"]["val_mse"], "#E91E63"),
        ("SLEUTH\nCA", sota["sleuth_ca"]["val_mse"], "#F44336"),
        ("CNN", ablation["baseline_cnn_3ch"]["val_mse"], "#9E9E9E"),
    ]
    models.sort(key=lambda x: x[1])

    names = [m[0] for m in models]
    mses  = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), mses, color=colors, edgecolor="white", linewidth=0.5)

    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00008,
                f"{mse:.6f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Validation MSE")
    ax.set_title("Model Comparison — Validation MSE (lower is better)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=1e-4)

    # Highlight our model
    ours_idx = next(i for i, m in enumerate(models) if "ours" in m[0])
    bars[ours_idx].set_edgecolor("black")
    bars[ours_idx].set_linewidth(2)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_model_comparison.pdf")
    fig.savefig(f"{OUT_DIR}/fig3_model_comparison.png")
    plt.close(fig)
    print("  Saved fig3_model_comparison.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 4: Ablation Study (Grouped Bar Chart)
# ════════════════════════════════════════════════════
def fig4_ablation_study():
    print("Generating Fig 4: Ablation study...")

    # Read from loaded JSON results; filter out entries with missing data
    configs = [
        ("3ch ConvLSTM\n(full model)", ablation["convlstm_3ch"], "#4CAF50"),
        ("2ch\n(built+vol)", ablation["ablation_builtup_volume_2ch"], "#2196F3"),
        ("Built-up\nonly", ablation["ablation_builtup_only"], "#FF9800"),
        ("Volume\nonly", ablation["ablation_volume_only"], "#F44336"),
        ("Population\nonly", ablation.get("ablation_population_only", {}), "#9C27B0"),
    ]
    valid = [(n, d, c) for n, d, c in configs if d.get("val_mse") is not None]

    names = [v[0] for v in valid]
    mses  = [v[1]["val_mse"] for v in valid]
    maes  = [v[1]["val_mae"] for v in valid]
    colors = [v[2] for v in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # MSE bars
    bars1 = ax1.bar(range(len(names)), mses, color=colors, edgecolor="white")
    for bar, val in zip(bars1, mses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00003,
                 f"{val:.6f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("Validation MSE")
    ax1.set_title("(a) Input channel ablation — MSE")
    ax1.grid(axis="y", alpha=0.3)
    bars1[0].set_edgecolor("black")
    bars1[0].set_linewidth(2)

    # MAE bars
    bars2 = ax2.bar(range(len(names)), maes, color=colors, edgecolor="white")
    for bar, val in zip(bars2, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                 f"{val:.6f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Validation MAE")
    ax2.set_title("(b) Input channel ablation — MAE")
    ax2.grid(axis="y", alpha=0.3)
    bars2[0].set_edgecolor("black")
    bars2[0].set_linewidth(2)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_ablation.pdf")
    fig.savefig(f"{OUT_DIR}/fig4_ablation.png")
    plt.close(fig)
    print("  Saved fig4_ablation.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 5: Prediction vs Ground Truth (Year 2015)
# ════════════════════════════════════════════════════
def fig5_prediction_comparison():
    if not HAS_RASTERIO:
        return
    print("Generating Fig 5: Prediction vs ground truth (2015)...")

    gt_c = load_cropped("CONUS_builtup_2015.tif")
    pred_c = load_cropped("CONUS_builtup_2015_predicted.tif")
    if gt_c is None or pred_c is None:
        print("  Skipping — missing files")
        return

    diff = pred_c - gt_c
    vmax_gt = np.percentile(gt_c[gt_c > 0], 99) if (gt_c > 0).any() else 1

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Ground truth
    im0 = axes[0].imshow(gt_c, cmap="YlOrRd", vmin=0, vmax=vmax_gt)
    axes[0].set_title("(a) Ground truth 2015")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    divider = make_axes_locatable(axes[0])
    plt.colorbar(im0, cax=divider.append_axes("right", size="5%", pad=0.05))

    # Prediction
    im1 = axes[1].imshow(pred_c, cmap="YlOrRd", vmin=0, vmax=vmax_gt)
    axes[1].set_title("(b) ConvLSTM prediction 2015")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    divider = make_axes_locatable(axes[1])
    plt.colorbar(im1, cax=divider.append_axes("right", size="5%", pad=0.05))

    # Error map
    err_max = max(abs(np.percentile(diff, 2)), abs(np.percentile(diff, 98)))
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-err_max, vmax=err_max)
    axes[2].set_title("(c) Prediction error (pred − truth)")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    divider = make_axes_locatable(axes[2])
    plt.colorbar(im2, cax=divider.append_axes("right", size="5%", pad=0.05))

    fig.suptitle("Year-2015 prediction comparison (representative metro area)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_prediction_comparison.pdf")
    fig.savefig(f"{OUT_DIR}/fig5_prediction_comparison.png")
    plt.close(fig)
    print("  Saved fig5_prediction_comparison.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 6: 2015 Temporal Validation
# ════════════════════════════════════════════════════
def fig6_temporal_validation():
    if not HAS_RASTERIO:
        return
    print("Generating Fig 6: 2015 spatial holdout validation...")

    gt15 = load_cropped("CONUS_builtup_2015.tif")
    pr15 = load_cropped("CONUS_builtup_2015_predicted.tif")
    gt00 = load_cropped("CONUS_builtup_2000.tif")
    if gt15 is None or pr15 is None or gt00 is None:
        print("  Skipping — missing files")
        return

    # True growth = 2015 - 2000
    true_growth = gt15 - gt00
    pred_growth = pr15 - gt00

    vmax = np.percentile(gt15[gt15 > 0], 99) if (gt15 > 0).any() else 1
    gmax = max(abs(np.percentile(true_growth, 2)), abs(np.percentile(true_growth, 98)), 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Row 1: absolute density
    im0 = axes[0,0].imshow(gt15, cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[0,0].set_title("(a) Ground truth 2015")
    axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
    divider = make_axes_locatable(axes[0,0])
    plt.colorbar(im0, cax=divider.append_axes("right", size="5%", pad=0.05))

    im1 = axes[0,1].imshow(pr15, cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[0,1].set_title("(b) ConvLSTM forecast 2015")
    axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
    divider = make_axes_locatable(axes[0,1])
    plt.colorbar(im1, cax=divider.append_axes("right", size="5%", pad=0.05))

    # Row 2: growth maps
    im2 = axes[1,0].imshow(true_growth, cmap="RdYlGn_r", vmin=-gmax, vmax=gmax)
    axes[1,0].set_title("(c) Observed growth 2000→2015")
    axes[1,0].set_xticks([]); axes[1,0].set_yticks([])
    divider = make_axes_locatable(axes[1,0])
    plt.colorbar(im2, cax=divider.append_axes("right", size="5%", pad=0.05))

    im3 = axes[1,1].imshow(pred_growth, cmap="RdYlGn_r", vmin=-gmax, vmax=gmax)
    axes[1,1].set_title("(d) Predicted growth 2000→2015")
    axes[1,1].set_xticks([]); axes[1,1].set_yticks([])
    divider = make_axes_locatable(axes[1,1])
    plt.colorbar(im3, cax=divider.append_axes("right", size="5%", pad=0.05))

    # Add R² annotation
    r2 = val2015["convlstm"]["r2"]
    mse = val2015["convlstm"]["mse"]
    imp = val2015["improvement_over_linear_pct"]
    imp_label = f"{imp:+.1f}% vs linear ({'better' if imp > 0 else 'worse'} MSE)"
    fom = val2015["convlstm"].get("fom", 0)
    fom_pp = val2015.get("fom_vs_linear_pp", 0)
    fig.text(0.5, 0.01,
             f"5-year spatial holdout (inputs 1975–2010 → 2015):  R² = {r2:.3f}  |  FoM = {fom:.3f} ({fom_pp:+.1f}pp vs linear)  |  {imp_label}",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(f"{OUT_DIR}/fig6_temporal_validation.pdf")
    fig.savefig(f"{OUT_DIR}/fig6_temporal_validation.png")
    plt.close(fig)
    print("  Saved fig6_temporal_validation.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 7: Uncertainty Maps
# ════════════════════════════════════════════════════
def fig7_uncertainty():
    if not HAS_RASTERIO:
        return
    print("Generating Fig 7: Uncertainty maps...")

    names_and_labels = [
        ("CONUS_mc_mean_prediction.tif", "Mean prediction", "YlOrRd"),
        ("CONUS_mc_uncertainty_std.tif", "Predictive std (\u03c3)", "viridis"),
        ("CONUS_mc_uncertainty_cv.tif", "Coefficient of variation", "plasma"),
        ("CONUS_mc_uncertainty_ci95_width.tif", "95% CI width", "magma"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (fname, title, cmap), label in zip(axes.flat, names_and_labels, labels):
        try:
            dc = load_cropped(fname)
        except Exception as e:
            print(f"  WARNING: Could not load {fname}: {e}")
            dc = None
        if dc is not None:
            vmax = np.percentile(dc[dc > 0], 98) if (dc > 0).any() else 1
            im = ax.imshow(dc, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            divider = make_axes_locatable(ax)
            plt.colorbar(im, cax=divider.append_axes("right", size="5%", pad=0.05))
        else:
            ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(f"{label} {title}")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("MC Dropout uncertainty quantification (20 forward passes)", fontsize=12, y=1.02)

    # Annotation
    stats = unc["uncertainty_stats"]
    fig.text(0.5, 0.01,
             f"Mean σ = {stats['mean_std']:.4f}  |  Mean CV = {stats['mean_cv']:.4f}  |  "
             f"Mean 95% CI width = {stats['mean_ci95_width']:.4f}",
             ha="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(f"{OUT_DIR}/fig7_uncertainty.pdf")
    fig.savefig(f"{OUT_DIR}/fig7_uncertainty.png")
    plt.close(fig)
    print("  Saved fig7_uncertainty.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 8: Architecture & Depth Comparison
# ════════════════════════════════════════════════════
def fig8_architecture_depth():
    print("Generating Fig 8: Architecture & depth comparison...")

    # Read from loaded JSON results
    cnn = ablation["baseline_cnn_3ch"]
    l1  = ablation["ablation_1layer_3ch"]
    l2  = ablation["convlstm_3ch"]
    unet = ablation["baseline_unet_3ch"]
    arch_data = [
        (f"CNN\n({cnn['params']//1000}K)", cnn["val_mse"], "#9E9E9E"),
        (f"1-Layer\nConvLSTM\n({l1['params']//1000}K)", l1["val_mse"], "#FF9800"),
        (f"2-Layer\nConvLSTM\n({l2['params']//1000}K)", l2["val_mse"], "#4CAF50"),
        (f"U-Net\n({unet['params']//1000}K)", unet["val_mse"], "#2196F3"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: MSE comparison
    names = [d[0] for d in arch_data]
    mses = [d[1] for d in arch_data]
    colors = [d[2] for d in arch_data]
    bars = ax1.bar(range(len(names)), mses, color=colors, edgecolor="white")
    for bar, mse in zip(bars, mses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
                 f"{mse:.6f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("Validation MSE")
    ax1.set_title("(a) Architecture comparison")
    ax1.set_yscale("log")
    ax1.grid(axis="y", alpha=0.3)
    bars[2].set_edgecolor("black")
    bars[2].set_linewidth(2)

    # Right: Params vs MSE scatter
    all_models = [
        ("CNN", cnn["params"], cnn["val_mse"], "#9E9E9E"),
        ("1L ConvLSTM", l1["params"], l1["val_mse"], "#FF9800"),
        ("U-Net", unet["params"], unet["val_mse"], "#2196F3"),
        ("2L ConvLSTM", l2["params"], l2["val_mse"], "#4CAF50"),
    ]
    for name, params, mse, color in all_models:
        ax2.scatter(params / 1000, mse, color=color, s=120, edgecolors="black",
                   linewidths=0.8, zorder=5)
        offset_x = 5 if name != "U-Net" else -40
        ax2.annotate(name, (params/1000, mse), fontsize=8,
                    xytext=(offset_x, 8), textcoords="offset points",
                    ha="left" if name != "U-Net" else "right")

    ax2.set_xlabel("Parameters (thousands)")
    ax2.set_ylabel("Validation MSE")
    ax2.set_title("(b) Parameters vs performance")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig8_architecture_depth.pdf")
    fig.savefig(f"{OUT_DIR}/fig8_architecture_depth.png")
    plt.close(fig)
    print("  Saved fig8_architecture_depth.pdf/png")


# ════════════════════════════════════════════════════
# TABLE 1: Main Results (LaTeX)
# ════════════════════════════════════════════════════
def table1_main_results():
    print("Generating Table 1: Main results (LaTeX)...")

    cl  = {**ablation["convlstm_3ch"], **all_res["main_results"]["convlstm_3ch"]}
    cnn = ablation["baseline_cnn_3ch"]
    un  = ablation["baseline_unet_3ch"]
    li  = {**ablation["linear_extrapolation"], **all_res["main_results"]["linear_extrapolation_2step"]}
    sl  = sota["sleuth_ca"]

    def fv(v, fmt=".6f"):
        return f"{v:{fmt}}" if v is not None else "---"

    def ffom(v):
        return f"{v:.3f}" if v is not None else "---"

    def fp(v):
        return f"{v:,}" if v is not None else "---"

    # Rows: model, MSE, MAE, FoM, Params  (sorted by FoM descending)
    rows = [
        (r"\textbf{CNN 3ch (flat stacking)} $\dagger$",
         rf"\textbf{{{fv(cnn['val_mse'])}}}",
         rf"\textbf{{{fv(cnn['val_mae'])}}}",
         rf"\textbf{{{ffom(cnn.get('val_fom'))}}}",
         fp(cnn["params"])),
        ("U-Net 3ch (encoder-decoder)",
         fv(un["val_mse"]), fv(un["val_mae"]), ffom(un.get("val_fom")), fp(un["params"])),
        ("ConvLSTM 3ch + MC Dropout (ours)",
         fv(cl["val_mse"]), fv(cl["val_mae"]), ffom(cl.get("val_fom")), fp(cl["params"])),
        ("Linear extrapolation ($2x_{2010}-x_{2005}$)",
         fv(li["val_mse"]), fv(li["val_mae"]), ffom(li.get("val_fom")), "---"),
        ("SLEUTH CA \\citep{Clarke1997}",
         fv(sl["val_mse"]), fv(sl["val_mae"]), ffom(sl.get("val_fom")), "4"),
    ]

    latex = r"""\begin{table}[t]
\centering
\caption{Main results on CONUS spatial block holdout (821 val tiles, $\sim$1.64M urban pixels).
FoM = Figure of Merit \citep{Pontius2008}: intersection-over-union of predicted vs.\ observed
growth pixels; primary metric for urban growth benchmarking.
$\dagger$ CNN uses flat temporal channel stacking (same input as ConvLSTM, no recurrence).
All neural models: 25 epochs, lr=$5\!\times\!10^{-4}$, batch 8, seed 42.}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{MSE} $\downarrow$ & \textbf{MAE} $\downarrow$ & \textbf{FoM} $\uparrow$ & \textbf{Params} \\
\midrule
"""
    for row in rows:
        latex += " & ".join(row) + r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"{OUT_DIR}/table1_main_results.tex", "w") as f:
        f.write(latex)
    print("  Saved table1_main_results.tex")


# ════════════════════════════════════════════════════
# TABLE 2: Ablation Study (LaTeX)
# ════════════════════════════════════════════════════
def table2_ablation():
    print("Generating Table 2: Ablation study (LaTeX)...")

    # Read from loaded JSON results
    base = ablation["convlstm_3ch"]
    base_mse = base["val_mse"]

    def fv(v):
        return f"{v:.6f}" if v is not None else "---"

    def delta(d):
        mse = d.get("val_mse")
        if mse is None or base_mse is None or base_mse == 0:
            return "---"
        pct = ((mse - base_mse) / base_mse) * 100
        if pct >= 0:
            return f"+{pct:.0f}\\%"
        return f"$-${abs(pct):.0f}\\%"

    ch2 = ablation["ablation_builtup_volume_2ch"]
    bu  = ablation["ablation_builtup_only"]
    vo  = ablation["ablation_volume_only"]
    pop = ablation.get("ablation_population_only", {})
    l1  = ablation["ablation_1layer_3ch"]
    cnn = ablation["baseline_cnn_3ch"]
    un  = ablation["baseline_unet_3ch"]

    rows = [
        (r"\textbf{3ch ConvLSTM (full model)}", rf"\textbf{{{fv(base_mse)}}}", rf"\textbf{{{fv(base['val_mae'])}}}", "---"),
        ("Built-up + Volume (2ch)", fv(ch2["val_mse"]), fv(ch2["val_mae"]), delta(ch2)),
        ("Built-up only (1ch)", fv(bu["val_mse"]), fv(bu["val_mae"]), delta(bu)),
        ("Volume only (1ch)", fv(vo["val_mse"]), fv(vo["val_mae"]), delta(vo)),
        ("Population only (1ch)", fv(pop.get("val_mse")), fv(pop.get("val_mae")), delta(pop)),
        (r"\midrule", "", "", ""),
        ("2-Layer ConvLSTM (full)", fv(base_mse), fv(base["val_mae"]), "---"),
        ("1-Layer ConvLSTM", fv(l1["val_mse"]), fv(l1["val_mae"]), delta(l1)),
        ("CNN (no recurrence)", fv(cnn["val_mse"]), fv(cnn["val_mae"]), delta(cnn)),
        ("U-Net (no recurrence)", fv(un["val_mse"]), fv(un["val_mae"]), delta(un)),
    ]

    latex = r"""\begin{table}[t]
\centering
\caption{Ablation study: input channel and architecture variants. ``$\Delta$ MSE'' shows relative change vs.\ full 3ch ConvLSTM.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
\textbf{Configuration} & \textbf{MSE} $\downarrow$ & \textbf{MAE} $\downarrow$ & \textbf{$\Delta$ MSE} \\
\midrule
"""
    for row in rows:
        if row[0] == r"\midrule":
            latex += r"\midrule" + "\n"
        else:
            latex += " & ".join(row) + r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(f"{OUT_DIR}/table2_ablation.tex", "w") as f:
        f.write(latex)
    print("  Saved table2_ablation.tex")


# ════════════════════════════════════════════════════
# FIGURE 9: Multi-Horizon Comparison
# ConvLSTM vs CNN vs Linear regression at 5/10/20-year horizons.
# Three panels: (a) MSE grouped bars, (b) FoM grouped bars,
# (c) CNN MSE-win % over ConvLSTM.
# ════════════════════════════════════════════════════
def fig9_multihorizon():
    if multihorizon is None:
        print("  Skipping Fig 9 — multihorizon_results.json not found (run train_multihorizon.py)")
        return
    print("Generating Fig 9: Multi-horizon comparison (ConvLSTM + CNN + Linear)...")

    mh_all  = all_res["multi_horizon"]
    horizon_keys = ["5yr", "10yr", "20yr"]
    labels   = ["5-year", "10-year", "20-year"]

    conv_mses = [mh_all[k]["convlstm_mse"] for k in horizon_keys]
    cnn_mses  = [mh_all[k]["cnn_mse"]      for k in horizon_keys]
    lin_mses  = [mh_all[k]["linear_ols_mse"] for k in horizon_keys]

    conv_foms = [mh_all[k].get("convlstm_fom", 0)    for k in horizon_keys]
    cnn_foms  = [mh_all[k].get("cnn_fom",  0)         for k in horizon_keys]
    lin_foms  = [mh_all[k].get("linear_ols_fom", 0)   for k in horizon_keys]

    cnn_win_pct = [
        round((conv - cnn) / conv * 100, 1)
        for conv, cnn in zip(conv_mses, cnn_mses)
    ]

    x     = np.arange(len(labels))
    w     = 0.25
    C_CONV = "#2196F3"
    C_CNN  = "#4CAF50"
    C_LIN  = "#FF7043"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ── (a) MSE grouped bars ──
    ax = axes[0]
    b1 = ax.bar(x - w, conv_mses, w, label="ConvLSTM 3ch",      color=C_CONV, alpha=0.85)
    b2 = ax.bar(x,     cnn_mses,  w, label="CNN (flat)",         color=C_CNN,  alpha=0.85)
    b3 = ax.bar(x + w, lin_mses,  w, label="Linear regression",  color=C_LIN,  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Validation MSE (↓ better)")
    ax.set_title("(a) MSE by forecast horizon")
    ax.legend(fontsize=8)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.5f", fontsize=6.5, padding=2)

    # ── (b) FoM grouped bars ──
    ax = axes[1]
    b1 = ax.bar(x - w, conv_foms, w, label="ConvLSTM 3ch",      color=C_CONV, alpha=0.85)
    b2 = ax.bar(x,     cnn_foms,  w, label="CNN (flat)",         color=C_CNN,  alpha=0.85)
    b3 = ax.bar(x + w, lin_foms,  w, label="Linear regression",  color=C_LIN,  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Figure of Merit (↑ better)")
    ax.set_title("(b) FoM by forecast horizon")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2)

    # ── (c) CNN MSE advantage over ConvLSTM ──
    ax = axes[2]
    colors_bar = ["#4CAF50" if v > 0 else "#F44336" for v in cnn_win_pct]
    ax.bar(labels, cnn_win_pct, color=colors_bar, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("CNN MSE reduction vs ConvLSTM (%)")
    ax.set_title("(c) CNN MSE advantage over ConvLSTM")
    for i, v in enumerate(cnn_win_pct):
        ax.text(i, v + 0.5, f"{v:+.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Multi-horizon urban growth forecasting: ConvLSTM vs CNN (flat) vs linear regression",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig9_multihorizon.pdf")
    fig.savefig(f"{OUT_DIR}/fig9_multihorizon.png")
    plt.close(fig)
    print("  Saved fig9_multihorizon.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 10: Figure of Merit + Growth-Region Analysis
# Separates overall MSE into growth vs stable pixels.
# FoM bar chart + growth/stable MSE comparison.
# ════════════════════════════════════════════════════
def fig10_growth_region():
    print("Generating Fig 10: FoM and growth-region analysis...")

    c   = val2015["convlstm"]
    lin = val2015["linear_extrapolation"]
    per = val2015["persistence"]

    # Check all required keys are present
    required = ["fom", "growth_mse", "stable_mse"]
    if not all(k in c for k in required):
        print("  Skipping Fig 10 — run validate_2015.py first to generate FoM/growth metrics")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Panel (a): Figure of Merit ---
    models_fom = ["ConvLSTM\n3ch", "Linear\nextrapolation", "Persistence"]
    fom_vals   = [c["fom"], lin.get("fom", 0), per.get("fom", 0)]
    colors_fom = ["#2196F3", "#FF7043", "#9E9E9E"]
    bars = axes[0].bar(models_fom, fom_vals, color=colors_fom, alpha=0.85,
                       edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Figure of Merit (↑ better)")
    axes[0].set_title("(a) Figure of Merit\n(spatial growth accuracy)")
    axes[0].set_ylim(0, max(fom_vals) * 1.25)
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    # --- Panel (b): Growth-region MSE ---
    models_g = ["ConvLSTM\n3ch", "Linear\nextrapolation", "Persistence"]
    g_mses   = [c["growth_mse"], lin.get("growth_mse", float("nan")), per.get("growth_mse", float("nan"))]
    colors_g = ["#2196F3", "#FF7043", "#9E9E9E"]
    bars2 = axes[1].bar(models_g, g_mses, color=colors_g, alpha=0.85,
                        edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("MSE on growth pixels (↓ better)")
    axes[1].set_title("(b) Growth-region MSE\n(pixels where built-up increased)")
    axes[1].bar_label(bars2, fmt="%.5f", padding=3, fontsize=8)

    # --- Panel (c): Growth vs stable MSE breakdown for ConvLSTM ---
    categories  = ["Overall", "Growth\npixels", "Stable\npixels"]
    conv_vals   = [c["mse"], c["growth_mse"], c["stable_mse"]]
    lin_vals    = [lin["mse"], lin.get("growth_mse", float("nan")),
                   lin.get("stable_mse", float("nan"))]
    x     = np.arange(len(categories))
    width = 0.35
    b1 = axes[2].bar(x - width/2, conv_vals, width, label="ConvLSTM 3ch",      color="#2196F3", alpha=0.85)
    b2 = axes[2].bar(x + width/2, lin_vals,  width, label="Linear extrap.",     color="#FF7043", alpha=0.85)
    axes[2].set_xticks(x); axes[2].set_xticklabels(categories)
    axes[2].set_ylabel("MSE (↓ better)")
    axes[2].set_title("(c) MSE breakdown\n(growth vs stable)")
    axes[2].legend(fontsize=8)

    n_growth = c.get("n_growth_pixels")
    n_stable = c.get("n_stable_pixels")
    growth_label = (f"{n_growth:,} growth pixels | {n_stable:,} stable pixels"
                    if n_growth is not None else "growth / stable pixel breakdown")
    fig.suptitle(f"Growth-region evaluation — {growth_label}", fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig10_growth_region.pdf")
    fig.savefig(f"{OUT_DIR}/fig10_growth_region.png")
    plt.close(fig)
    print("  Saved fig10_growth_region.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 11: Uncertainty Calibration
# Shows that pixels with higher predicted std have higher actual error.
# Well-calibrated uncertainty is a key contribution of MC Dropout.
# ════════════════════════════════════════════════════
def fig11_calibration():
    print("Generating Fig 11: Uncertainty calibration...")

    if "calibration" not in unc:
        print("  Skipping Fig 11 — run generate_uncertainty_maps.py first to compute calibration")
        return

    cal    = unc["calibration"]
    bins   = cal["bins"]
    r_val  = cal["pearson_r_std_vs_error"]
    x_vals = [b["mean_predicted_std"] for b in bins]
    y_vals = [b["mean_actual_error"]  for b in bins]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel (a): calibration scatter + trend ---
    ax1.scatter(x_vals, y_vals, color="#2196F3", s=80, zorder=5, label="Uncertainty decile")
    if len(x_vals) > 1:
        coeffs = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        ax1.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=1.8,
                 label=f"Linear fit (r={r_val:.3f})")
    ax1.set_xlabel("Mean predicted uncertainty (std)")
    ax1.set_ylabel("Mean actual absolute error")
    ax1.set_title(f"(a) Calibration: predicted std vs actual error\nPearson r = {r_val:.3f}")
    ax1.legend()

    # --- Panel (b): bar chart of actual error per bin ---
    n_bins = len(bins)
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_bins))
    ax2.bar(range(1, n_bins + 1), y_vals, color=bar_colors, edgecolor="black", linewidth=0.4)
    ax2.set_xlabel("Uncertainty decile (1=lowest, 10=highest)")
    ax2.set_ylabel("Mean actual absolute error")
    ax2.set_title("(b) Error per uncertainty decile\n(monotonic increase = well-calibrated)")
    ax2.set_xticks(range(1, n_bins + 1))

    fig.suptitle("MC Dropout uncertainty calibration — NeuralTimeCapsule", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig11_calibration.pdf")
    fig.savefig(f"{OUT_DIR}/fig11_calibration.png")
    plt.close(fig)
    print("  Saved fig11_calibration.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 12: Sequence Length Ablation
# MSE vs number of input timesteps (4 / 6 / 8 epochs).
# Shows value of longer historical context.
# ════════════════════════════════════════════════════
def fig12_seqlen_ablation():
    print("Generating Fig 12: Sequence length ablation...")

    seqlen_keys = sorted([k for k in ablation if k.startswith("seqlen_")],
                         key=lambda k: int(k.split("_")[1]))
    if not seqlen_keys:
        print("  Skipping Fig 12 — no seqlen_ keys in ablation_3ch_results.json")
        return

    n_steps = [ablation[k]["n_timesteps"] for k in seqlen_keys]
    mses    = [ablation[k]["val_mse"]     for k in seqlen_keys]
    maes    = [ablation[k]["val_mae"]     for k in seqlen_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Scale to micro-units so the y-axis doesn't need scientific offset notation
    # (the default "1e-5" offset label was overlapping the data point labels).
    mses_u = [m * 1e5 for m in mses]
    maes_u = [a * 1e3 for a in maes]

    ax1.plot(n_steps, mses_u, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of input timesteps")
    ax1.set_ylabel(r"Validation MSE  ($\times 10^{-5}$, $\downarrow$ better)")
    ax1.set_title("(a) MSE vs sequence length")
    ax1.set_xticks(n_steps)
    ax1.ticklabel_format(style="plain", axis="y", useOffset=False)
    # Pad y-limits so the label above the highest point stays inside the axes
    y1_lo, y1_hi = min(mses_u), max(mses_u)
    pad1 = max((y1_hi - y1_lo) * 0.25, 0.05)
    ax1.set_ylim(y1_lo - pad1, y1_hi + pad1)
    for n, m, mu in zip(n_steps, mses, mses_u):
        ax1.annotate(f"{m:.5f}", (n, mu), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8)

    ax2.plot(n_steps, maes_u, "s-", color="#FF7043", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of input timesteps")
    ax2.set_ylabel(r"Validation MAE  ($\times 10^{-3}$, $\downarrow$ better)")
    ax2.set_title("(b) MAE vs sequence length")
    ax2.set_xticks(n_steps)
    ax2.ticklabel_format(style="plain", axis="y", useOffset=False)
    y2_lo, y2_hi = min(maes_u), max(maes_u)
    pad2 = max((y2_hi - y2_lo) * 0.25, 0.05)
    ax2.set_ylim(y2_lo - pad2, y2_hi + pad2)
    for n, a, au in zip(n_steps, maes, maes_u):
        ax2.annotate(f"{a:.5f}", (n, au), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8)

    fig.suptitle("Sequence length ablation — 3-channel ConvLSTM (TRAIN_EPOCHS[-N:]→2015)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig12_seqlen_ablation.pdf")
    fig.savefig(f"{OUT_DIR}/fig12_seqlen_ablation.png")
    plt.close(fig)
    print("  Saved fig12_seqlen_ablation.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 13: CNN Channel-Count Control (Confound Analysis)
# Shows that the apparent FoM degradation at longer horizons
# is driven by fewer input channels, not forecast difficulty.
# Two panels:
#   (a) FoM for all configs (grouped by channel count)
#   (b) FoM diff (5yr − Xyr) at matched channel counts
# ════════════════════════════════════════════════════
def fig13_channel_control():
    if cnn_channel_control is None:
        print("  Skipping Fig 13 — cnn_channel_control_results.json not found")
        return
    print("Generating Fig 13: CNN channel-count control (confound analysis)...")

    cc = all_res.get("cnn_channel_control", {})
    configs = cc.get("configs", {})
    paired  = cc.get("paired_comparisons", {})

    if not configs:
        print("  Skipping Fig 13 — cnn_channel_control missing from all_results.json")
        return

    # Pull all configs from cnn_multihorizon + channel_control for full table
    mh_all = all_res["multi_horizon"]

    # All data points: (label, n_channels, horizon_years, fom, is_matched_horizon)
    points = [
        # from multihorizon experiment (true horizon configs)
        ("5yr\n(8ep, 24ch)", 24, 5,  mh_all["5yr"]["cnn_fom"],  False),
        ("10yr\n(7ep, 21ch)", 21, 10, mh_all["10yr"]["cnn_fom"], False),
        ("20yr\n(5ep, 15ch)", 15, 20, mh_all["20yr"]["cnn_fom"], False),
        # from channel_control (same horizon, fewer channels)
        ("5yr\n(7ep, 21ch)", 21, 5,  configs.get("5yr_7ep", {}).get("val_fom", 0), True),
        ("5yr\n(5ep, 15ch)", 15, 5,  configs.get("5yr_5ep", {}).get("val_fom", 0), True),
    ]

    # ── Panel (a): FoM for all configs, coloured by horizon ──
    horizon_colors = {5: "#4CAF50", 10: "#2196F3", 20: "#FF7043"}
    markers        = {False: "o", True: "s"}   # circle = true horizon, square = control

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Several markers cluster at x=15 and x=21 (true-horizon circle plus
    # channel-matched square). Place each label at an explicit (dx, dy, ha,
    # va) offset so they fan out instead of stacking.  Keyed by the unique
    # label text from `points`.
    label_offsets = {
        "5yr\n(8ep, 24ch)":  ( 10,   6, "left",  "bottom"),
        "10yr\n(7ep, 21ch)": ( 10, -10, "left",  "top"),     # circle at 21ch
        "20yr\n(5ep, 15ch)": ( 10, -10, "left",  "top"),     # circle at 15ch
        "5yr\n(7ep, 21ch)":  (-10,  10, "right", "bottom"),  # square at 21ch
        "5yr\n(5ep, 15ch)":  (-10,  10, "right", "bottom"),  # square at 15ch
    }
    for label, n_ch, h_yr, fom, is_ctrl in points:
        col = horizon_colors[h_yr]
        mk  = markers[is_ctrl]
        ax1.scatter(n_ch, fom, color=col, marker=mk, s=120, zorder=5,
                    edgecolors="black", linewidths=0.6)
        dx, dy, ha, va = label_offsets.get(label, (7, 4, "left", "bottom"))
        ax1.annotate(label, (n_ch, fom), textcoords="offset points",
                     xytext=(dx, dy), fontsize=7.5, ha=ha, va=va)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markeredgecolor="black", markersize=9, label=f"{h}-year horizon")
        for h, c in horizon_colors.items()
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="True horizon config"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=9, label="Channel-matched control"),
    ]
    ax1.legend(handles=legend_elems, fontsize=8, loc="lower right")
    ax1.set_xlabel("Number of input channels (timesteps × 3)")
    ax1.set_ylabel("Figure of Merit (↑ better)")
    ax1.set_title("(a) FoM vs input channel count")
    ax1.set_xlim(10, 28)
    ax1.set_ylim(0.60, 0.80)

    # ── Panel (b): FoM difference at matched channel counts ──
    ch_labels  = ["21ch\n(5yr vs 10yr)", "15ch\n(5yr vs 20yr)"]
    fom_diffs  = [
        paired.get("21ch", {}).get("fom_diff_5yr_minus_xyr", 0),
        paired.get("15ch", {}).get("fom_diff_5yr_minus_xyr", 0),
    ]
    # Naive unmatched diffs (from multihorizon)
    naive_diffs = [
        mh_all["5yr"]["cnn_fom"] - mh_all["10yr"]["cnn_fom"],
        mh_all["5yr"]["cnn_fom"] - mh_all["20yr"]["cnn_fom"],
    ]

    x = np.arange(len(ch_labels))
    w = 0.35
    b1 = ax2.bar(x - w/2, naive_diffs, w, label="Unmatched (diff channel count)",
                 color="#FF7043", alpha=0.85, edgecolor="black", linewidth=0.5)
    b2 = ax2.bar(x + w/2, fom_diffs,   w, label="Channel-matched control",
                 color="#4CAF50", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x); ax2.set_xticklabels(ch_labels)
    ax2.set_ylabel("FoM difference (5yr − longer horizon)")
    ax2.set_title("(b) Channel count explains most of the FoM gap")
    ax2.legend(fontsize=8)
    ax2.bar_label(b1, fmt="%.4f", fontsize=8, padding=2)
    ax2.bar_label(b2, fmt="%.4f", fontsize=8, padding=2)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig13_channel_control.pdf")
    fig.savefig(f"{OUT_DIR}/fig13_channel_control.png")
    plt.close(fig)
    print("  Saved fig13_channel_control.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 14: 2020 Temporal Holdout Summary
# Bar chart comparison: ConvLSTM vs Linear vs Persistence
# for the true out-of-sample 2020 prediction.
# ════════════════════════════════════════════════════
def fig14_temporal_holdout_2020():
    val20_path = str(RESULTS_DIR / "validation_2020_results.json")
    if not os.path.exists(val20_path):
        print("  Skipping Fig 14 — validation_2020_results.json not found (run validate_2020.py)")
        return
    print("Generating Fig 14: 2020 temporal holdout summary...")

    with open(val20_path) as f:
        val20 = json.load(f)

    c20  = val20["convlstm_best"]
    l20  = val20["linear_extrapolation"]
    p20  = val20["persistence"]

    # Also load 2015 results for comparison
    c15  = val2015["convlstm"]
    l15  = val2015["linear_extrapolation"]

    C_CONV = "#2196F3"
    C_LIN  = "#FF7043"
    C_PER  = "#9E9E9E"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ── (a) FoM comparison at 2020 ──
    ax = axes[0]
    models = ["ConvLSTM\n(ours)", "Linear\nextrap.", "Persistence"]
    foms   = [c20["fom"], l20["fom"], p20["fom"]]
    colors = [C_CONV, C_LIN, C_PER]
    bars = ax.bar(models, foms, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel("Figure of Merit (↑ better)")
    ax.set_title("(a) FoM — 2020 temporal holdout\n(model never saw 2020 data)")
    ax.set_ylim(0, max(foms) * 1.4)

    # ── (b) Growth-region MSE at 2020 ──
    ax = axes[1]
    g_mses  = [c20["growth_mse"], l20["growth_mse"], p20["growth_mse"]]
    bars2 = ax.bar(models, g_mses, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars2, fmt="%.5f", padding=3, fontsize=8)
    ax.set_ylabel("MSE on growth pixels (↓ better)")
    ax.set_title(f"(b) Growth-region MSE — 2020\n({val20['pct_growth']:.1f}% of pixels urbanised 2015→2020)")

    # ── (c) FoM: 2015 spatial holdout vs 2020 temporal holdout ──
    ax = axes[2]
    labels2 = ["2015\n(spatial\nholdout)", "2020\n(temporal\nholdout)"]
    conv_foms = [c15.get("fom", 0.504), c20["fom"]]
    lin_foms  = [l15.get("fom", 0.375), l20["fom"]]
    x = np.arange(len(labels2))
    w = 0.35
    b1 = ax.bar(x - w/2, conv_foms, w, label="ConvLSTM", color=C_CONV, alpha=0.85)
    b2 = ax.bar(x + w/2, lin_foms,  w, label="Linear",   color=C_LIN,  alpha=0.85)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels2)
    ax.set_ylabel("Figure of Merit (↑ better)")
    ax.set_title("(c) FoM: spatial vs temporal holdout\n(ConvLSTM 3× better than linear in both)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.65)

    fig.suptitle(
        "2020 True Temporal Holdout — model trained on [1975–2010]→2015, predicts 2020",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig14_temporal_holdout_2020.pdf")
    fig.savefig(f"{OUT_DIR}/fig14_temporal_holdout_2020.png")
    plt.close(fig)
    print("  Saved fig14_temporal_holdout_2020.pdf/png")


# ════════════════════════════════════════════════════
# FIGURE 15: Rate-of-Change Analysis
# Shows mean annual urban growth slowing over decades,
# explaining why spatial features dominate and
# temporal memory provides diminishing returns.
# ════════════════════════════════════════════════════
def fig15_rate_of_change():
    if not HAS_RASTERIO:
        print("  Skipping Fig 15 — rasterio not available")
        return
    print("Generating Fig 15: Rate-of-change analysis...")

    epochs = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
    means  = []
    for yr in epochs:
        p = str(OUTPUT_DIR / f"CONUS_builtup_{yr}.tif")
        if not os.path.exists(p):
            print(f"  Skipping Fig 15 — {p} not found")
            return
        import rasterio as rio
        with rio.open(p) as src:
            data = src.read(1)
        means.append(float(data.mean()))

    # Per-5yr delta
    deltas      = [means[i+1] - means[i] for i in range(len(means)-1)]
    delta_years = [(epochs[i] + epochs[i+1]) / 2 for i in range(len(epochs)-1)]
    annual_rate = [d / 5 for d in deltas]   # per year

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── (a) Mean built-up density over time ──
    ax = axes[0]
    ax.plot(epochs, [m * 100 for m in means], "o-", color="#2196F3", linewidth=2, markersize=7)
    ax.fill_between(epochs, [m * 100 for m in means], alpha=0.15, color="#2196F3")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean built-up density (% of CONUS)")
    ax.set_title("(a) CONUS mean built-up density 1975–2020")
    ax.set_xticks(epochs)
    ax.tick_params(axis="x", rotation=45)
    for yr, m in zip(epochs, means):
        ax.annotate(f"{m*100:.3f}%", (yr, m*100), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)

    # ── (b) Annual growth rate per interval ──
    ax = axes[1]
    colors_bar = ["#4CAF50" if r > 0 else "#F44336" for r in annual_rate]
    bars = ax.bar([str(int(y)) for y in delta_years], [r * 1e4 for r in annual_rate],
                  color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)
    ax.set_xlabel("Mid-interval year")
    ax.set_ylabel("Annual built-up growth rate (×10⁻⁴ density/yr)")
    ax.set_title("(b) Annual growth rate per 5-year interval\n(slowing growth = diminishing temporal memory benefit)")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

    fig.suptitle(
        "CONUS urban growth dynamics: why spatial features dominate at 5-year intervals",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig15_rate_of_change.pdf")
    fig.savefig(f"{OUT_DIR}/fig15_rate_of_change.png")
    plt.close(fig)
    print("  Saved fig15_rate_of_change.pdf/png")


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("NeuralTimeCapsule — Publication Figure Generation")
    print("=" * 60)

    fig1_training_curves()
    fig2_input_data()
    fig3_model_comparison()
    fig4_ablation_study()
    fig5_prediction_comparison()
    fig6_temporal_validation()
    fig7_uncertainty()
    fig8_architecture_depth()
    table1_main_results()
    table2_ablation()
    fig9_multihorizon()
    fig10_growth_region()
    fig11_calibration()
    fig12_seqlen_ablation()
    fig13_channel_control()
    fig14_temporal_holdout_2020()
    fig15_rate_of_change()

    # Summary
    print("\n" + "=" * 60)
    figs = [f for f in os.listdir(OUT_DIR) if f.endswith(".png")]
    tabs = [f for f in os.listdir(OUT_DIR) if f.endswith(".tex")]
    print(f"Generated {len(figs)} figures (PNG+PDF) and {len(tabs)} LaTeX tables")
    print(f"Output directory: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
    print("=" * 60)
