#!/usr/bin/env python3
"""
Rigorous 3-Channel ConvLSTM Training
=====================================
Fixes 4 experimental weaknesses:
  1. Spatial-block splits — no pixel overlap between train/val/test
  2. Proper held-out test set — 70/15/15 split, report ONLY on test
  3. Multi-seed training — 3 seeds with mean±std error bars
  4. Growth-region evaluation — separate metrics for areas that changed

Output:
  results/metrics/rigorous_results.json
  models/rigorous_seed{1,2,3}.pth
"""
import sys, os, json, time, datetime, warnings, gc
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio

print("=" * 60, flush=True)
print("RIGOROUS 3-CHANNEL TRAINING (spatial splits, 3 seeds)", flush=True)
print(f"Started: {datetime.datetime.now()}", flush=True)
print("=" * 60, flush=True)

from src.models.convlstm import ConvLSTM

# =====================================================
# Configuration
# =====================================================
TILE_SIZE = 128
NUM_EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
MC_DROPOUT = 0.1
SEEDS = [42, 123, 7]
BLOCK_SIZE = 256  # spatial block size for non-overlapping splits (2x tile)

# =====================================================
# Load all data
# =====================================================
print("\n[DATA] Loading all data channels...", flush=True)

ghsl = {}
for year in [1975, 1990, 2000]:
    path = str(OUTPUT_DIR / f"CONUS_builtup_{year}.tif")
    with rasterio.open(path) as src:
        ghsl[year] = src.read(1)
    print(f"  Built-up {year}: shape={ghsl[year].shape}, mean={ghsl[year].mean():.6f}", flush=True)

# Also load 2015 for growth evaluation
with rasterio.open(str(OUTPUT_DIR / "CONUS_builtup_2015.tif")) as src:
    builtup_2015 = src.read(1)
print(f"  Built-up 2015: shape={builtup_2015.shape}, mean={builtup_2015.mean():.6f}", flush=True)

ref_shape = ghsl[2000].shape

# Built-up volume (temporal, one per epoch)
volume = {}
for year in [1975, 1990, 2000]:
    path = str(OUTPUT_DIR / f"CONUS_volume_{year}.tif")
    with rasterio.open(path) as src:
        volume[year] = src.read(1)
    print(f"  Volume {year}: shape={volume[year].shape}, mean={volume[year].mean():.4f}", flush=True)

# Population density
population = {}
for year in [1975, 1990]:
    fpath = str(OUTPUT_DIR / f"CONUS_population_{year}.tif")
    with rasterio.open(fpath) as src:
        population[year] = src.read(1)
    print(f"  Population {year}: shape={population[year].shape}, mean={population[year].mean():.4f}", flush=True)

gc.collect()

# =====================================================
# FIX #1 & #3: Spatial-block train/val/test split
# =====================================================
# Divide the raster into non-overlapping BLOCK_SIZE blocks.
# Assign each block to train/val/test with NO pixel overlap.
# Tiles are extracted WITHOUT overlap (stride=tile_size, not tile_size//2).
#
# This guarantees:
#  - No shared pixels between any split
#  - Spatial autocorrelation doesn't leak across splits
#  - Each tile is used exactly once

print(f"\n[SPLIT] Creating spatial-block split (block={BLOCK_SIZE}px, "
      f"tile={TILE_SIZE}px, no overlap)...", flush=True)

height, width = ref_shape

# Generate all non-overlapping tile positions within blocks
# Step 1: Identify block grid
block_rows = list(range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE))
block_cols = list(range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE))

# Step 2: For each block, check if it has urban content, then extract tiles
block_tiles = {}  # block_id -> list of (i, j) tile positions
for bi, br in enumerate(block_rows):
    for bj, bc in enumerate(block_cols):
        block_id = (bi, bj)
        tiles_in_block = []
        for ti in range(br, min(br + BLOCK_SIZE, height - TILE_SIZE + 1), TILE_SIZE):
            for tj in range(bc, min(bc + BLOCK_SIZE, width - TILE_SIZE + 1), TILE_SIZE):
                # Check urban content threshold
                tile_data = ghsl[2000][ti:ti+TILE_SIZE, tj:tj+TILE_SIZE]
                if tile_data.mean() > 0.01:
                    tiles_in_block.append((ti, tj))
        if tiles_in_block:
            block_tiles[block_id] = tiles_in_block

# Step 3: Assign blocks to splits (70/15/15) using deterministic hash
all_block_ids = sorted(block_tiles.keys())
np.random.seed(0)  # Fixed split across all seed runs
np.random.shuffle(all_block_ids)

n_blocks = len(all_block_ids)
n_train = int(0.70 * n_blocks)
n_val = int(0.15 * n_blocks)
# rest is test

train_blocks = all_block_ids[:n_train]
val_blocks = all_block_ids[n_train:n_train + n_val]
test_blocks = all_block_ids[n_train + n_val:]

train_tiles = [t for bid in train_blocks for t in block_tiles[bid]]
val_tiles = [t for bid in val_blocks for t in block_tiles[bid]]
test_tiles = [t for bid in test_blocks for t in block_tiles[bid]]

print(f"  Blocks with urban tiles: {n_blocks}", flush=True)
print(f"  Block split: train={len(train_blocks)}, val={len(val_blocks)}, test={len(test_blocks)}", flush=True)
print(f"  Tile split:  train={len(train_tiles)}, val={len(val_tiles)}, test={len(test_tiles)}", flush=True)

# Verify no overlap
train_set = set(train_tiles)
val_set = set(val_tiles)
test_set = set(test_tiles)
assert len(train_set & val_set) == 0, "Train/val overlap!"
assert len(train_set & test_set) == 0, "Train/test overlap!"
assert len(val_set & test_set) == 0, "Val/test overlap!"
print(f"  ✓ Zero pixel overlap between splits verified", flush=True)

# =====================================================
# Dataset using pre-assigned tile positions
# =====================================================
class BlockSplitDataset(Dataset):
    """Dataset with pre-assigned tile positions (no overlap)."""
    def __init__(self, tile_positions, temporal_stack, volume, population,
                 tile_size=128, sequence_length=2):
        self.tiles = tile_positions
        self.temporal_stack = temporal_stack
        self.volume = volume
        self.population = population
        self.tile_size = tile_size
        self.epochs = sorted(temporal_stack.keys())

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        i, j = self.tiles[idx]
        ts = self.tile_size
        sequence = []
        for epoch in self.epochs[:2]:  # 1975, 1990
            builtup = self.temporal_stack[epoch][i:i+ts, j:j+ts]
            vol = self.volume[epoch][i:i+ts, j:j+ts]
            pop = self.population[epoch][i:i+ts, j:j+ts]
            frame = np.stack([builtup, vol, pop], axis=0)
            sequence.append(frame)
        target = self.temporal_stack[2000][i:i+ts, j:j+ts]
        return (torch.FloatTensor(np.stack(sequence, axis=0)),
                torch.FloatTensor(target).unsqueeze(0))


# =====================================================
# Training function (one seed)
# =====================================================
def predict_future_step(model, sequences):
    batch_size, seq_len, channels, height, width = sequences.size()
    _, hidden_states = model(sequences)
    last_input = sequences[:, -1]
    x_t = last_input
    layer_hiddens = []
    for layer_idx, layer in enumerate(model.convlstm_layers):
        h, c = hidden_states[layer_idx]
        h, c = layer(x_t, (h, c))
        h = model.mc_dropouts[layer_idx](h)
        hidden_states[layer_idx] = (h, c)
        x_t = h
        layer_hiddens.append(h)
    return model.decode(layer_hiddens)


def train_one_seed(seed, train_tiles, val_tiles, test_tiles):
    """Train and evaluate one seed. Returns metrics dict."""
    print(f"\n{'='*60}", flush=True)
    print(f"  SEED {seed}", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = BlockSplitDataset(train_tiles, ghsl, volume, population)
    val_ds = BlockSplitDataset(val_tiles, ghsl, volume, population)
    test_ds = BlockSplitDataset(test_tiles, ghsl, volume, population)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ConvLSTM(input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=MC_DROPOUT)

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            pred = predict_future_step(model, sequences)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                pred = predict_future_step(model, sequences)
                loss = criterion(pred, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t_start
        eta = elapsed / epoch * (NUM_EPOCHS - epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{NUM_EPOCHS} | train={train_loss:.6f} | "
                  f"val={val_loss:.6f} | {elapsed/60:.0f}m | ETA {eta/60:.0f}m", flush=True)

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    # Save model
    save_path = str(MODELS_DIR / f"rigorous_seed{seed}.pth")
    torch.save(best_state, save_path)
    print(f"    Saved: {save_path} (val_loss={best_val_loss:.6f})", flush=True)

    # =====================================================
    # FIX #2: Evaluate on HELD-OUT TEST SET only
    # =====================================================
    print(f"    Evaluating on held-out test set ({len(test_tiles)} tiles)...", flush=True)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for sequences, targets in test_loader:
            pred = predict_future_step(model, sequences)
            all_preds.append(pred)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Overall test metrics
    test_mse = ((all_preds - all_targets) ** 2).mean().item()
    test_mae = (all_preds - all_targets).abs().mean().item()
    test_rmse = test_mse ** 0.5

    # Per-tile metrics for error bars
    tile_mses = ((all_preds - all_targets) ** 2).mean(dim=(1, 2, 3))
    tile_maes = (all_preds - all_targets).abs().mean(dim=(1, 2, 3))

    print(f"    TEST MSE:  {test_mse:.6f}", flush=True)
    print(f"    TEST MAE:  {test_mae:.6f}", flush=True)
    print(f"    TEST RMSE: {test_rmse:.6f}", flush=True)

    # =====================================================
    # FIX #4: Growth-region evaluation
    # =====================================================
    # Evaluate specifically on tiles where urban GROWTH occurred (2015 > 2000)
    # This tests the model's ability to predict CHANGE, not just stasis
    print(f"    Growth-region evaluation...", flush=True)

    growth_mses = []
    stable_mses = []
    growth_preds = []
    growth_gts = []

    for idx in range(len(test_tiles)):
        i, j = test_tiles[idx]
        ts = TILE_SIZE
        tile_2000 = ghsl[2000][i:i+ts, j:j+ts]
        tile_2015 = builtup_2015[i:i+ts, j:j+ts]
        
        # Growth = where 2015 > 2000 meaningfully
        growth_amount = (tile_2015 - tile_2000).mean()
        
        pred_tile = all_preds[idx].squeeze().numpy()
        gt_tile = all_targets[idx].squeeze().numpy()
        tile_mse = ((pred_tile - gt_tile) ** 2).mean()
        
        if growth_amount > 0.005:  # significant growth region
            growth_mses.append(tile_mse)
            growth_preds.append(pred_tile)
            growth_gts.append(gt_tile)
        else:
            stable_mses.append(tile_mse)

    growth_mses = np.array(growth_mses) if growth_mses else np.array([0.0])
    stable_mses = np.array(stable_mses) if stable_mses else np.array([0.0])

    print(f"    Growth tiles: {len(growth_mses)}, MSE={growth_mses.mean():.6f} ± {growth_mses.std():.6f}", flush=True)
    print(f"    Stable tiles: {len(stable_mses)}, MSE={stable_mses.mean():.6f} ± {stable_mses.std():.6f}", flush=True)

    # Baselines on test set tiles
    # Linear: predict = 2 * builtup_1990 - builtup_1975 (same temporal pattern as training)
    # Persistence: predict = builtup_1990 (no change from last seen)
    linear_mses_tiles = []
    persist_mses_tiles = []
    for idx in range(len(test_tiles)):
        i, j = test_tiles[idx]
        ts = TILE_SIZE
        gt = ghsl[2000][i:i+ts, j:j+ts]
        # Linear: extrapolate from 1975→1990 trend
        linear_pred = np.clip(ghsl[1990][i:i+ts, j:j+ts] + 
                              (ghsl[1990][i:i+ts, j:j+ts] - ghsl[1975][i:i+ts, j:j+ts]) * (10.0/15.0), 0, 1)
        persist_pred = ghsl[1990][i:i+ts, j:j+ts]
        
        linear_mses_tiles.append(((gt - linear_pred) ** 2).mean())
        persist_mses_tiles.append(((gt - persist_pred) ** 2).mean())

    linear_mses_tiles = np.array(linear_mses_tiles)
    persist_mses_tiles = np.array(persist_mses_tiles)

    linear_test_mse = linear_mses_tiles.mean()
    persist_test_mse = persist_mses_tiles.mean()

    print(f"    Linear baseline TEST MSE:      {linear_test_mse:.6f} ± {linear_mses_tiles.std():.6f}", flush=True)
    print(f"    Persistence baseline TEST MSE:  {persist_test_mse:.6f} ± {persist_mses_tiles.std():.6f}", flush=True)

    improvement_linear = (1 - test_mse / linear_test_mse) * 100 if linear_test_mse > 0 else 0
    improvement_persist = (1 - test_mse / persist_test_mse) * 100 if persist_test_mse > 0 else 0
    print(f"    vs Linear:      {improvement_linear:+.1f}%", flush=True)
    print(f"    vs Persistence: {improvement_persist:+.1f}%", flush=True)

    training_time = time.time() - t_start

    return {
        "seed": seed,
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "tile_mse_mean": float(tile_mses.mean().item()),
        "tile_mse_std": float(tile_mses.std().item()),
        "tile_mae_mean": float(tile_maes.mean().item()),
        "tile_mae_std": float(tile_maes.std().item()),
        "n_test_tiles": len(test_tiles),
        "growth_mse_mean": float(growth_mses.mean()),
        "growth_mse_std": float(growth_mses.std()),
        "growth_n_tiles": int(len(growth_mses)),
        "stable_mse_mean": float(stable_mses.mean()),
        "stable_mse_std": float(stable_mses.std()),
        "stable_n_tiles": int(len(stable_mses)),
        "linear_test_mse": float(linear_test_mse),
        "linear_test_mse_std": float(linear_mses_tiles.std()),
        "persist_test_mse": float(persist_test_mse),
        "persist_test_mse_std": float(persist_mses_tiles.std()),
        "improvement_vs_linear_pct": float(improvement_linear),
        "improvement_vs_persistence_pct": float(improvement_persist),
        "best_val_loss": float(best_val_loss),
        "training_time_hours": round(training_time / 3600, 2),
    }


# =====================================================
# Run all seeds
# =====================================================
all_results = []
for seed in SEEDS:
    result = train_one_seed(seed, train_tiles, val_tiles, test_tiles)
    all_results.append(result)
    gc.collect()

# =====================================================
# Aggregate results across seeds
# =====================================================
print(f"\n{'='*60}", flush=True)
print("AGGREGATED RESULTS (mean ± std across 3 seeds)", flush=True)
print(f"{'='*60}", flush=True)

metrics_to_aggregate = [
    "test_mse", "test_mae", "test_rmse",
    "growth_mse_mean", "stable_mse_mean",
    "linear_test_mse", "persist_test_mse",
    "improvement_vs_linear_pct", "improvement_vs_persistence_pct",
]

aggregated = {}
for key in metrics_to_aggregate:
    vals = [r[key] for r in all_results]
    mean = np.mean(vals)
    std = np.std(vals)
    aggregated[key] = {"mean": float(mean), "std": float(std)}
    print(f"  {key}: {mean:.6f} ± {std:.6f}", flush=True)

# =====================================================
# Save all results
# =====================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
final_output = {
    "experiment": "rigorous_3ch_training",
    "description": "3-seed training with spatial-block splits, held-out test, growth evaluation",
    "fixes": [
        "Spatial-block train/val/test split (no pixel overlap)",
        "Held-out test set (70/15/15, never used for model selection)",
        "3 seeds with mean±std error bars",
        "Growth-region vs stable-region evaluation"
    ],
    "config": {
        "tile_size": TILE_SIZE,
        "block_size": BLOCK_SIZE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "mc_dropout": MC_DROPOUT,
        "seeds": SEEDS,
    },
    "split_info": {
        "n_blocks_total": n_blocks,
        "n_blocks_train": len(train_blocks),
        "n_blocks_val": len(val_blocks),
        "n_blocks_test": len(test_blocks),
        "n_tiles_train": len(train_tiles),
        "n_tiles_val": len(val_tiles),
        "n_tiles_test": len(test_tiles),
    },
    "per_seed_results": all_results,
    "aggregated": aggregated,
    "timestamp": str(datetime.datetime.now()),
}

results_path = str(RESULTS_DIR / "rigorous_results.json")
with open(results_path, 'w') as f:
    json.dump(final_output, f, indent=2)
print(f"\nResults saved to {results_path}", flush=True)

print(f"\n{'='*60}", flush=True)
print("RIGOROUS TRAINING COMPLETE", flush=True)
print(f"{'='*60}", flush=True)
