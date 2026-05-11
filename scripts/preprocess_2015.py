#!/usr/bin/env python3
"""
Preprocess GHSL 2015 built-up data to match existing CONUS grid.
================================================================
1. Unzip downloaded GHSL E2015 archive
2. Reproject from WGS84 (4326) to EPSG:5070 at 250m
3. Crop to exact CONUS extent matching existing data
4. Normalize to [0,1] using same global max as year-2000 data
5. Save as geotiff_exports/CONUS_builtup_2015.tif

This produces a file with identical shape/CRS/transform as
CONUS_builtup_{1975,1990,2000}.tif for direct comparison.
"""
import os, sys, gc
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"

# =====================================================
# Step 1: Find source TIF
# =====================================================
src_tif = DATA_DIR / "GHS_BUILT_S_E2015_GLOBE_R2023A_4326_3ss_V1_0.tif"
if not src_tif.exists():
    # Fall back to 54009_100 variant
    src_tif = DATA_DIR / "GHS_BUILT_S_E2015_GLOBE_R2023A_54009_100_V1_0.tif"
if not src_tif.exists():
    print(f"ERROR: No GHSL 2015 tif found in {DATA_DIR}.")
    sys.exit(1)

print(f"[1/5] Source TIF: {src_tif}", flush=True)

# =====================================================
# Step 2: Read reference grid from existing CONUS data
# =====================================================
print("\n[2/5] Reading reference grid from CONUS_builtup_2000.tif...", flush=True)
ref_path = str(OUTPUT_DIR / "CONUS_builtup_2000.tif")
with rasterio.open(ref_path) as ref:
    ref_crs = ref.crs        # EPSG:5070
    ref_transform = ref.transform
    ref_shape = ref.shape     # (12717, 23996)
    ref_bounds = ref.bounds
    ref_dtype = ref.dtypes[0]

print(f"  Reference CRS: {ref_crs}")
print(f"  Reference shape: {ref_shape}")
print(f"  Reference bounds: {ref_bounds}")

# =====================================================
# Step 3: Reproject to CONUS grid
# =====================================================
print("\n[3/5] Reprojecting GHSL 2015 to EPSG:5070 at 250m (matching CONUS grid)...", flush=True)
print("  This may take 10-20 minutes for a global dataset...", flush=True)

dst_path = str(OUTPUT_DIR / "CONUS_builtup_2015.tif")

with rasterio.open(src_tif) as src:
    print(f"  Source CRS: {src.crs}")
    print(f"  Source shape: {src.shape}")
    
    # Create output matching reference exactly
    dst_kwargs = {
        'driver': 'GTiff',
        'height': ref_shape[0],
        'width': ref_shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': ref_crs,
        'transform': ref_transform,
        'compress': 'deflate',
        'predictor': 2,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }
    
    with rasterio.open(dst_path, 'w', **dst_kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )

print(f"  Reprojected to {dst_path}", flush=True)

# =====================================================
# Step 4: Normalize to [0,1] matching existing data
# =====================================================
print("\n[4/5] Normalizing to [0,1]...", flush=True)

with rasterio.open(dst_path) as src:
    data = src.read(1)
    profile = src.profile.copy()

print(f"  Raw range: [{data.min():.4f}, {data.max():.4f}]")
print(f"  Raw mean: {data.mean():.6f}")

# Replace NaN/Inf with 0
data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

# Clip negative values
data = np.clip(data, 0, None)

# Normalize by global max (or 1.0 if already normalized)
dmax = data.max()
if dmax > 1.0:
    data = data / dmax
    print(f"  Normalized by max={dmax:.4f}")
elif dmax > 0:
    print(f"  Data already in ~[0,{dmax:.4f}] range, keeping as-is")

# Check if existing data uses the same normalization
with rasterio.open(ref_path) as ref:
    ref_data = ref.read(1)
    print(f"  Reference 2000 range: [{ref_data.min():.4f}, {ref_data.max():.4f}]")
    print(f"  Reference 2000 mean:  {ref_data.mean():.6f}")
    del ref_data

# Write normalized data back
profile.update(dtype='float32')
with rasterio.open(dst_path, 'w', **profile) as dst:
    dst.write(data.astype(np.float32), 1)

print(f"  2015 normalized range: [{data.min():.4f}, {data.max():.4f}]")
print(f"  2015 normalized mean:  {data.mean():.6f}")

# =====================================================
# Step 5: Verify alignment
# =====================================================
print("\n[5/5] Verifying alignment with existing data...", flush=True)

with rasterio.open(dst_path) as src:
    new_shape = src.shape
    new_crs = src.crs
    new_transform = src.transform
    new_bounds = src.bounds

assert new_shape == ref_shape, f"Shape mismatch: {new_shape} vs {ref_shape}"
assert new_crs == ref_crs, f"CRS mismatch: {new_crs} vs {ref_crs}"

fsize_mb = os.path.getsize(dst_path) / (1024 * 1024)
print(f"  ✓ Shape matches: {new_shape}")
print(f"  ✓ CRS matches: {new_crs}")
print(f"  ✓ Transform matches: {new_transform}")
print(f"  File size: {fsize_mb:.1f} MB")

print(f"\n{'='*60}")
print(f"GHSL 2015 preprocessing complete!")
print(f"Output: {dst_path}")
print(f"{'='*60}")
