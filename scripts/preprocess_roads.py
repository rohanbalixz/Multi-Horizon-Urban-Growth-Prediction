#!/usr/bin/env python3
"""
Preprocess TIGER/Line Primary Road Network for CONUS.

Downloads US Census Bureau TIGER/Line primary road shapefiles for all
48 contiguous US states + DC, merges them, reprojects to EPSG:5070,
rasterizes to the 250m CONUS grid, and computes a log-normalized
road proximity raster.

Road proximity = inverse log-distance to nearest primary road.
Higher values = closer to a primary road (Interstate or US highway).

This is used as the 'roads' input to the SLEUTH CA model, replacing
the built-up volume proxy previously used.

TIGER/Line primary roads include:
  - Interstates (MTFCC=S1100)
  - US highways (MTFCC=S1200)

These are stable across the full 1975-2020 study period — the
Interstate system was ~95% complete by 1975, making a static
2020 road network a valid proxy for the entire study period.

Usage:
    python scripts/preprocess_roads.py

Output:
    geotiff_exports/CONUS_road_proximity.tif
    geotiff_exports/CONUS_road_distance_m.tif  (raw distance in metres)

Runtime: ~2-3 hours (download ~800MB + rasterization of 12718x23997 grid)
"""
import sys, os, io, zipfile, time, requests, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
CACHE_DIR  = PROJECT_ROOT / "data" / "tiger_roads"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box

print("=" * 60)
print("PREPROCESSING TIGER/LINE PRIMARY ROADS")
print("=" * 60)

# ─── Reference grid from existing CONUS builtup file ──────────
ref_path = OUTPUT_DIR / "CONUS_builtup_2015.tif"
with rasterio.open(str(ref_path)) as ref:
    ref_transform = ref.transform
    ref_crs       = ref.crs
    ref_width     = ref.width
    ref_height    = ref.height
    ref_bounds    = ref.bounds

print(f"Reference grid: {ref_height}×{ref_width}, CRS={ref_crs}")
print(f"Bounds: {ref_bounds}")

# ─── FIPS codes for 48 contiguous states + DC ─────────────────
CONUS_FIPS = [
    "01","04","05","06","08","09","10","11","12","13",
    "16","17","18","19","20","21","22","23","24","25",
    "26","27","28","29","30","31","32","33","34","35",
    "36","37","38","39","40","41","42","44","45","46",
    "47","48","49","50","51","53","54","55","56",
]

# TIGER/Line 2022 primary roads URL template
# MTFCC S1100 = Interstate, S1200 = US highway
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2022/PRISECROADS/tl_2022_{fips}_prisecroads.zip"

# ─── Download + cache shapefiles ──────────────────────────────
print(f"\n[1] Downloading TIGER/Line primary road shapefiles ({len(CONUS_FIPS)} states)...")
t0 = time.time()

all_gdfs = []
failed   = []

session = requests.Session()
session.headers.update({"User-Agent": "NeuralTimeCapsule/1.0 research project"})

for i, fips in enumerate(CONUS_FIPS):
    cache_file = CACHE_DIR / f"tl_2022_{fips}_prisecroads.gpkg"

    if cache_file.exists():
        print(f"  [{i+1:02d}/{len(CONUS_FIPS)}] FIPS={fips} — cached", flush=True)
        gdf = gpd.read_file(str(cache_file))
        all_gdfs.append(gdf)
        continue

    url = TIGER_URL.format(fips=fips)
    try:
        print(f"  [{i+1:02d}/{len(CONUS_FIPS)}] FIPS={fips} downloading...", end=" ", flush=True)
        resp = session.get(url, timeout=120)
        resp.raise_for_status()

        # Extract shapefile from zip in memory
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find the .shp file
            shp_names = [n for n in zf.namelist() if n.endswith(".shp")]
            if not shp_names:
                print(f"WARNING: no .shp in zip for FIPS={fips}")
                failed.append(fips)
                continue

            # Extract all components to cache dir
            zf.extractall(str(CACHE_DIR / f"tmp_{fips}"))

        shp_path = CACHE_DIR / f"tmp_{fips}" / shp_names[0]
        gdf = gpd.read_file(str(shp_path))

        # Filter to primary roads only (S1100=Interstate, S1200=US highway)
        if "MTFCC" in gdf.columns:
            gdf = gdf[gdf["MTFCC"].isin(["S1100", "S1200"])].copy()

        if len(gdf) == 0:
            print(f"0 primary roads — skipping")
            continue

        # Cache as GeoPackage for fast re-read
        gdf.to_file(str(cache_file), driver="GPKG")
        all_gdfs.append(gdf)
        print(f"{len(gdf):,} primary road segments", flush=True)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        failed.append(fips)

print(f"\n  Downloaded {len(all_gdfs)} states in {(time.time()-t0)/60:.1f} min")
if failed:
    print(f"  WARNING: Failed states: {failed}")

if not all_gdfs:
    print("ERROR: No road data downloaded. Check network connection.")
    sys.exit(1)

# ─── Merge all states ─────────────────────────────────────────
print("\n[2] Merging all state road networks...", flush=True)
roads = gpd.pd.concat(all_gdfs, ignore_index=True)
print(f"  Total segments: {len(roads):,}")
print(f"  Original CRS: {roads.crs}")

# ─── Reproject to EPSG:5070 ───────────────────────────────────
print("\n[3] Reprojecting to EPSG:5070 (Albers Equal Area)...", flush=True)
roads = roads.to_crs(ref_crs)
print(f"  Reprojected CRS: {roads.crs}")

# Clip to CONUS bounds (with 50km buffer)
conus_box = box(
    ref_bounds.left   - 50000,
    ref_bounds.bottom - 50000,
    ref_bounds.right  + 50000,
    ref_bounds.top    + 50000,
)
roads = roads[roads.intersects(conus_box)].copy()
print(f"  After CONUS clip: {len(roads):,} segments")

# ─── Rasterize road network ───────────────────────────────────
print("\n[4] Rasterizing road network to 250m grid...", flush=True)
print(f"  Grid: {ref_height}×{ref_width}", flush=True)
t1 = time.time()

# Burn value=1 where roads exist
road_raster = np.zeros((ref_height, ref_width), dtype=np.uint8)

# Use rasterio.features.rasterize for efficiency
shapes = ((geom, 1) for geom in roads.geometry if geom is not None and not geom.is_empty)
road_raster = rasterize(
    shapes=shapes,
    out_shape=(ref_height, ref_width),
    transform=ref_transform,
    fill=0,
    dtype=np.uint8,
    all_touched=True,   # include all pixels touched by road line
)

n_road_pixels = int(road_raster.sum())
print(f"  Road pixels: {n_road_pixels:,} of {ref_height*ref_width:,} "
      f"({100*n_road_pixels/(ref_height*ref_width):.2f}%)")
print(f"  Rasterization: {time.time()-t1:.1f}s", flush=True)

# ─── Distance transform ───────────────────────────────────────
print("\n[5] Computing Euclidean distance transform...", flush=True)
t2 = time.time()

# distance_transform_edt returns distance in pixels; multiply by 250 for metres
non_road = (road_raster == 0).astype(np.uint8)
dist_pixels = distance_transform_edt(non_road)
dist_metres = (dist_pixels * 250.0).astype(np.float32)

print(f"  Distance transform: {time.time()-t2:.1f}s")
print(f"  Distance range: min={dist_metres.min():.0f}m, "
      f"max={dist_metres.max()/1000:.1f}km, "
      f"mean={dist_metres.mean()/1000:.2f}km")

# ─── Save raw distance raster ─────────────────────────────────
print("\n[6] Saving raw distance raster...", flush=True)
dist_path = str(OUTPUT_DIR / "CONUS_road_distance_m.tif")
with rasterio.open(
    dist_path, 'w', driver='GTiff',
    height=ref_height, width=ref_width,
    count=1, dtype='float32',
    crs=ref_crs, transform=ref_transform,
    compress='lzw'
) as dst:
    dst.write(dist_metres, 1)
print(f"  Saved: {dist_path} ({os.path.getsize(dist_path)/1024/1024:.1f} MB)")

# ─── Normalize to [0, 1] proximity raster ─────────────────────
# proximity = 1 / (1 + log1p(dist_metres / scale))
# scale=5000m: at 5km from highway → proximity≈0.5
# This gives high values near roads, low values far away.
# Matches SLEUTH road_gravity semantics: higher = more road influence.
print("\n[7] Computing log-normalized road proximity [0,1]...", flush=True)
SCALE = 5000.0   # 5km reference scale
proximity = (1.0 / (1.0 + np.log1p(dist_metres / SCALE))).astype(np.float32)

print(f"  Proximity range: min={proximity.min():.4f}, "
      f"max={proximity.max():.4f}, mean={proximity.mean():.4f}")
print(f"  Pixels within 1km of highway: "
      f"{int((dist_metres < 1000).sum()):,} "
      f"({100*(dist_metres < 1000).mean():.1f}%)")
print(f"  Pixels within 5km of highway: "
      f"{int((dist_metres < 5000).sum()):,} "
      f"({100*(dist_metres < 5000).mean():.1f}%)")

prox_path = str(OUTPUT_DIR / "CONUS_road_proximity.tif")
with rasterio.open(
    prox_path, 'w', driver='GTiff',
    height=ref_height, width=ref_width,
    count=1, dtype='float32',
    crs=ref_crs, transform=ref_transform,
    compress='lzw'
) as dst:
    dst.write(proximity, 1)
print(f"  Saved: {prox_path} ({os.path.getsize(prox_path)/1024/1024:.1f} MB)")

# ─── Cleanup temp directories ─────────────────────────────────
import shutil
for fips in CONUS_FIPS:
    tmp = CACHE_DIR / f"tmp_{fips}"
    if tmp.exists():
        shutil.rmtree(str(tmp))

print("\n" + "=" * 60)
print("ROAD PREPROCESSING COMPLETE")
print("=" * 60)
print(f"Total time: {(time.time()-t0)/60:.1f} min")
print(f"Output files:")
print(f"  {prox_path}")
print(f"  {dist_path}")
print(f"\nReady to use as SLEUTH road input.")
