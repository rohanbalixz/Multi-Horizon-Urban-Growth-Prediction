#!/usr/bin/env python3
"""
Preprocess All Data: GHSL Built-Up + GHS-POP → CONUS Grid
==========================================================
Unzips, reprojects, and normalizes all GHSL data to a common grid:
  - CRS: EPSG:5070 (Albers Equal Area)
  - Resolution: 250m
  - Extent: CONUS bounds
  - Output: float32 GeoTIFFs in geotiff_exports/

Channels:
  CH0: GHSL built-up surface density [0, 1]
  CH1: GHS-BUILT-V built-up volume (log-normalized [0, 1])
  CH2: GHS-POP population density (log-normalized [0, 1])

Memory-efficient: uses rasterio.band() for disk-based reprojection
instead of loading entire global rasters into RAM.

Resumable: if the script crashes mid-run, re-running will skip already-
completed temp files and pick up from where it left off.

Usage:
    python scripts/preprocess_all_data.py              # process all channels
    python scripts/preprocess_all_data.py --bu-only     # process only built-up surface
    python scripts/preprocess_all_data.py --vol-only    # process only built-up volume
    python scripts/preprocess_all_data.py --pop-only    # process only population
"""
import argparse, json, os, subprocess, sys, shutil, zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "geotiff_exports"
OUTPUT_DIR.mkdir(exist_ok=True)

EPSG_TARGET = 5070
RESOLUTION = 250
CONUS_BOUNDS_4326 = {'west': -125.0, 'east': -66.0, 'south': 24.0, 'north': 49.5}

# Epochs to process (all available except 2025/2030)
BUILTUP_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
VOLUME_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]
POP_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015]


def get_reference_grid():
    """Compute the target CONUS grid in EPSG:5070 at 250m.

    Runs in a subprocess to avoid GDAL state leaks in the main process.
    """
    script = f"""
import json, rasterio
from rasterio.warp import calculate_default_transform
from pathlib import Path

OUTPUT_DIR = Path("{OUTPUT_DIR}")
ref_file = OUTPUT_DIR / "CONUS_builtup_2000.tif"
if ref_file.exists():
    with rasterio.open(ref_file) as src:
        t = src.transform
        result = {{"a": t.a, "b": t.b, "c": t.c, "d": t.d, "e": t.e, "f": t.f,
                   "width": src.width, "height": src.height, "crs": str(src.crs)}}
else:
    dst_crs = 'EPSG:{EPSG_TARGET}'
    src_width = int(({CONUS_BOUNDS_4326['east']} - {CONUS_BOUNDS_4326['west']}) / (3.0 / 3600))
    src_height = int(({CONUS_BOUNDS_4326['north']} - {CONUS_BOUNDS_4326['south']}) / (3.0 / 3600))
    dst_transform, dst_width, dst_height = calculate_default_transform(
        'EPSG:4326', dst_crs, src_width, src_height,
        left={CONUS_BOUNDS_4326['west']}, bottom={CONUS_BOUNDS_4326['south']},
        right={CONUS_BOUNDS_4326['east']}, top={CONUS_BOUNDS_4326['north']},
        resolution={RESOLUTION}
    )
    t = dst_transform
    result = {{"a": t.a, "b": t.b, "c": t.c, "d": t.d, "e": t.e, "f": t.f,
               "width": dst_width, "height": dst_height, "crs": dst_crs}}

print(json.dumps(result))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  get_reference_grid subprocess failed:\n{r.stderr}", flush=True)
        sys.exit(1)
    d = json.loads(r.stdout.strip())
    # Return transform as tuple (a, b, c, d, e, f) — workers reconstruct Affine internally
    transform = (d["a"], d["b"], d["c"], d["d"], d["e"], d["f"])
    return transform, d["width"], d["height"], d["crs"]




def find_builtup_source(year):
    """Find the GHSL built-up source for a given year.

    Looks for tif files directly in DATA_DIR (flat structure).
    Checks both 4326_3ss and 54009_100 variants.
    """
    # Check for tifs directly in data/ (prefer 4326_3ss, fall back to 54009_100)
    for variant in [f"GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_3ss_V1_0.tif",
                    f"GHS_BUILT_S_E{year}_GLOBE_R2023A_54009_100_V1_0.tif"]:
        tif_path = DATA_DIR / variant
        if tif_path.exists():
            return str(tif_path), None

    return None, None


def find_pop_source(year):
    """Find the GHS-POP source for a given year.

    Looks for tif files directly in DATA_DIR (flat structure).
    """
    tif_path = DATA_DIR / f"GHS_POP_E{year}_GLOBE_R2023A_54009_100_V1_0.tif"
    if tif_path.exists():
        return str(tif_path), None

    return None, None


def find_volume_source(year):
    """Find the GHS-BUILT-V source for a given year.

    Looks for tif files directly in DATA_DIR (flat structure).
    """
    tif_path = DATA_DIR / f"GHS_BUILT_V_E{year}_GLOBE_R2023A_54009_100_V1_0.tif"
    if tif_path.exists():
        return str(tif_path), None

    return None, None


def _run_worker(worker_name, args_str, timeout=1800, max_retries=3):
    """Run a worker function in a fresh subprocess and return parsed stats.

    Retries on transient GDAL/rasterio failures (e.g., 'not recognized as
    supported file format') which occur after many consecutive subprocess
    reprojections due to OS-level resource pressure.
    """
    import time
    cmd = [
        sys.executable, '-c',
        'import sys; sys.path.insert(0, ""); '
        f'from scripts.preprocess_all_data import {worker_name}; '
        f'{worker_name}({args_str})'
    ]
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(PROJECT_ROOT), timeout=timeout)
        for line in result.stdout.splitlines():
            if not line.startswith("STATS:"):
                print(line, flush=True)

        if result.returncode == 0:
            for line in reversed(result.stdout.splitlines()):
                if line.startswith("STATS:"):
                    return json.loads(line[6:])
            raise RuntimeError(f"No stats returned from {worker_name}")

        if attempt < max_retries and "not recognized as" in result.stderr:
            print(f"    Transient GDAL error (attempt {attempt}/{max_retries}), "
                  f"retrying in 5s...", flush=True)
            time.sleep(5)
            continue

        print(f"    SUBPROCESS ERROR:\n{result.stderr}", flush=True)
        raise RuntimeError(f"Subprocess {worker_name} failed")


def reproject_single_worker(src_path, tmp_path, ref_file, convert_to_density, log_transform):
    """Worker: reproject one file, save temp, print stats as JSON."""
    import json
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(ref_file) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = str(ref.crs)

    dst_data = np.zeros((ref_height, ref_width), dtype=np.float32)

    with rasterio.open(src_path) as src:
        res_x, res_y = src.res
        if src.crs.is_geographic:
            mid_lat = 37.0
            cell_area = (res_x * 111320 * np.cos(np.radians(mid_lat))) * (res_y * 110540)
        else:
            cell_area = abs(res_x * res_y)
        src_nodata = src.nodata
        print(f"    Source: {src.height}x{src.width}, CRS={src.crs}, "
              f"cell_area={cell_area:.0f} m², nodata={src_nodata}", flush=True)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=0.0,
        )

    dst_data = np.nan_to_num(dst_data, nan=0.0, posinf=0.0, neginf=0.0)

    if convert_to_density:
        dst_data[dst_data > cell_area] = 0.0
        dst_data /= cell_area
        dst_data = np.clip(dst_data, 0.0, 1.0)
    else:
        # Clamp leaked nodata values from bilinear interpolation.
        # For uint32 rasters (volume), nodata=4294967295 leaks through GDAL's
        # bilinear resampling as enormous float values. Any value > 0.1% of
        # the nodata sentinel is physically impossible and must be zeroed.
        if src_nodata is not None and float(src_nodata) > 1e6:
            dst_data[dst_data > float(src_nodata) * 0.001] = 0.0
        dst_data = np.clip(dst_data, 0, None)

    if log_transform:
        dst_data = np.log1p(dst_data).astype(np.float32)

    stats = {'min': float(dst_data.min()), 'max': float(dst_data.max()),
             'mean': float(dst_data.mean())}

    with rasterio.open(
        tmp_path, 'w', driver='GTiff',
        height=ref_height, width=ref_width,
        count=1, dtype='float32',
        crs=ref_crs, transform=ref_transform,
        compress='lzw'
    ) as dst:
        dst.write(dst_data.astype(np.float32), 1)

    print(f"STATS:{json.dumps(stats)}", flush=True)


def read_stats_worker(tif_path):
    """Worker: read a GeoTIFF and print min/max/mean stats."""
    import json
    import numpy as np
    import rasterio

    with rasterio.open(tif_path) as src:
        data = src.read(1)
    stats = {'min': float(data.min()), 'max': float(data.max()),
             'mean': float(data.mean())}
    print(f"STATS:{json.dumps(stats)}", flush=True)


def normalize_worker(tmp_path, output_path, global_max, ref_file):
    """Worker: read temp file, normalize by global_max, save final output."""
    import json
    import numpy as np
    import rasterio

    with rasterio.open(ref_file) as ref:
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_crs = str(ref.crs)

    with rasterio.open(tmp_path) as src:
        raw_data = src.read(1)

    if global_max > 0:
        normalized = raw_data / global_max
    else:
        normalized = raw_data
    del raw_data

    stats = {'min': float(normalized.min()), 'max': float(normalized.max()),
             'mean': float(normalized.mean())}

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=ref_height, width=ref_width,
        count=1, dtype='float32',
        crs=ref_crs, transform=ref_transform,
        compress='lzw'
    ) as dst:
        dst.write(normalized.astype(np.float32), 1)

    print(f"STATS:{json.dumps(stats)}", flush=True)


def reproject_in_subprocess(src_path, tmp_path, ref_file, convert_to_density=False,
                            log_transform=False):
    """Run reprojection in a fresh subprocess to avoid GDAL state leaks."""
    args = (f'r"{src_path}", r"{tmp_path}", r"{ref_file}", '
            f'{convert_to_density}, {log_transform}')
    return _run_worker('reproject_single_worker', args)


def read_stats_in_subprocess(tif_path):
    """Read GeoTIFF stats in a fresh subprocess."""
    return _run_worker('read_stats_worker', f'r"{tif_path}"', timeout=300)


def normalize_in_subprocess(tmp_path, output_path, global_max, ref_file):
    """Normalize and save in a fresh subprocess."""
    args = f'r"{tmp_path}", r"{output_path}", {global_max}, r"{ref_file}"'
    return _run_worker('normalize_worker', args, timeout=300)


def process_builtup():
    """Process all GHSL built-up epochs with consistent normalization.

    All rasterio operations run in subprocesses to prevent GDAL state leaks.
    Pass 1: Reproject each year (subprocess), save raw to temp files, track global max.
    Pass 2: Normalize all years by global max (subprocess), save final.
    """
    print("\n" + "=" * 60)
    print("PROCESSING GHSL BUILT-UP SURFACE")
    print("=" * 60)

    ref_file = str(OUTPUT_DIR / "CONUS_builtup_2000.tif")

    # Pass 1: reproject and track global max
    print("\n  Pass 1: Reprojecting and finding global max...", flush=True)
    global_max = 0
    temp_files = {}

    for year in BUILTUP_EPOCHS:
        tmp_path = OUTPUT_DIR / f"_temp_builtup_{year}.tif"

        if tmp_path.exists():
            print(f"\n  [{year}] Temp file exists, reading max...", flush=True)
            stats = read_stats_in_subprocess(str(tmp_path))
            local_max = stats['max']
            print(f"    Cached density range: [{stats['min']:.4f}, {local_max:.4f}], "
                  f"mean={stats['mean']:.6f}", flush=True)
            if local_max > global_max:
                global_max = local_max
            temp_files[year] = tmp_path
            continue

        print(f"\n  [{year}] Processing GHSL built-up (subprocess)...", flush=True)
        src_path, tmpdir = find_builtup_source(year)
        if src_path is None:
            print(f"    ERROR: No source found for {year}! Skipping.", flush=True)
            continue

        try:
            stats = reproject_in_subprocess(
                src_path, str(tmp_path), ref_file,
                convert_to_density=True, log_transform=False)
            local_max = stats['max']
            if local_max > global_max:
                global_max = local_max
            print(f"    Density range: [{stats['min']:.4f}, {local_max:.4f}], "
                  f"mean={stats['mean']:.6f}", flush=True)
            temp_files[year] = tmp_path
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # Pass 2: normalize all years by global max (each in subprocess)
    print(f"\n  Global built-up max: {global_max:.4f}", flush=True)
    print("  Pass 2: Normalizing and saving final files...", flush=True)

    for year, tmp_path in temp_files.items():
        output_path = OUTPUT_DIR / f"CONUS_builtup_{year}.tif"
        stats = normalize_in_subprocess(
            str(tmp_path), str(output_path), global_max, ref_file)
        print(f"  [{year}] Saving: range=[{stats['min']:.4f}, {stats['max']:.4f}], "
              f"mean={stats['mean']:.6f}", flush=True)
        print(f"    Saved: {output_path}", flush=True)
        tmp_path.unlink(missing_ok=True)

    print("  Temp files cleaned up.", flush=True)


def process_population():
    """Process all GHS-POP epochs with consistent log-normalization.

    All rasterio operations run in subprocesses to prevent GDAL state leaks.
    Pass 1: Reproject each year (subprocess), save raw to temp files, track global log-max.
    Pass 2: Normalize all years by global max (subprocess), save final.
    """
    print("\n" + "=" * 60)
    print("PROCESSING GHS-POP POPULATION DENSITY")
    print("=" * 60)

    ref_file = str(OUTPUT_DIR / "CONUS_builtup_2000.tif")

    print("\n  Pass 1: Reprojecting and finding global log-max...", flush=True)
    global_log_max = 0
    temp_files = {}

    for year in POP_EPOCHS:
        tmp_path = OUTPUT_DIR / f"_temp_pop_{year}.tif"

        if tmp_path.exists():
            print(f"\n  [{year}] Temp file exists, reading max...", flush=True)
            stats = read_stats_in_subprocess(str(tmp_path))
            local_max = stats['max']
            print(f"    Cached log-transformed: max={local_max:.4f}, "
                  f"mean={stats['mean']:.6f}", flush=True)
            if local_max > global_log_max:
                global_log_max = local_max
            temp_files[year] = tmp_path
            continue

        print(f"\n  [{year}] Loading GHS-POP (subprocess)...", flush=True)
        src_path, tmpdir = find_pop_source(year)
        if src_path is None:
            print(f"    ERROR: No source found for {year}! Skipping.", flush=True)
            continue

        try:
            stats = reproject_in_subprocess(
                src_path, str(tmp_path), ref_file,
                convert_to_density=False, log_transform=True)
            local_max = stats['max']
            if local_max > global_log_max:
                global_log_max = local_max
            print(f"    Log-transformed: max={local_max:.4f}, mean={stats['mean']:.6f}",
                  flush=True)
            temp_files[year] = tmp_path
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # Pass 2: normalize each by global max (each in subprocess)
    print(f"\n  Global log-population max: {global_log_max:.4f}", flush=True)
    print("  Pass 2: Normalizing and saving final files...", flush=True)

    for year, tmp_path in temp_files.items():
        output_path = OUTPUT_DIR / f"CONUS_population_{year}.tif"
        stats = normalize_in_subprocess(
            str(tmp_path), str(output_path), global_log_max, ref_file)
        print(f"  [{year}] Saving: range=[{stats['min']:.4f}, {stats['max']:.4f}], "
              f"mean={stats['mean']:.6f}", flush=True)
        print(f"    Saved: {output_path}", flush=True)
        tmp_path.unlink(missing_ok=True)

    print("  Temp files cleaned up.", flush=True)


def process_volume():
    """Process all GHS-BUILT-V epochs with consistent normalization.

    Volume data (m³/cell) uses log-normalization (same as population)
    because volume values are heavily skewed. NOT convert_to_density
    since the >cell_area filter would destroy data (m³ >> m²).
    All rasterio operations run in subprocesses.
    """
    print("\n" + "=" * 60)
    print("PROCESSING GHS-BUILT-V VOLUME")
    print("=" * 60)

    ref_file = str(OUTPUT_DIR / "CONUS_builtup_2000.tif")

    print("\n  Pass 1: Reprojecting and finding global max...", flush=True)
    global_max = 0
    temp_files = {}

    for year in VOLUME_EPOCHS:
        tmp_path = OUTPUT_DIR / f"_temp_volume_{year}.tif"

        if tmp_path.exists():
            print(f"\n  [{year}] Temp file exists, reading max...", flush=True)
            stats = read_stats_in_subprocess(str(tmp_path))
            local_max = stats['max']
            print(f"    Cached log-volume: [{stats['min']:.4f}, {local_max:.4f}], "
                  f"mean={stats['mean']:.6f}", flush=True)
            if local_max > global_max:
                global_max = local_max
            temp_files[year] = tmp_path
            continue

        print(f"\n  [{year}] Processing GHS-BUILT-V (subprocess)...", flush=True)
        src_path, tmpdir = find_volume_source(year)
        if src_path is None:
            print(f"    ERROR: No source found for {year}! Skipping.", flush=True)
            continue

        try:
            stats = reproject_in_subprocess(
                src_path, str(tmp_path), ref_file,
                convert_to_density=False, log_transform=True)
            local_max = stats['max']
            if local_max > global_max:
                global_max = local_max
            print(f"    Log-volume: [{stats['min']:.4f}, {local_max:.4f}], "
                  f"mean={stats['mean']:.6f}", flush=True)
            temp_files[year] = tmp_path
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # Pass 2: normalize all years by global max
    print(f"\n  Global volume max: {global_max:.4f}", flush=True)
    print("  Pass 2: Normalizing and saving final files...", flush=True)

    for year, tmp_path in temp_files.items():
        output_path = OUTPUT_DIR / f"CONUS_volume_{year}.tif"
        stats = normalize_in_subprocess(
            str(tmp_path), str(output_path), global_max, ref_file)
        print(f"  [{year}] Saving: range=[{stats['min']:.4f}, {stats['max']:.4f}], "
              f"mean={stats['mean']:.6f}", flush=True)
        print(f"    Saved: {output_path}", flush=True)
        tmp_path.unlink(missing_ok=True)

    print("  Temp files cleaned up.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess GHSL data to CONUS grid")
    parser.add_argument('--bu-only', action='store_true', help='Process only built-up surface')
    parser.add_argument('--vol-only', action='store_true', help='Process only built-up volume')
    parser.add_argument('--pop-only', action='store_true', help='Process only population')
    args = parser.parse_args()

    only_flags = [args.bu_only, args.vol_only, args.pop_only]
    any_only = any(only_flags)
    do_bu = args.bu_only or not any_only
    do_vol = args.vol_only or not any_only
    do_pop = args.pop_only or not any_only

    print("=" * 60)
    print("FULL DATA PREPROCESSING PIPELINE")
    print(f"Target: EPSG:{EPSG_TARGET}, {RESOLUTION}m, CONUS")
    parts = []
    if do_bu: parts.append("BU-surface")
    if do_vol: parts.append("BU-volume")
    if do_pop: parts.append("POP")
    print(f"Mode: {' + '.join(parts)}")
    print("=" * 60)

    _, ref_width, ref_height, _ = get_reference_grid()
    print(f"Reference grid: {ref_height} x {ref_width} pixels")

    if do_bu:
        process_builtup()
    if do_vol:
        process_volume()
    if do_pop:
        process_population()

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("\nOutput files:")
    for prefix in ["CONUS_builtup", "CONUS_volume", "CONUS_population"]:
        for f in sorted(OUTPUT_DIR.glob(f"{prefix}_*.tif")):
            print(f"  {f} ({f.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
