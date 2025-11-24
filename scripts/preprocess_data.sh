#!/bin/bash
# Preprocessing script for GHSL and OSM data

set -e

echo "=== Neural Time Capsule Data Preprocessing ==="
echo ""

# Check if data directories exist
if [ ! -d "data/ghsl" ]; then
    echo "Error: data/ghsl/ directory not found"
    echo "Please download GHSL R2023A data first"
    exit 1
fi

if [ ! -f "data/us-251031.osm.pbf" ]; then
    echo "Error: OSM data not found"
    echo "Please download OpenStreetMap data first"
    exit 1
fi

echo "✓ Data directories found"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Warning: .venv not found, using system Python"
fi

# Run preprocessing
echo ""
echo "Starting preprocessing pipeline..."
python -m src.data.preprocessing \
    --ghsl_dir data/ghsl \
    --osm_file data/us-251031.osm.pbf \
    --output_dir data/processed \
    --tile_size 128 \
    --overlap 0.5

echo ""
echo "✓ Preprocessing complete!"
echo "Processed tiles saved to data/processed/"
