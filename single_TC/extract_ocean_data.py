#!/usr/bin/env python3
"""
extract_ocean_data.py - Unified script to extract ocean data (temperature/salinity/chlorophyll) 
from HYCOM, OISST, or Copernicus datasets for hurricane track analysis.

Supports:
- HYCOM sea surface temperature (SST) and salinity (SSS)
- HYCOM temperature and salinity at specified depths
- NOAA OISST v2.1 sea surface temperature
- Copernicus Ocean Color chlorophyll-a concentration

Each fix gets var(D−15)…var(D)…var(D+15) => 31 new columns.

Scale and Offset:
• HYCOM water_temp: actual_temp = (raw_value * 0.001) + 20  [°C]
• HYCOM salinity: actual_salinity = (raw_value * 0.001) + 20  [PSU]
• OISST sst: actual_temp = (raw_value * 0.01) + 0  [°C]
• COPERNICUS chlor_a: actual_chlor = (raw_value * 1.0) + 0  [mg/m³]
  Note: Copernicus scale/offset should be verified with dataset documentation

Usage:
------
$ python extract_ocean_data.py <dataset> <variable> [depth] <trackfile.txt>

Where:
- dataset: HYCOM, OISST, or COPERNICUS
- variable: T (temperature), S (salinity), or C (chlorophyll-a)
  - T: available for HYCOM and OISST
  - S: available for HYCOM only
  - C: available for COPERNICUS only
- depth: depth level (optional, defaults to 0 for surface)
- trackfile.txt: HURDAT-style best-track file

Examples:
---------
# HYCOM surface temperature
$ python extract_ocean_data.py HYCOM T AL312020_IOTA_26.txt
→ ../t_data/AL312020_IOTA_26_HYCOM_T_0.txt

# HYCOM salinity at 10m depth
$ python extract_ocean_data.py HYCOM S 10 AL312020_IOTA_26.txt
→ ../zt_data/AL312020_IOTA_26_HYCOM_S_10.txt

# OISST surface temperature
$ python extract_ocean_data.py OISST T AL312020_IOTA_26.txt
→ ../t_data/AL312020_IOTA_26_OISST.txt

# Copernicus chlorophyll-a
$ python extract_ocean_data.py COPERNICUS C AL312020_IOTA_26.txt
→ ../t_data/AL312020_IOTA_26_COPERNICUS_CHLOR.txt
"""
import sys
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import ee


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    """Configuration for different datasets and variables."""
    
    HYCOM_SCALE = 0.001
    HYCOM_OFFSET = 20.0
    OISST_SCALE = 0.01
    OISST_OFFSET = 0.0
    # Note: Copernicus chlorophyll-a scale/offset should be verified with documentation
    # These values assume data is already in mg/m³ units
    COPERNICUS_SCALE = 1.0
    COPERNICUS_OFFSET = 0.0
    
    DATASET_INFO = {
        'HYCOM': {
            'collection': 'HYCOM/sea_temp_salinity',
            'variables': {
                'T': {'prefix': 'water_temp', 'units': '°C'},
                'S': {'prefix': 'salinity', 'units': 'PSU'}
            },
            'scale': HYCOM_SCALE,
            'offset': HYCOM_OFFSET
        },
        'OISST': {
            'collection': 'NOAA/CDR/OISST/V2_1',
            'variables': {
                'T': {'band': 'sst', 'units': '°C'}
            },
            'scale': OISST_SCALE,
            'offset': OISST_OFFSET
        },
        'COPERNICUS': {
            'collection': 'COPERNICUS/MARINE/SATELLITE_OCEAN_COLOR/V6',
            'variables': {
                'C': {'band': 'chlor_a', 'units': 'mg/m³'}
            },
            'scale': COPERNICUS_SCALE,
            'offset': COPERNICUS_OFFSET
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def parse_latlon(token: str) -> float:
    """Convert '13.4N' → +13.4 ; '82.7W' → -82.7."""
    v, hemi = float(token[:-1]), token[-1].upper()
    return v if hemi in ("N", "E") else -v


def get_band_name(dataset: str, variable: str, depth: int) -> str:
    """Get the appropriate band name for the dataset/variable/depth combination."""
    if dataset in ['OISST', 'COPERNICUS']:
        return Config.DATASET_INFO[dataset]['variables'][variable]['band']
    else:  # HYCOM
        prefix = Config.DATASET_INFO[dataset]['variables'][variable]['prefix']
        return f"{prefix}_{depth}"


def get_output_path(in_path: Path, dataset: str, variable: str, depth: int) -> Path:
    """Determine the output path based on dataset and depth."""
    if dataset == 'OISST':
        out_dir = in_path.parent.parent / "t_data"
        out_file = out_dir / f"{in_path.stem}_OISST.txt"
    elif dataset == 'COPERNICUS':
        out_dir = in_path.parent.parent / "t_data"
        out_file = out_dir / f"{in_path.stem}_COPERNICUS_CHLOR.txt"
    else:  # HYCOM
        if depth == 0:
            out_dir = in_path.parent.parent / "t_data"
            suffix = "T_0" if variable == 'T' else "S_0"
        else:
            out_dir = in_path.parent.parent / "zt_data"
            suffix = f"{variable}_{depth}"
        out_file = out_dir / f"{in_path.stem}_HYCOM_{suffix}.txt"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_file


def get_column_prefix(dataset: str, variable: str, depth: int) -> str:
    """Get the column prefix for output columns."""
    if dataset == 'OISST':
        return 'sst'
    elif dataset == 'COPERNICUS':
        return 'chlor_a'
    else:  # HYCOM
        prefix = Config.DATASET_INFO[dataset]['variables'][variable]['prefix']
        if depth == 0:
            return prefix
        else:
            return f"{prefix}_{depth}"


def get_daily_value(imgcol: ee.ImageCollection, ymd: str, lon: float, lat: float,
                    band_name: str, scale: float, offset: float) -> float:
    """Extract daily value for given date and location, or NaN if masked."""
    d0 = ee.Date.fromYMD(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
    img = imgcol.filterDate(d0, d0.advance(1, "day")).first()
    if img is None:
        return float("nan")
    
    pt = ee.Geometry.Point(lon, lat)
    try:
        val = (img.select(band_name)
                  .reduceRegion(ee.Reducer.first(), pt, scale=1_000)
                  .get(band_name))
        # Apply scale and offset
        return ee.Number(val).multiply(scale).add(offset).getInfo()
    except Exception:
        return float("nan")


def read_track_file(in_path: Path) -> tuple:
    """Read HURDAT-style track file and return header lines and data rows."""
    date_line = re.compile(r"^\d{8},")
    header, rows = [], []
    
    with in_path.open() as f:
        for line in f:
            if date_line.match(line):
                parts = [p.strip() for p in line.split(",")]
                rows.append(dict(
                    raw=line.rstrip("\n"),
                    ymd=parts[0],
                    lat=parse_latlon(parts[4]),
                    lon=parse_latlon(parts[5]),
                ))
            else:
                header.append(line.rstrip("\n"))
    
    return header, pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Main processing function
# ──────────────────────────────────────────────────────────────────────────────
def process_ocean_data(dataset: str, variable: str, depth: int, trackfile: str) -> None:
    """Main processing function to extract ocean data for hurricane tracks."""
    
    # Validate inputs
    dataset = dataset.upper()
    variable = variable.upper()
    
    if dataset not in Config.DATASET_INFO:
        sys.exit(f"✗ Dataset must be one of {list(Config.DATASET_INFO.keys())}, not '{dataset}'")
    
    if variable not in Config.DATASET_INFO[dataset]['variables']:
        valid_vars = list(Config.DATASET_INFO[dataset]['variables'].keys())
        sys.exit(f"✗ Variable must be one of {valid_vars} for {dataset}")
    
    # Check dataset-specific constraints
    if dataset == 'OISST' and variable == 'S':
        sys.exit("✗ OISST only supports temperature (T), not salinity (S)")
    
    if dataset == 'COPERNICUS' and depth != 0:
        sys.exit("✗ COPERNICUS chlorophyll-a is only available at surface (depth 0)")
    
    if dataset in ['OISST', 'COPERNICUS'] and depth != 0:
        print(f"Note: {dataset} only provides surface data. Using depth 0.")
        depth = 0
    
    # Setup paths
    in_path = Path(trackfile).expanduser().resolve()
    if not in_path.is_file():
        sys.exit(f"✗ '{trackfile}' not found")
    
    out_file = get_output_path(in_path, dataset, variable, depth)
    
    # Initialize Earth Engine
    ee.Initialize(project='ee-cnggao')
    collection_name = Config.DATASET_INFO[dataset]['collection']
    imgcol = ee.ImageCollection(collection_name)
    
    # Read track file
    header, df = read_track_file(in_path)
    
    # Setup for data extraction
    band_name = get_band_name(dataset, variable, depth)
    scale = Config.DATASET_INFO[dataset]['scale']
    offset = Config.DATASET_INFO[dataset]['offset']
    
    # Sample 31-day window
    window = range(-15, 16)  # -15...+15
    col_prefix = get_column_prefix(dataset, variable, depth)
    val_cols = [f"{col_prefix}{d:+d}" for d in window]
    
    def value_window(row):
        base = datetime.strptime(row.ymd, "%Y%m%d")
        return [get_daily_value(imgcol,
                               (base + timedelta(days=off)).strftime("%Y%m%d"),
                               row.lon, row.lat, band_name, scale, offset)
                for off in window]
    
    # Process data
    units = Config.DATASET_INFO[dataset]['variables'][variable].get('units', '')
    
    # Note about Copernicus scaling and data availability
    if dataset == 'COPERNICUS':
        print("Note: Chlorophyll-a scale/offset values should be verified with dataset documentation.")
        print("      Chlorophyll-a data may be missing due to cloud cover or non-ocean areas.")
    
    print(f"Processing {dataset} {variable} data at depth {depth}m "
          f"(scale: {scale}, offset: {offset}) for {len(df)} records...")
    
    df[val_cols] = df.apply(value_window, axis=1, result_type="expand")
    
    # Write output
    with out_file.open("w") as f:
        # Write header
        for h in header:
            f.write(h + "\n")
        
        # Write data rows
        for _, r in df.iterrows():
            # Use appropriate formatting based on variable type
            if variable == 'C':  # Chlorophyll-a may have smaller values
                val_str = ", ".join(f"{v:8.4f}" for v in r[val_cols])
            else:  # Temperature and salinity
                val_str = ", ".join(f"{v:6.2f}" for v in r[val_cols])
            f.write(f"{r.raw}, {val_str}\n")
    
    # Print friendly path
    try:
        display_path = out_file.relative_to(Path.cwd())
    except ValueError:
        display_path = os.path.relpath(out_file, Path.cwd())
    print(f"✓ Wrote {display_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Parse command line arguments and run processing."""
    if len(sys.argv) < 4:
        print("Usage: python extract_ocean_data.py <dataset> <variable> [depth] <trackfile.txt>")
        print()
        print("Datasets:")
        print("  HYCOM      - HYCOM sea temperature and salinity")
        print("  OISST      - NOAA OISST v2.1 sea surface temperature")
        print("  COPERNICUS - Copernicus Ocean Color chlorophyll-a")
        print()
        print("Variables:")
        print("  T - Temperature (°C) - HYCOM and OISST")
        print("  S - Salinity (PSU) - HYCOM only")
        print("  C - Chlorophyll-a (mg/m³) - COPERNICUS only")
        print()
        print("Depth:")
        print("  Optional depth in meters (default: 0 for surface)")
        print("  Only used for HYCOM dataset")
        print()
        print("Examples:")
        print("  python extract_ocean_data.py HYCOM T track.txt          # HYCOM SST")
        print("  python extract_ocean_data.py HYCOM S 10 track.txt       # HYCOM salinity at 10m")
        print("  python extract_ocean_data.py OISST T track.txt          # OISST SST")
        print("  python extract_ocean_data.py COPERNICUS C track.txt     # Chlorophyll-a")
        sys.exit(1)
    
    # Parse arguments
    if len(sys.argv) == 4:
        # No depth specified, assume surface (0)
        dataset, variable, trackfile = sys.argv[1:4]
        depth = 0
    else:
        # Depth specified
        dataset, variable = sys.argv[1:3]
        try:
            depth = int(sys.argv[3])
        except ValueError:
            # If third argument isn't a number, assume it's the trackfile
            # and depth is 0
            trackfile = sys.argv[3]
            depth = 0
        else:
            trackfile = sys.argv[4]
    
    if not trackfile.lower().endswith(".txt"):
        sys.exit("✗ Track file must have .txt extension")
    
    # Process the data
    process_ocean_data(dataset, variable, depth, trackfile)


if __name__ == "__main__":
    main()
