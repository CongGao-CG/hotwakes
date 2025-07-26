#!/usr/bin/env python3
"""
extract_HYCOM_surface.py  ––  append a 31-day surface variable window (HYCOM sea_temp_salinity)
to every record of a HURDAT-style best-track text file.
• Each fix gets var(D−15)…var(D)…var(D+15)  => 31 new columns.
• Output is written one level up, in ../t_data/<base>_HYCOM_T_0.txt or _HYCOM_S_0.txt
• HYCOM data is on a 0.08° lat/long grid; water_temp_0 and salinity_0 are surface values

Scale and Offset:
• water_temp_0 (SST): actual_temp = (raw_value * 0.001) + 20  [°C]
• salinity_0 (SSS): actual_salinity = (raw_value * 0.001) + 20  [PSU]

Note: These scale/offset values are based on typical HYCOM encoding. Verify with
the specific dataset documentation if results seem incorrect.

Example
-------
$ python extract_HYCOM_surface.py SST AL312020_IOTA_26.txt
→  ../t_data/AL312020_IOTA_26_HYCOM_T_0.txt

$ python extract_HYCOM_surface.py SSS AL312020_IOTA_26.txt
→  ../t_data/AL312020_IOTA_26_HYCOM_S_0.txt
"""
import sys, os, re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import ee
# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_latlon(token: str) -> float:
    """'13.4N' → +13.4 ;  '82.7W' → –82.7."""
    v, hemi = float(token[:-1]), token[-1].upper()
    return v if hemi in ("N", "E") else -v

def get_daily_surface_value(imgcol: ee.ImageCollection, ymd: str,
                            lon: float, lat: float, var_type: str) -> float:
    """Daily HYCOM surface value for *ymd* at lon/lat, or NaN if masked.
    
    var_type: 'SST' for sea surface temperature or 'SSS' for sea surface salinity
    
    HYCOM stores values as scaled integers with different scale/offset for each variable:
    - water_temp_0 (SST): actual_temp = (value * 0.001) + 20  [°C]
    - salinity_0 (SSS): actual_salinity = (value * 0.001) + 20  [PSU]
    
    Data is on a 0.08° grid (~8.9 km).
    """
    d0 = ee.Date.fromYMD(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
    img = imgcol.filterDate(d0, d0.advance(1, "day")).first()
    if img is None:
        return float("nan")
    
    # Select appropriate band and scaling parameters
    if var_type == "SST":
        band_name = "water_temp_0"
        scale_factor = 0.001
        offset = 20.0
    else:  # SSS
        band_name = "salinity_0"
        scale_factor = 0.001
        offset = 20.0
    
    pt = ee.Geometry.Point(lon, lat)
    
    try:
        val = (img.select(band_name)
                  .reduceRegion(ee.Reducer.first(), pt, scale=20_000)
                  .get(band_name))
        # Apply scale and offset to get actual values
        # Formula: actual_value = (raw_value * scale_factor) + offset
        return ee.Number(val).multiply(scale_factor).add(offset).getInfo()
    except Exception:
        return float("nan")
# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main(var_type: str, infile: str) -> None:
    # Validate variable type
    var_type = var_type.upper()
    if var_type not in ["SST", "SSS"]:
        sys.exit(f"✗ Variable type must be 'SST' or 'SSS', not '{var_type}'")
    
    in_path = Path(infile).expanduser().resolve()
    if not in_path.is_file():
        sys.exit(f"✗ '{infile}' not found")
    
    # ../t_data/<base>_HYCOM_T_0.txt or _HYCOM_S_0.txt
    out_dir  = in_path.parent.parent / "t_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "T_0" if var_type == "SST" else "S_0"
    out_file = out_dir / f"{in_path.stem}_HYCOM_{suffix}.txt"
    
    ee.Initialize(project='ee-cnggao')
    hycom = ee.ImageCollection("HYCOM/sea_temp_salinity")
    
    # ── read track file ───────────────────────────────────────────────────────
    date_line = re.compile(r"^\d{8},")
    header, rows = [], []
    with in_path.open() as f:
        for line in f:
            if date_line.match(line):
                parts = [p.strip() for p in line.split(",")]
                rows.append(dict(
                    raw = line.rstrip("\n"),
                    ymd = parts[0],
                    lat = parse_latlon(parts[4]),
                    lon = parse_latlon(parts[5]),
                ))
            else:
                header.append(line.rstrip("\n"))
    
    df = pd.DataFrame(rows)
    
    # ── sample 31-day surface variable window ─────────────────────────────────
    window = range(-15, 16)  # –15…+15
    
    # Column names based on variable type
    if var_type == "SST":
        col_prefix = "water_temp"
        units = "°C"
    else:  # SSS
        col_prefix = "salinity"
        units = "PSU"
    
    var_cols = [f"{col_prefix}{d:+d}" for d in window]
    
    def surface_window(row):
        base = datetime.strptime(row.ymd, "%Y%m%d")
        return [get_daily_surface_value(hycom,
                                        (base + timedelta(days=off)).strftime("%Y%m%d"),
                                        row.lon, row.lat, var_type)
                for off in window]
    
    print(f"Processing {var_type} data (scale: 0.001, offset: 20) for {len(df)} records...")
    df[var_cols] = df.apply(surface_window, axis=1, result_type="expand")
    
    # ── write output ──────────────────────────────────────────────────────────
    with out_file.open("w") as f:
        # Write header
        for h in header:
            f.write(h + "\n")
        
        # Write data rows
        for _, r in df.iterrows():
            var_vals = ", ".join(f"{v:6.2f}" for v in r[var_cols])
            f.write(f"{r.raw}, {var_vals}\n")
    
    # print a friendly path
    try:
        display_path = out_file.relative_to(Path.cwd())
    except ValueError:
        display_path = os.path.relpath(out_file, Path.cwd())
    print(f"✓ Wrote {display_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:  python extract_HYCOM_surface.py <SST|SSS> <trackfile.txt>")
        print("  SST - Sea Surface Temperature (°C)")
        print("  SSS - Sea Surface Salinity (PSU)")
        sys.exit(1)
    
    if not sys.argv[2].lower().endswith(".txt"):
        sys.exit("✗ Track file must have .txt extension")
    
    main(sys.argv[1], sys.argv[2])
