#!/usr/bin/env python3
"""
extract_HYCOM_TS.py  ––  append a 31-day water temperature or salinity window 
(HYCOM sea_temp_salinity) at specified depth to every record of a HURDAT-style 
best-track text file.

• Each fix gets var_depth(D−15)…var_depth(D)…var_depth(D+15)  => 31 new columns.
• Output is written one level up, in ../zt_data/<base>_HYCOM_<var>_<depth>.txt
• HYCOM water_temp values are scaled: actual_temp = (value * 0.001) + 20
• HYCOM salinity values are scaled: actual_salinity = (value * 0.001) + 20
• HYCOM data is on a 0.08° lat/long grid

Usage
-----
$ python extract_HYCOM_TS.py <trackfile.txt> <T|S> <depth>
  where T = temperature, S = salinity
  depth = depth level (e.g., 0, 10, 20, etc.)

Examples
--------
$ python extract_HYCOM_TS.py AL312020_IOTA_26.txt T 10
→  ../zt_data/AL312020_IOTA_26_HYCOM_T_10.txt

$ python extract_HYCOM_TS.py AL312020_IOTA_26.txt S 10  
→  ../zt_data/AL312020_IOTA_26_HYCOM_S_10.txt
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

def get_daily_value(imgcol: ee.ImageCollection, ymd: str,
                    lon: float, lat: float, band_name: str) -> float:
    """Daily HYCOM value for *ymd* at lon/lat, or NaN if masked.
    
    HYCOM stores temperature and salinity as scaled integers: 
    actual_value = (value * 0.001) + 20
    Data is on a 0.08° grid (~8.9 km).
    """
    d0 = ee.Date.fromYMD(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
    img = imgcol.filterDate(d0, d0.advance(1, "day")).first()
    if img is None:
        return float("nan")
    pt = ee.Geometry.Point(lon, lat)
    try:
        val = (img.select(band_name)
                  .reduceRegion(ee.Reducer.first(), pt, scale=20_000)
                  .get(band_name))
        # Apply scale (0.001) and offset (20)
        return ee.Number(val).multiply(0.001).add(20).getInfo()
    except Exception:
        return float("nan")
# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main(infile: str, var_type: str, depth: str) -> None:
    in_path = Path(infile).expanduser().resolve()
    if not in_path.is_file():
        sys.exit(f"✗ '{infile}' not found")
    
    # Validate variable type
    var_type = var_type.upper()
    if var_type not in ['T', 'S']:
        sys.exit("✗ Variable type must be 'T' (temperature) or 'S' (salinity)")
    
    # Determine band name
    if var_type == 'T':
        band_name = f"water_temp_{depth}"
        var_label = "water_temp"
    else:
        band_name = f"salinity_{depth}"
        var_label = "salinity"
    
    # ../zt_data/<base>_HYCOM_<var>_<depth>.txt
    out_dir  = in_path.parent.parent / "zt_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{in_path.stem}_HYCOM_{var_type}_{depth}.txt"
    
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
    
    # ── sample 31-day window ──────────────────────────────────────────────────
    window = range(-15, 16)  # –15…+15
    col_prefix = f"{var_label}_{depth}"
    val_cols = [f"{col_prefix}{d:+d}" for d in window]
    
    def value_window(row):
        base = datetime.strptime(row.ymd, "%Y%m%d")
        return [get_daily_value(hycom,
                               (base + timedelta(days=off)).strftime("%Y%m%d"),
                               row.lon, row.lat, band_name)
                for off in window]
    
    df[val_cols] = df.apply(value_window, axis=1, result_type="expand")
    
    # ── write output ──────────────────────────────────────────────────────────
    with out_file.open("w") as f:
        for h in header:
            f.write(h + "\n")
        for _, r in df.iterrows():
            val_str = ", ".join(f"{v:6.2f}" for v in r[val_cols])
            f.write(f"{r.raw}, {val_str}\n")
    
    # print a friendly path
    try:
        display_path = out_file.relative_to(Path.cwd())
    except ValueError:
        display_path = os.path.relpath(out_file, Path.cwd())
    print(f"✓ Wrote {display_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage:  python extract_HYCOM_TS.py <trackfile.txt> <T|S> <depth>")
    if not sys.argv[1].lower().endswith(".txt"):
        sys.exit("✗ First argument must be a .txt file")
    main(sys.argv[1], sys.argv[2], sys.argv[3])
