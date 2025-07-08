#!/usr/bin/env python3
"""
extract_sst.py  ––  append a 31-day SST window (NOAA OISST v2.1, °C)
to every record of a HURDAT-style best-track text file.

• Each fix gets SST(D−15)…SST(D)…SST(D+15)  => 31 new columns.
• Output is written one level up, in ../t_data/<base>_SST.txt

Example
-------
$ python extract_sst.py AL312020_IOTA_26.txt
→  ../t_data/AL312020_IOTA_26_SST.txt
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


def get_daily_sst(imgcol: ee.ImageCollection, ymd: str,
                  lon: float, lat: float) -> float:
    """Daily OISST v2.1 SST in °C for *ymd* at lon/lat, or NaN if masked."""
    d0 = ee.Date.fromYMD(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:]))
    img = imgcol.filterDate(d0, d0.advance(1, "day")).first()
    if img is None:
        return float("nan")

    pt = ee.Geometry.Point(lon, lat)
    try:
        val = (img.select("sst")
                  .reduceRegion(ee.Reducer.first(), pt, scale=20_000)
                  .get("sst"))
        return ee.Number(val).multiply(0.01).getInfo()   # 0.01 °C → °C
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────
def main(infile: str) -> None:
    in_path = Path(infile).expanduser().resolve()
    if not in_path.is_file():
        sys.exit(f"✗ '{infile}' not found")

    # ../t_data/<base>_SST.txt
    out_dir  = in_path.parent.parent / "t_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{in_path.stem}_SST.txt"

    ee.Initialize()
    oisst = ee.ImageCollection("NOAA/CDR/OISST/V2_1")

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

    # ── sample 31-day SST window ──────────────────────────────────────────────
    window   = range(-15, 16)                         # –15…+15
    sst_cols = [f"sst{d:+d}" for d in window]

    def sst_window(row):
        base = datetime.strptime(row.ymd, "%Y%m%d")
        return [get_daily_sst(oisst,
                              (base + timedelta(days=off)).strftime("%Y%m%d"),
                              row.lon, row.lat)
                for off in window]

    df[sst_cols] = df.apply(sst_window, axis=1, result_type="expand")

    # ── write output ──────────────────────────────────────────────────────────
    with out_file.open("w") as f:
        for h in header:
            f.write(h + "\n")
        for _, r in df.iterrows():
            sst_vals = ", ".join(f"{v:6.2f}" for v in r[sst_cols])
            f.write(f"{r.raw}, {sst_vals}\n")

    # print a friendly path
    try:
        display_path = out_file.relative_to(Path.cwd())
    except ValueError:
        display_path = os.path.relpath(out_file, Path.cwd())
    print(f"✓ Wrote {display_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].lower().endswith(".txt"):
        sys.exit("Usage:  python extract_sst.py <trackfile.txt>")
    main(sys.argv[1])