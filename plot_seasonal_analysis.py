#!/usr/bin/env python3
"""
plot_seasonal_analysis.py – Monthly counts of cold vs. hot ΔSST events by basin.

ΔSST definition
---------------
ΔSST = SST(Day 0) − mean[SST(Day −10 … −4)]

Layout
------
Two stacked subplots (same column):

* **a** – OISST source
* **b** – HYCOM source

Each subplot shows monthly counts of:

* **cold** – ΔSST < 0  (blue)
* **hot**  – ΔSST > 0  (red)

Neutral rows (ΔSST == 0) are ignored.

Files
-----
The figure is saved as *seasonal_analysis_<BASIN>.png* and *.pdf*.

Usage
-----
python plot_seasonal_analysis.py BASIN [t_data_dir]
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sst_loader import load_windows_with_date

# -----------------------------------------------------------------------------
# Window indices (columns 1‑31 correspond to days −15…+15)
# -----------------------------------------------------------------------------
IDX_M10 = 5   # Day −10
IDX_M4 = 11   # Day −4
IDX_0 = 15    # Day 0

# -----------------------------------------------------------------------------
# ΔSST computation
# -----------------------------------------------------------------------------

def compute_delta(arr32: np.ndarray) -> np.ndarray:
    sst = arr32[:, 1:]
    baseline = sst[:, IDX_M10 : IDX_M4 + 1].mean(axis=1)
    return sst[:, IDX_0] - baseline

# -----------------------------------------------------------------------------
# Extract month from YYYYMMDD integer
# -----------------------------------------------------------------------------

def extract_month(date_int) -> int | None:
    try:
        month = int(str(int(date_int))[4:6])
        return month if 1 <= month <= 12 else None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Monthly cold/hot counts
# -----------------------------------------------------------------------------

def monthly_counts(arr32: np.ndarray):
    cold = np.zeros(12, int)
    hot = np.zeros(12, int)
    if arr32.size == 0:
        return cold, hot

    months = np.array([extract_month(d) for d in arr32[:, 0]])
    delta = compute_delta(arr32)

    for m in range(1, 13):
        mask = months == m
        cold[m - 1] = int(np.count_nonzero((delta < 0) & mask))
        hot[m - 1]  = int(np.count_nonzero((delta > 0) & mask))
    return cold, hot

# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------

def plot_monthly(ax, cold, hot, basin: str, source: str, panel: str):
    months = np.arange(1, 13)

    # Cold in blue, hot in red
    ax.plot(months, cold, color="blue", marker="o", lw=1.8, label="ΔSST<0 (cold)")
    ax.plot(months, hot,  color="red",  marker="o", lw=1.8, label="ΔSST>0 (hot)")

    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Case count")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8)

    # Panel-specific title (user‑specified format)
    ax.set_title(f"ΔSST: Day 0 − mean(Day −10...−4) ({basin} + {source})", fontsize=10)

    # Panel label
    ax.text(0.02, 0.96, f"$\\mathbf{{{panel}}}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=11)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_seasonal_analysis.py BASIN [t_data_dir]")

    basin = sys.argv[1].upper()
    t_data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("t_data")
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")

    sources = ["OISST", "HYCOM"]
    data = {}
    for src in sources:
        try:
            arr = load_windows_with_date(t_data_dir, basin=basin, source=src)
        except Exception as e:
            print(f"⚠ {src} load failed for basin {basin}: {e}")
            arr = np.empty((0, 32))
        data[src] = arr

    if all(arr.size == 0 for arr in data.values()):
        sys.exit(f"✗ No data found for basin '{basin}' in OISST or HYCOM")

    # Figure: 2 rows, 1 column (independent axes)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    for ax, src, panel in zip(axes, sources, ["a", "b"]):
        cold, hot = monthly_counts(data[src])
        plot_monthly(ax, cold, hot, basin, src, panel)

    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(Path(f"plot/seasonal_analysis_{basin}.{ext}"), dpi=300)
    print(f"✓ Figure saved as plot/seasonal_analysis_{basin}.png and .pdf")

    plt.show()

if __name__ == "__main__":
    main()
