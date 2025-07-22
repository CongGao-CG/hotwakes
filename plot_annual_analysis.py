#!/usr/bin/env python3
"""
plot_annual_analysis.py – Yearly counts of cold vs. hot ΔSST events by basin.

ΔSST definition
----------------
ΔSST = SST(Day 0) − mean[SST(Day −10 … −4)]

Layout
------
Two stacked sub-plots (same column):

* **a** – OISST source
* **b** – HYCOM source

Within each subplot two series are displayed:

* **cold** – ΔSST < 0  (blue)
* **hot**  – ΔSST > 0  (red)

Neutral rows (ΔSST == 0) are ignored.

Output
------
Saves *annual_analysis_<BASIN>.png* and *.pdf* in the current directory.

Usage
-----
python plot_annual_analysis.py BASIN [t_data_dir]
"""
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from sst_loader import load_windows_with_date

# -----------------------------------------------------------------------------
# Constants – indices within the 31-day SST window (columns 1-31)
# -----------------------------------------------------------------------------
IDX_M10 = 5  # Day −10
IDX_M4  = 11 # Day −4
IDX_0   = 15 # Day  0

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def compute_delta(arr32: np.ndarray) -> np.ndarray:
    """Compute ΔSST row-wise for an array of shape ``(n, 32)``."""
    sst = arr32[:, 1:]
    baseline = sst[:, IDX_M10:IDX_M4 + 1].mean(axis=1)
    return sst[:, IDX_0] - baseline


def extract_year(date_int) -> int | None:
    """Return 4-digit year from YYYYMMDD int, or ``None`` on failure."""
    try:
        y = int(str(int(date_int))[:4])
        return y
    except Exception:
        return None


def annual_counts(arr32: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (years, cold_counts, hot_counts).

    ``years`` is a 1-D sorted array of distinct years present in *arr32*.
    Counts correspond element-wise to ``years``.
    """
    if arr32.size == 0:
        return np.array([], int), np.array([], int), np.array([], int)

    years_all = np.array([extract_year(d) for d in arr32[:, 0] if extract_year(d) is not None])
    if years_all.size == 0:
        return np.array([], int), np.array([], int), np.array([], int)

    unique_years = np.unique(years_all)
    cold_counts = np.zeros_like(unique_years, dtype=int)
    hot_counts  = np.zeros_like(unique_years, dtype=int)

    delta = compute_delta(arr32)
    years_row = np.array([extract_year(d) for d in arr32[:, 0]])

    for i, yr in enumerate(unique_years):
        mask = years_row == yr
        cold_counts[i] = int(np.count_nonzero((delta < 0) & mask))
        hot_counts[i]  = int(np.count_nonzero((delta > 0) & mask))

    return unique_years, cold_counts, hot_counts

# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------

def plot_annual(ax: plt.Axes, years: np.ndarray, cold: np.ndarray, hot: np.ndarray,
                basin: str, source: str, panel: str):
    ax.plot(years, cold, color="blue", marker="o", lw=1.8, label="ΔSST<0 (cold)")
    ax.plot(years, hot,  color="red",  marker="o", lw=1.8, label="ΔSST>0 (hot)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Case count")
    ax.grid(True, ls=":")
    ax.legend(fontsize=8)

    title = f"ΔSST: Day 0 − mean(Day −10...−4) ({basin} + {source})"
    ax.set_title(title, fontsize=10)

    # Improve x-tick labels: show every n years if too crowded
    if years.size > 24:  # heuristic threshold
        step = int(np.ceil(years.size / 24))
        ax.set_xticks(years[::step])
    ax.tick_params(axis="x", rotation=45)

    ax.text(0.02, 0.96, f"$\\mathbf{{{panel}}}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=11)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_annual_analysis.py BASIN [t_data_dir]")

    basin = sys.argv[1].upper()
    t_data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("t_data")
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")

    sources = ["OISST", "HYCOM"]
    arrays: Dict[str, np.ndarray] = {}
    for src in sources:
        try:
            arrays[src] = load_windows_with_date(t_data_dir, basin=basin, source=src)
        except Exception as e:
            print(f"⚠ {src} load failed for basin {basin}: {e}")
            arrays[src] = np.empty((0, 32))

    if all(arr.size == 0 for arr in arrays.values()):
        sys.exit(f"✗ No valid SST windows for basin '{basin}' across OISST & HYCOM")

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    for ax, src, panel in zip(axes, sources, ["a", "b"]):
        years, cold, hot = annual_counts(arrays[src])
        if years.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue
        plot_annual(ax, years, cold, hot, basin, src, panel)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(f"annual_analysis_{basin}.{ext}"), dpi=300)
    print(f"✓ Figure saved as annual_analysis_{basin}.png and .pdf")
    plt.show()


if __name__ == "__main__":
    main()
