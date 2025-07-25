#!/usr/bin/env python3
"""
plot_sst_window.py – 31-day SST anomalies for TS & HU fixes
(two panels, baseline = mean SST Day −10 ... −4).

Usage: python plot_sst_window.py basin [t_data_dir]

Panels
------
**a**  ΔSST = SST0 − mean(SST−10...−4) (basin + OISST)
**b**  ΔSST = SST0 − mean(SST−10...−4) (basin + HYCOM)

Line colours
------------
* **black** – Median (solid) & Mean (dashed) for *all* rows
* **blue**  – Median & Mean for ΔSST < 0
* **red**   – Median & Mean for ΔSST > 0

The plot is saved as *sst_window_stats_basin.png* and *.pdf* and displayed.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sst_loader import load_windows

# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────
def stats(arr: np.ndarray):
    if arr.size == 0:
        return np.full(31, np.nan), np.full(31, np.nan)
    return np.nanmedian(arr, axis=0), np.nanmean(arr, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helper
# ─────────────────────────────────────────────────────────────────────────────
def plot_sst_window(ax, data: np.ndarray, panel: str, basin: str, source: str):
    days = np.arange(-15, 16)
    idx_m10, idx_m4, idx0 = 5, 11, 15
    
    # baseline: mean Day −10 ... −4
    baseline = data[:, idx_m10:idx_m4+1].mean(axis=1, keepdims=True)
    anom = data - baseline
    delta = data[:, idx0] - baseline.squeeze()   # ΔSST definition for grouping
    
    grp_all = anom
    grp_neg = anom[delta < 0]
    grp_pos = anom[delta > 0]
    
    med_all, mean_all = stats(grp_all)
    med_neg, mean_neg = stats(grp_neg)
    med_pos, mean_pos = stats(grp_pos)
    
    # Plot lines
    ax.plot(days, med_all,  color='black', lw=1.8, label='Median (all)')
    ax.plot(days, mean_all, color='black', lw=1.8, ls='--', label='Mean (all)')
    ax.plot(days, med_neg,  color='blue',  lw=1.2, label='Median ΔSST<0')
    ax.plot(days, mean_neg, color='blue',  lw=1.2, ls='--', label='Mean ΔSST<0')
    ax.plot(days, med_pos,  color='red',   lw=1.2, label='Median ΔSST>0')
    ax.plot(days, mean_pos, color='red',   lw=1.2, ls='--', label='Mean ΔSST>0')
    
    ax.set_xlabel('Days from storm passage')
    ax.set_ylabel('Sea surface temperature anomaly (°C)')
    ax.set_title(f'ΔSST: Day 0 − mean(Day −10...−4) ({basin} + {source})', fontsize=10)
    ax.axvline(0, color='k', lw=0.8, alpha=0.6)
    ax.grid(True, ls=':')
    ax.text(0.02, 0.96, f'$\\mathbf{{{panel}}}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=11)
    ax.legend(loc='lower left', fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        sys.exit("Usage: python plot_sst_window.py basin [t_data_dir]")
    
    basin = sys.argv[1].upper()  # Convert to uppercase for consistency
    
    # Get t_data directory (use second argument or default)
    if len(sys.argv) > 2:
        t_data_dir = Path(sys.argv[2])
    else:
        t_data_dir = Path(__file__).with_name('t_data')
    
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")
    
    # Load data for specified basin from both sources
    try:
        data_oisst = load_windows(t_data_dir, basin=basin, source="OISST")
        data_hycom = load_windows(t_data_dir, basin=basin, source="HYCOM")
    except Exception as e:
        sys.exit(f"✗ Error loading data for basin '{basin}': {e}")
    
    # Create 2x1 subplot layout
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    
    # Plot both panels
    plot_sst_window(axes[0], data_oisst, 'a', basin, 'OISST')
    plot_sst_window(axes[1], data_hycom, 'b', basin, 'HYCOM')
    
    fig.tight_layout()
    
    # Save figure with basin name in filename
    for ext in ('png', 'pdf'):
        filename = f'plot/sst_window_stats_{basin}.{ext}'
        fig.savefig(Path(filename), dpi=300)
    
    print(f'✓ Figure saved as plot/sst_window_stats_{basin}.png and .pdf ({basin} basin: OISST vs HYCOM, TS & HU only)')
    plt.show()

if __name__ == '__main__':
    main()
