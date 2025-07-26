#!/usr/bin/env python3
"""
plot_sst_diff_pdfs.py – PDFs of three ΔSST metrics **for tropical‑storm and
hurricane fixes only** (status codes TS and HU) gathered from *_OISST.txt or *_HYCOM_T_0.txt files.
Panels
------
Row 1 (basin + OISST):
a : ΔT = SST(Day 0) − SST(Day −15)
b : ΔT = SST(Day 0) − SST(Day −10)
c : ΔT = SST(Day 0) − mean[SST(Day −10 … −4)]

Row 2 (basin + HYCOM):
d : ΔT = SST(Day 0) − SST(Day −15)
e : ΔT = SST(Day 0) − SST(Day −10)
f : ΔT = SST(Day 0) − mean[SST(Day −10 … −4)]

Key points
----------
* Only rows whose fourth column is **TS** or **HU** contribute.
* ΔT > 0 region filled **red**; ΔT < 0 **blue**.
* Bold panel letters at upper‑left; descriptive text centred.
* "XX.X % of ΔSST > 0" shown at upper‑right.
* Figure saved as *sst_diff_pdfs_basin.png* and *.pdf* and displayed.
Usage
-----
$ python plot_sst_diff_pdfs.py AL                    # AL basin, scans ./t_data
$ python plot_sst_diff_pdfs.py EP                    # EP basin, scans ./t_data
$ python plot_sst_diff_pdfs.py AL /path/to/t_data    # AL basin, custom path
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sst_loader import load_windows

try:
    from scipy.stats import gaussian_kde  # type: ignore
    HAVE_KDE = True
except ImportError:
    HAVE_KDE = False

# ─────────────────────────────────────────────────────────────────────────────
# plotting helper (updated to include basin and source info)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pdf(ax, data: np.ndarray, panel: str, desc: str, basin: str, source: str):
    data = data[np.isfinite(data)]
    if data.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    
    pct_pos = (data > 0).mean() * 100
    
    if HAVE_KDE and data.size >= 2:
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 400)
        y_vals = kde(x_vals)
        pos_mask = x_vals > 0
        ax.fill_between(x_vals[~pos_mask], 0, y_vals[~pos_mask], color='blue', alpha=0.4)
        ax.fill_between(x_vals[pos_mask],  0, y_vals[pos_mask],  color='red',  alpha=0.4)
        ax.plot(x_vals, y_vals, color='black', lw=1.2)
    else:
        counts, bins = np.histogram(data, bins='auto', density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        colors = ['red' if c > 0 else 'blue' for c in centers]
        ax.bar(centers, counts, width=np.diff(bins), color=colors, alpha=0.7, align='center')
    
    ax.set_xlabel('ΔSST (°C)')
    ax.set_ylabel('Probability density')
    ax.set_title(f'{desc} ({basin} + {source})', fontsize=10)
    ax.grid(True, ls=':')
    ax.text(0.02, 0.96, f'$\\mathbf{{{panel}}}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=11)
    ax.text(0.98, 0.95, f'{pct_pos:.1f}% of ΔSST > 0', transform=ax.transAxes,
            ha='right', va='top', fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        sys.exit("✗ Usage: python plot_sst_diff_pdfs.py BASIN [/path/to/t_data]")
    
    basin = sys.argv[1].upper()
    
    # Determine t_data directory
    if len(sys.argv) > 2:
        t_data_dir = Path(sys.argv[2])
    else:
        t_data_dir = Path(__file__).with_name('t_data')
    
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")
    
    # Load data for specified basin from both sources
    data_oisst = load_windows(t_data_dir, basin=basin, source="OISST")
    data_hycom = load_windows(t_data_dir, basin=basin, source="HYCOM")
    
    # Check if data exists for this basin
    if data_oisst.size == 0 and data_hycom.size == 0:
        sys.exit(f"✗ No data found for basin '{basin}'")
    
    # Calculate differences for both datasets
    idx0, idx_m15, idx_m10, idx_m4 = 15, 0, 5, 11
    
    # OISST differences
    if data_oisst.size > 0:
        diff_a_oisst = data_oisst[:, idx0] - data_oisst[:, idx_m15]
        diff_b_oisst = data_oisst[:, idx0] - data_oisst[:, idx_m10]
        diff_c_oisst = data_oisst[:, idx0] - data_oisst[:, idx_m10:idx_m4+1].mean(axis=1)
    else:
        diff_a_oisst = diff_b_oisst = diff_c_oisst = np.array([])
    
    # HYCOM differences
    if data_hycom.size > 0:
        diff_a_hycom = data_hycom[:, idx0] - data_hycom[:, idx_m15]
        diff_b_hycom = data_hycom[:, idx0] - data_hycom[:, idx_m10]
        diff_c_hycom = data_hycom[:, idx0] - data_hycom[:, idx_m10:idx_m4+1].mean(axis=1)
    else:
        diff_a_hycom = diff_b_hycom = diff_c_hycom = np.array([])
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    
    # Row 1: basin + OISST
    plot_pdf(axes[0, 0], diff_a_oisst, 'a', 'ΔSST: Day 0 − Day −15', basin, 'OISST')
    plot_pdf(axes[0, 1], diff_b_oisst, 'b', 'ΔSST: Day 0 − Day −10', basin, 'OISST')
    plot_pdf(axes[0, 2], diff_c_oisst, 'c', 'ΔSST: Day 0 − mean(Day −10…−4)', basin, 'OISST')
    
    # Row 2: basin + HYCOM
    plot_pdf(axes[1, 0], diff_a_hycom, 'd', 'ΔSST: Day 0 − Day −15', basin, 'HYCOM')
    plot_pdf(axes[1, 1], diff_b_hycom, 'e', 'ΔSST: Day 0 − Day −10', basin, 'HYCOM')
    plot_pdf(axes[1, 2], diff_c_hycom, 'f', 'ΔSST: Day 0 − mean(Day −10…−4)', basin, 'HYCOM')
    
    # Enable y-axis labels for all subplots
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(axis='y', labelleft=True)
    
    fig.tight_layout()
    
    # Save figure with basin in filename
    for ext in ('png', 'pdf'):
        filename = f'plot/sst_diff_pdfs_{basin}.{ext}'
        fig.savefig(Path(filename), dpi=300)
    
    print(f'✓ Figure saved as plot/sst_diff_pdfs_{basin}.png and .pdf ({basin} basin: OISST vs HYCOM, TS & HU only)')
    plt.show()

if __name__ == '__main__':
    main()
