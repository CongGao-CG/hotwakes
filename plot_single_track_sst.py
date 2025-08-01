#!/usr/bin/env python3
"""
plot_single_track_sst.py – Plot raw sea-surface temperatures from a single
*_OISST.txt best-track file produced by extract_sst.py.
The script draws every 31-day SST window (Day −15 … +15) as a separate coloured
line with a legend showing date/time labels (MMDDHHMM format).

**Updated**: 
- Only processes records with status TS or HU in the fourth column
- Excludes time series with NaN values
- Only plots hot wakes where ΔT > 0, with ΔT = SST(Day 0) - mean[SST(Day -10 ... -4)]
- Shows legend with date/time labels for each line

Usage
-----
$ python plot_single_track_sst.py t_data/AL201984_LILI_49_OISST.txt
• The figure title is the basename of the input file.
• The plot is *not* saved to disk; it is only displayed.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_hurricane_data import read_hurricane_data

VALID_STATUSES = {"TS", "HU"}

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python plot_single_track_sst.py <*_OISST.txt>')
    txt_path = Path(sys.argv[1]).expanduser().resolve()
    if not txt_path.is_file():
        sys.exit(f"✗ '{txt_path}' not found")
    df     = read_hurricane_data(txt_path, hurricane_only=False)
    df     = df[df['status'].isin(VALID_STATUSES)]
    data   = np.array(df.iloc[:, -31:])
    labels = pd.to_datetime(df['time']).dt.strftime('%m%d%H%M')
    sst_day0 = data[:,15]
    sst_prem = data[:,5:12].mean(axis=1)
    data   = data[sst_day0 > sst_prem,:]
    labels = labels[sst_day0 > sst_prem].to_list()
    n = data.shape[0]
    days = np.arange(-15, 16)
    # colour cycle
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (sst, c, label) in enumerate(zip(data, colors, labels)):
        ax.plot(days, sst, color=c, linewidth=1.2, label=label)
    ax.set_xlabel('Days from storm passage')
    ax.set_ylabel('Sea surface temperature (°C)')
    ax.set_title(f'{txt_path.name} - Hot wakes (n={n})')
    ax.axvline(0, color='k', linewidth=0.8, alpha=0.6)
    ax.grid(True, ls=':')
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()
