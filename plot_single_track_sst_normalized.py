#!/usr/bin/env python3
"""
plot_single_track_sst_normalized.py – Plot normalized sea-surface temperatures from a single
*_OISST.txt best-track file produced by extract_sst.py.

The script draws every 31-day SST window (Day −15 … +15) as a separate coloured
line, *without* a legend, and shows the figure interactively.

Usage
-----
$ python plot_single_track_sst_normalized.py t_data/AL201984_LILI_49_OISST.txt

• The figure title is the basename of the input file.
• The plot is *not* saved to disk; it is only displayed.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from read_hurricane_data import read_hurricane_data

VALID_STATUSES = {"TS", "HU"}

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python plot_single_track_sst_normalized.py <*_OISST.txt>')

    txt_path = Path(sys.argv[1]).expanduser().resolve()
    if not txt_path.is_file():
        sys.exit(f"✗ '{txt_path}' not found")
    df     = read_hurricane_data(txt_path, hurricane_only=False)
    df     = df[df['status'].isin(VALID_STATUSES)]
    data   = np.array(df.iloc[:, -31:])
    row_mean = data.mean(axis=1, keepdims=True)
    row_std  = data.std(axis=1,  keepdims=True)
    data     = (data - row_mean) / row_std
    n = data.shape[0]
    days = np.arange(-15, 16)

    # colour cycle
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (sst, c) in enumerate(zip(data, colors)):
        ax.plot(days, sst, color=c, linewidth=1.2)

    ax.set_xlabel('Days from storm passage')
    ax.set_ylabel('Sea surface temperature (°C)')
    ax.set_title(txt_path.name)
    ax.axvline(0, color='k', linewidth=0.8, alpha=0.6)
    ax.grid(True, ls=':')

    plt.show()


if __name__ == '__main__':
    main()
