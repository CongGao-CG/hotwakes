#!/usr/bin/env python3
"""
plot_single_track_sst.py – Plot raw sea-surface temperatures from a single
*_SST.txt best-track file produced by extract_sst.py.

The script draws every 31-day SST window (Day −15 … +15) as a separate coloured
line, *without* a legend, and shows the figure interactively.

Usage
-----
$ python plot_single_track_sst.py AL201984_LILI_49_SST.txt

• The figure title is the basename of the input file.
• The plot is *not* saved to disk; it is only displayed.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# load SST data from a single file
# ─────────────────────────────────────────────────────────────────────────────

def load_windows(txt_path: Path) -> np.ndarray:
    """Return array (n, 31) of SSTs from one *_SST.txt file."""
    windows = []
    with txt_path.open() as f:
        for line in f:
            if not line[:8].isdigit():
                continue  # skip header/meta
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 31:
                continue
            try:
                sst = np.array(parts[-31:], dtype=float)
            except ValueError:
                continue
            windows.append(sst)
    if not windows:
        raise RuntimeError(f'No SST windows found in {txt_path}')
    return np.stack(windows)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python plot_single_track_sst.py <*_SST.txt>')

    txt_path = Path(sys.argv[1]).expanduser().resolve()
    if not txt_path.is_file():
        sys.exit(f"✗ '{txt_path}' not found")

    data = load_windows(txt_path)
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