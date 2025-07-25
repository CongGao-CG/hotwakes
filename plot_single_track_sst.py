#!/usr/bin/env python3
"""
plot_single_track_sst.py – Plot raw sea-surface temperatures from a single
*_SST.txt best-track file produced by extract_sst.py.
The script draws every 31-day SST window (Day −15 … +15) as a separate coloured
line with a legend showing date/time labels (MMDDHHMM format).

**Updated**: 
- Only processes records with status TS or HU in the fourth column
- Excludes time series with NaN values
- Only plots hot wakes where ΔT > 0, with ΔT = SST(Day 0) - mean[SST(Day -10 ... -4)]
- Shows legend with date/time labels for each line

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
def load_windows(txt_path: Path) -> tuple[np.ndarray, list[str]]:
    """Return array (n, 31) of SSTs and list of date/time labels from one *_SST.txt file.
    
    Only includes records where:
    - Fourth column contains 'TS' or 'HU'
    - No NaN values present
    - ΔT > 0 (hot wake), where ΔT = SST(Day 0) - mean[SST(Day -10 ... -4)]
    
    Returns:
        tuple: (sst_array, labels) where labels are formatted as MMDDHHMM
    """
    windows = []
    labels = []
    with txt_path.open() as f:
        for line in f:
            if not line[:8].isdigit():
                continue  # skip header/meta
            parts = [p.strip() for p in line.split(',')]
            
            # Check if fourth column (index 3) is TS or HU
            if len(parts) < 4:
                continue
            if parts[3] not in ['TS', 'HU']:
                continue
                
            if len(parts) < 31:
                continue
            try:
                sst = np.array(parts[-31:], dtype=float)
            except ValueError:
                continue
            
            # Skip if any NaN values present
            if np.any(np.isnan(sst)):
                continue
            
            # Calculate ΔT = SST(Day 0) - mean[SST(Day -10 ... -4)]
            # Day 0 is at index 15, Day -10 is at index 5, Day -4 is at index 11
            sst_day0 = sst[15]
            sst_pre = sst[5:12]  # indices 5 through 11 (Day -10 to Day -4)
            delta_t = sst_day0 - np.mean(sst_pre)
            
            # Only keep hot wakes (ΔT > 0)
            if delta_t > 0:
                windows.append(sst)
                # Format date/time label as MMDDHHMM
                date_str = parts[0]  # YYYYMMDD
                time_str = parts[1]  # HHMM
                label = date_str[4:] + time_str  # MMDDHHMM
                labels.append(label)
                
    if not windows:
        raise RuntimeError(f'No hot wake SST windows with TS/HU status found in {txt_path}')
    return np.stack(windows), labels
# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python plot_single_track_sst.py <*_SST.txt>')
    txt_path = Path(sys.argv[1]).expanduser().resolve()
    if not txt_path.is_file():
        sys.exit(f"✗ '{txt_path}' not found")
    data, labels = load_windows(txt_path)
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
