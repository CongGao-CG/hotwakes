#!/usr/bin/env python3
"""
plot_sst_diff_pdfs.py – PDFs of three ΔSST metrics **for tropical‑storm and
hurricane fixes only** (status codes TS and HU) gathered from *_SST.txt files.

Panels
------
a : ΔT = SST(Day 0) − SST(Day −15)
b : ΔT = SST(Day 0) − SST(Day −10)
c : ΔT = SST(Day 0) − mean[SST(Day −10 … −4)]

Key points
----------
* Only rows whose fourth column is **TS** or **HU** contribute.
* ΔT > 0 region filled **red**; ΔT < 0 **blue**.
* Bold panel letters at upper‑left; descriptive text centred.
* “XX.X % of ΔSST > 0” shown at upper‑right.
* Figure saved as *sst_diff_pdfs.png* and *.pdf* and displayed.

Usage
-----
$ python plot_sst_diff_pdfs.py            # scans ./t_data
$ python plot_sst_diff_pdfs.py /path/to/t_data
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List

try:
    from scipy.stats import gaussian_kde  # type: ignore
    HAVE_KDE = True
except ImportError:
    HAVE_KDE = False

VALID_STATUSES = {"TS", "HU"}

# ─────────────────────────────────────────────────────────────────────────────
# data loader (filter TS & HU)
# ─────────────────────────────────────────────────────────────────────────────

def load_windows(t_data_dir: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    for txt in sorted(t_data_dir.glob('*_SST.txt')):
        with txt.open() as f:
            for line in f:
                if not re.match(r"^\d{8},", line):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 31 or parts[3] not in VALID_STATUSES:
                    continue
                try:
                    sst = np.array(parts[-31:], dtype=float)
                except ValueError:
                    continue
                rows.append(sst)
    if not rows:
        raise RuntimeError('No TS or HU rows with SST data found.')
    return np.stack(rows)

# ─────────────────────────────────────────────────────────────────────────────
# plotting helper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pdf(ax, data: np.ndarray, panel: str, desc: str):
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
    ax.set_title(desc, fontsize=10)
    ax.grid(True, ls=':')

    ax.text(0.02, 0.96, f'$\\mathbf{{{panel}}}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=11)
    ax.text(0.98, 0.95, f'{pct_pos:.1f}% of ΔSST > 0', transform=ax.transAxes,
            ha='right', va='top', fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('t_data')
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")

    data = load_windows(t_data_dir)

    idx0, idx_m15, idx_m10, idx_m4 = 15, 0, 5, 11
    diff_a = data[:, idx0] - data[:, idx_m15]
    diff_b = data[:, idx0] - data[:, idx_m10]
    diff_c = data[:, idx0] - data[:, idx_m10:idx_m4+1].mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    plot_pdf(axes[0], diff_a, 'a', 'ΔSST: Day 0 − Day −15')
    plot_pdf(axes[1], diff_b, 'b', 'ΔSST: Day 0 − Day −10')
    plot_pdf(axes[2], diff_c, 'c', 'ΔSST: Day 0 − mean(Day −10…−4)')

    for ax in axes:
        ax.tick_params(axis='y', labelleft=True)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(Path(f'sst_diff_pdfs.{ext}'), dpi=300)
    print('✓ Figure saved as sst_diff_pdfs.png and .pdf (TS & HU only)')

    plt.show()


if __name__ == '__main__':
    main()
