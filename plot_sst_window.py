#!/usr/bin/env python3
"""
plot_sst_window.py – 31-day SST anomalies for TS & HU fixes
(single panel, baseline = mean SST Day −10 … −4).

Panel
-----
**c**  ΔSST = SST₀ − mean(SST₋₁₀…₋₄)

Line colours
------------
* **black** – Median (solid) & Mean (dashed) for *all* rows
* **blue**  – Median & Mean for ΔSST < 0
* **red**   – Median & Mean for ΔSST > 0

The plot is saved as *sst_window_stats.png* and *.pdf* and displayed.
"""
import sys, re
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

VALID_STATUSES = {"TS", "HU"}

# ─────────────────────────────────────────────────────────────────────────────
# Load 31-day windows (TS & HU only)
# ─────────────────────────────────────────────────────────────────────────────

def load_windows(t_data_dir: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    pat = re.compile(r"^\d{8},")
    for txt in sorted(t_data_dir.glob('*_SST.txt')):
        with txt.open() as f:
            for line in f:
                if not pat.match(line):
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
        raise RuntimeError('No TS or HU SST windows found.')
    return np.stack(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────

def stats(arr: np.ndarray):
    if arr.size == 0:
        return np.full(31, np.nan), np.full(31, np.nan)
    return np.nanmedian(arr, axis=0), np.nanmean(arr, axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('t_data')
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")

    data = load_windows(t_data_dir)  # (n, 31)
    days = np.arange(-15, 16)

    idx_m10, idx_m4, idx0 = 5, 11, 15

    # baseline: mean Day −10 … −4
    baseline = data[:, idx_m10:idx_m4+1].mean(axis=1, keepdims=True)
    anom = data - baseline

    delta = data[:, idx0] - baseline.squeeze()   # ΔSST definition for grouping

    grp_all = anom
    grp_neg = anom[delta < 0]
    grp_pos = anom[delta > 0]

    med_all, mean_all = stats(grp_all)
    med_neg, mean_neg = stats(grp_neg)
    med_pos, mean_pos = stats(grp_pos)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(days, med_all,  color='black', lw=1.8, label='Median (all)')
    ax.plot(days, mean_all, color='black', lw=1.8, ls='--', label='Mean (all)')
    ax.plot(days, med_neg,  color='blue',  lw=1.2, label='Median ΔSST<0')
    ax.plot(days, mean_neg, color='blue',  lw=1.2, ls='--', label='Mean ΔSST<0')
    ax.plot(days, med_pos,  color='red',   lw=1.2, label='Median ΔSST>0')
    ax.plot(days, mean_pos, color='red',   lw=1.2, ls='--', label='Mean ΔSST>0')

    ax.set_xlabel('Days from storm passage')
    ax.set_ylabel('Sea surface temperature anomaly (°C)')
    ax.set_title('ΔSST: Day 0 − mean(Day −10…−4)', fontsize=10)
    ax.axvline(0, color='k', lw=0.8, alpha=0.6)
    ax.grid(True, ls=':')
    ax.text(0.02, 0.96, '$\\mathbf{c}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=11)
    ax.legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(Path('sst_window_stats.' + ext), dpi=300)
    print('✓ Figure saved as sst_window_stats.png and .pdf (TS & HU only)')

    plt.show()


if __name__ == '__main__':
    main()