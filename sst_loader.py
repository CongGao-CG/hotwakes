#!/usr/bin/env python3
"""
sst_loader.py - SST data loading utilities for tropical cyclone analysis
"""
import re
from pathlib import Path
from typing import List
import numpy as np

VALID_STATUSES = {"TS", "HU"}

def load_windows(t_data_dir: Path) -> np.ndarray:
    """
    Load 31-day SST windows for TS & HU status storms only.
    
    Parameters
    ----------
    t_data_dir : Path
        Directory containing *_SST.txt files
        
    Returns
    -------
    np.ndarray
        Array of shape (n, 31) containing SST data for each storm
        
    Raises
    ------
    RuntimeError
        If no valid TS or HU SST windows are found
    """
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
