#!/usr/bin/env python3
"""
sst_loader.py - SST data loading utilities for tropical cyclone analysis
"""
from typing import Optional, Literal
from pathlib import Path
import numpy as np
import pandas as pd
from read_hurricane_data import read_hurricane_data

VALID_STATUSES = {"TS", "HU"}

# Basin code mappings
BASIN_CODES = {
    "AL": ["AL"],
    "EP": ["EP", "CP"],
    "WP": ["WP"],
    "IO": ["IO"],
    "SH": ["SH"],
    "GL": ["*"]  # Global: all basins
}

# Source file suffix mappings
SOURCE_SUFFIXES = {
    "OISST": "_OISST.txt",
    "HYCOM": "_HYCOM_T_0.txt"
}

def load_windows(
    t_data_dir: Path,
    basin: Optional[Literal["AL", "EP", "WP", "IO", "SH", "GL"]] = None,
    source: Literal["OISST", "HYCOM"] = "OISST",
    with_date = False,
    with_name = False
) -> np.ndarray:
    """
    Load 31-day SST windows for TS & HU status storms only.
    
    Parameters
    ----------
    t_data_dir : Path
        Directory containing SST data files
    basin : {"AL", "EP", "WP", "IO", "SH", "GL"}, optional
        Basin filter:
        - AL: Atlantic (AL*.txt)
        - EP: Eastern Pacific (EP*.txt and CP*.txt)
        - WP: Western Pacific (WP*.txt)
        - IO: Indian Ocean (IO*.txt)
        - SH: Southern Hemisphere (SH*.txt)
        - GL: Global, all basins (*.txt)
        If None, loads from all basins
    source : {"OISST", "HYCOM"}, default "OISST"
        Data source:
        - OISST: *_OISST.txt files
        - HYCOM: *_HYCOM_T_0.txt files
        
    Returns
    -------
    with_date = False
        np.ndarray of shape (n, 31) containing SST data for each storm
    with_date = True
        np.ndarray of shape (n, 32):
            - column 0: YYYYMMDD (int)
            - columns 1–31: SST values (float) 
    Raises
    ------
    ValueError
        If invalid basin or source is provided
    """
    # Validate source parameter
    if source not in SOURCE_SUFFIXES:
        raise ValueError(f"Invalid source '{source}'. Must be one of: {list(SOURCE_SUFFIXES.keys())}")
    
    # Validate basin parameter
    if basin is not None and basin not in BASIN_CODES:
        raise ValueError(f"Invalid basin '{basin}'. Must be one of: {list(BASIN_CODES.keys())}")
    
    suffix = SOURCE_SUFFIXES[source]
    
    # Generate file patterns based on basin and source
    if basin is None or basin == "GL":
        # Load from all basins
        file_patterns = [f"*{suffix}"]
    else:
        # Load from specific basin codes
        basin_codes = BASIN_CODES[basin]
        file_patterns = [f"{code}*{suffix}" for code in basin_codes]

    data = None
    name = None
    # Process files matching the patterns
    for pattern in file_patterns:
        for txt in sorted(t_data_dir.glob(pattern)):
            df = read_hurricane_data(txt, hurricane_only=False, with_name=True)
            df = df[df['status'].isin(VALID_STATUSES)]
            sst = np.array(df.iloc[:, -31:])
            sid = df[['name', 'lon', 'lat', 'time']]
            if with_date:
                date = np.array(pd.to_datetime(df['time']).dt.strftime('%Y%m%d').astype('Int64'))
                sst  = np.hstack((date.reshape(-1, 1), sst))
            if data is None:
                data = sst
            else:
                data = np.vstack((data, sst))
            if name is None:
                name = sid
            else:
                name = pd.concat([name, sid], axis=0)
    
    if with_name:
        return data, name
    else: 
        return data

