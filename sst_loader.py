#!/usr/bin/env python3
"""
sst_loader.py - SST data loading utilities for tropical cyclone analysis
"""
import re
from pathlib import Path
from typing import List, Literal, Optional
import numpy as np

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
    source: Literal["OISST", "HYCOM"] = "OISST"
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
        - OISST: *_SST.txt files
        - HYCOM: *_HYCOM.txt files
        
    Returns
    -------
    np.ndarray
        Array of shape (n, 31) containing SST data for each storm
        
    Raises
    ------
    RuntimeError
        If no valid TS or HU SST windows are found
    ValueError
        If invalid basin or source is provided
    """
    # Validate source parameter
    if source not in SOURCE_SUFFIXES:
        raise ValueError(f"Invalid source '{source}'. Must be one of: {list(SOURCE_SUFFIXES.keys())}")
    
    # Validate basin parameter
    if basin is not None and basin not in BASIN_CODES:
        raise ValueError(f"Invalid basin '{basin}'. Must be one of: {list(BASIN_CODES.keys())}")
    
    rows: List[np.ndarray] = []
    pat = re.compile(r"^\d{8},")
    suffix = SOURCE_SUFFIXES[source]
    
    # Generate file patterns based on basin and source
    if basin is None or basin == "GL":
        # Load from all basins
        file_patterns = [f"*{suffix}"]
    else:
        # Load from specific basin codes
        basin_codes = BASIN_CODES[basin]
        file_patterns = [f"{code}*{suffix}" for code in basin_codes]
    
    # Process files matching the patterns
    for pattern in file_patterns:
        for txt in sorted(t_data_dir.glob(pattern)):
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
        if basin is None:
            basin_str = ""
        elif basin == "GL":
            basin_str = " for all basins (GL)"
        else:
            basin_str = f" for basin {basin}"
        source_str = f" from {source} source"
        raise RuntimeError(f'No TS or HU SST windows found{basin_str}{source_str}.')
    
    return np.stack(rows)


def load_windows_with_date(
    t_data_dir: Path,
    basin: Optional[Literal["AL", "EP", "WP", "IO", "SH", "GL"]] = None,
    source: Literal["OISST", "HYCOM"] = "OISST"
) -> np.ndarray:
    """
    Load 31-day SST windows for TS & HU storms, **including the date**.

    The returned array has shape (n, 32):
        - column 0: YYYYMMDD (int)
        - columns 1–31: SST values (float)

    Parameters
    ----------
    t_data_dir : Path
        Directory containing SST data files.
    basin : {"AL", "EP", "WP", "IO", "SH", "GL"}, optional
        Basin filter (see `load_windows` docs).
    source : {"OISST", "HYCOM"}, default "OISST"
        Data source suffix.

    Returns
    -------
    np.ndarray
        Array of shape (n, 32) with the date followed by 31 SST values.

    Raises
    ------
    RuntimeError
        If no valid TS or HU SST windows are found.
    ValueError
        If invalid basin or source is provided.
    """
    # Re-use validation from load_windows
    if source not in SOURCE_SUFFIXES:
        raise ValueError(f"Invalid source '{source}'. Must be one of: {list(SOURCE_SUFFIXES.keys())}")
    if basin is not None and basin not in BASIN_CODES:
        raise ValueError(f"Invalid basin '{basin}'. Must be one of: {list(BASIN_CODES.keys())}")

    rows: List[np.ndarray] = []
    pat = re.compile(r"^\d{8},")         # date at start of line
    suffix = SOURCE_SUFFIXES[source]

    # Build file patterns
    if basin is None or basin == "GL":
        file_patterns = [f"*{suffix}"]
    else:
        file_patterns = [f"{code}*{suffix}" for code in BASIN_CODES[basin]]

    for pattern in file_patterns:
        for txt in sorted(t_data_dir.glob(pattern)):
            with txt.open() as f:
                for line in f:
                    if not pat.match(line):
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    # Need date + status + 31 SSTs  → total ≥ 32 parts
                    if len(parts) < 32 or parts[3] not in VALID_STATUSES:
                        continue
                    try:
                        sst = np.array(parts[-31:], dtype=float)
                        date_val = int(parts[0])
                    except ValueError:
                        # skip malformed SST or date
                        continue
                    rows.append(np.concatenate(([date_val], sst)))

    if not rows:
        basin_str = "" if basin is None else (" for all basins (GL)" if basin == "GL" else f" for basin {basin}")
        raise RuntimeError(f"No TS or HU SST windows found{basin_str} from {source} source.")

    return np.stack(rows)
