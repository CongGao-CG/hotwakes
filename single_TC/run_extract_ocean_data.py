#!/usr/bin/env python3
"""
run_extract_ocean_data.py - Run extract_ocean_data.py on every best-track file
for the appropriate year range based on dataset availability.

Usage: python run_extract_ocean_data.py <dataset> <variable> [depth]

Where:
  dataset: HYCOM, OISST, or COPERNICUS
  variable: T (temperature), S (salinity), or C (chlorophyll)
  depth: optional depth in meters (default: 0 for surface)

Examples:
  python run_extract_ocean_data.py HYCOM T          # HYCOM SST
  python run_extract_ocean_data.py HYCOM S 10       # HYCOM salinity at 10m
  python run_extract_ocean_data.py OISST T          # OISST SST
  python run_extract_ocean_data.py COPERNICUS C     # Chlorophyll-a

Year ranges:
  OISST: 1982-present
  HYCOM: 1993-2023
  COPERNICUS: varies (check dataset documentation)

If output file already exists, skip processing.
"""

import sys
import os
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional


# Configuration
DATASET_CONFIG = {
    'HYCOM': {
        'min_year': 1993,
        'max_year': 2023,
        'variables': ['T', 'S'],
        'description': 'HYCOM sea temperature and salinity'
    },
    'OISST': {
        'min_year': 1982,
        'max_year': 9999,
        'variables': ['T'],
        'description': 'NOAA OISST v2.1 sea surface temperature'
    },
    'COPERNICUS': {
        'min_year': 1997,
        'max_year': 9999,
        'variables': ['C'],
        'description': 'Copernicus Ocean Color chlorophyll-a'
    }
}


def print_header(dataset: str, variable: str, depth: int) -> None:
    """Print processing header."""
    config = DATASET_CONFIG[dataset]
    print("=" * 70)
    print(f"Running {dataset} extraction for: {variable} at depth {depth}m")
    print(f"Processing years: {config['min_year']} to {config['max_year']}")
    print("=" * 70)


def print_usage() -> None:
    """Print usage information."""
    print("Usage: python run_extract_ocean_data.py <dataset> <variable> [depth]")
    print()
    print("Datasets:")
    for name, config in DATASET_CONFIG.items():
        print(f"  {name:<10} - {config['description']}")
    print()
    print("Variables:")
    print("  T - Temperature (HYCOM, OISST)")
    print("  S - Salinity (HYCOM only)")
    print("  C - Chlorophyll-a (COPERNICUS only)")
    print()
    print("Depth:")
    print("  Optional depth in meters (default: 0 for surface)")
    print("  Only used for HYCOM dataset")


def validate_inputs(dataset: str, variable: str, depth: str) -> Tuple[str, str, int]:
    """Validate and normalize inputs."""
    # Normalize to uppercase
    dataset = dataset.upper()
    variable = variable.upper()
    
    # Validate dataset
    if dataset not in DATASET_CONFIG:
        print(f"Error: Dataset must be one of {list(DATASET_CONFIG.keys())}")
        sys.exit(1)
    
    # Validate variable for dataset
    if variable not in DATASET_CONFIG[dataset]['variables']:
        valid_vars = DATASET_CONFIG[dataset]['variables']
        print(f"Error: {dataset} supports variables: {', '.join(valid_vars)}")
        sys.exit(1)
    
    # Validate and convert depth
    try:
        depth_int = int(depth)
        if depth_int < 0:
            raise ValueError
    except ValueError:
        print("Error: Depth must be a non-negative integer")
        sys.exit(1)
    
    # Force surface-only datasets to depth 0
    if dataset in ['OISST', 'COPERNICUS'] and depth_int != 0:
        print(f"Note: {dataset} only provides surface data. Using depth 0.")
        depth_int = 0
    
    return dataset, variable, depth_int


def get_output_filename(base: str, dataset: str, variable: str, depth: int) -> Path:
    """Determine output filename based on dataset and parameters."""
    # Remove .txt extension
    base = base.replace('.txt', '')
    
    # Determine output directory
    if depth == 0:
        out_dir = Path("..") / "t_data"
    else:
        out_dir = Path("..") / "zt_data"
    
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename based on dataset
    if dataset == 'HYCOM':
        if depth == 0:
            suffix = "T_0" if variable == 'T' else "S_0"
        else:
            suffix = f"{variable}_{depth}"
        filename = f"{base}_HYCOM_{suffix}.txt"
    elif dataset == 'OISST':
        filename = f"{base}_OISST.txt"
    else:  # COPERNICUS
        filename = f"{base}_COPERNICUS_CHLOR.txt"
    
    return out_dir / filename


def extract_year_from_filename(filename: str) -> Optional[int]:
    """Extract year from filename (chars 5-8)."""
    if len(filename) < 8:
        return None
    
    year_str = filename[4:8]
    if re.match(r'^\d{4}$', year_str):
        return int(year_str)
    return None


def run_extraction(trackfile: str, dataset: str, variable: str, depth: int) -> bool:
    """Run the extraction for a single file."""
    # Build command
    cmd = ['python', 'extract_ocean_data.py', dataset, variable]
    
    # Add depth if not surface
    if depth > 0:
        cmd.append(str(depth))
    
    cmd.append(trackfile)
    
    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Print the output from the extraction script
            if result.stdout:
                print(result.stdout.strip())
            return True
        else:
            print(f"  Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  Error running extraction: {e}")
        return False


def main():
    """Main processing function."""
    # Parse arguments
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    dataset_arg = sys.argv[1]
    variable_arg = sys.argv[2]
    depth_arg = sys.argv[3] if len(sys.argv) > 3 else "0"
    
    # Validate inputs
    dataset, variable, depth = validate_inputs(dataset_arg, variable_arg, depth_arg)
    
    # Print header
    print_header(dataset, variable, depth)
    
    # Get configuration
    config = DATASET_CONFIG[dataset]
    min_year = config['min_year']
    max_year = config['max_year']
    
    # Process statistics
    processed = 0
    skipped = 0
    errors = 0
    
    # Process all .txt files in current directory
    for filename in sorted(os.listdir('.')):
        if not filename.endswith('.txt'):
            continue
        
        # Extract year from filename
        year = extract_year_from_filename(filename)
        if year is None:
            continue
        
        # Skip if year is outside dataset range
        if year < min_year or year > max_year:
            continue
        
        # Get output filename
        out_file = get_output_filename(filename, dataset, variable, depth)
        
        # Skip if output already exists
        if out_file.exists():
            print(f"▶ skipping  {filename}  (output exists)")
            skipped += 1
            continue
        
        # Process the file
        print(f"▶ processing {filename}  (year {year})")
        
        if run_extraction(filename, dataset, variable, depth):
            processed += 1
        else:
            errors += 1
    
    # Print summary
    print("=" * 70)
    print("Extraction complete!")
    print(f"  Processed: {processed} files")
    print(f"  Skipped:   {skipped} files (output already exists)")
    if errors > 0:
        print(f"  Errors:    {errors} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
