#!/usr/bin/env bash
# run_extract_ocean_data.sh ── run extract_ocean_data.py on every best-track file
# for the appropriate year range based on dataset availability.
#
# Usage: ./run_extract_ocean_data.sh <dataset> <variable> [depth]
#
# Where:
#   dataset: HYCOM, OISST, or COPERNICUS
#   variable: T (temperature), S (salinity), or C (chlorophyll)
#   depth: optional depth in meters (default: 0 for surface)
#
# Examples:
#   ./run_extract_ocean_data.sh HYCOM T          # HYCOM SST
#   ./run_extract_ocean_data.sh HYCOM S 10       # HYCOM salinity at 10m
#   ./run_extract_ocean_data.sh OISST T          # OISST SST
#   ./run_extract_ocean_data.sh COPERNICUS C     # Chlorophyll-a
#
# Year ranges:
#   OISST: 1982-present
#   HYCOM: 1993-2023
#   COPERNICUS: varies (check dataset documentation)
#
# If output file already exists, skip processing.

set -euo pipefail
shopt -s nullglob

# ────────────────────────────────────────────────────────────────────────────
# Parse command line arguments
# ────────────────────────────────────────────────────────────────────────────
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <dataset> <variable> [depth]"
    echo
    echo "Datasets:"
    echo "  HYCOM      - HYCOM sea temperature and salinity"
    echo "  OISST      - NOAA OISST v2.1 sea surface temperature"
    echo "  COPERNICUS - Copernicus Ocean Color chlorophyll-a"
    echo
    echo "Variables:"
    echo "  T - Temperature (HYCOM, OISST)"
    echo "  S - Salinity (HYCOM only)"
    echo "  C - Chlorophyll-a (COPERNICUS only)"
    echo
    echo "Depth:"
    echo "  Optional depth in meters (default: 0 for surface)"
    echo "  Only used for HYCOM dataset"
    exit 1
fi

DATASET="$(echo "$1" | tr '[:lower:]' '[:upper:]')"    # Convert to uppercase
VARIABLE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"   # Convert to uppercase
DEPTH="${3:-0}"                                         # Default to 0 if not provided

# ────────────────────────────────────────────────────────────────────────────
# Validate inputs
# ────────────────────────────────────────────────────────────────────────────
# Validate dataset
if [[ "$DATASET" != "HYCOM" && "$DATASET" != "OISST" && "$DATASET" != "COPERNICUS" ]]; then
    echo "Error: Dataset must be 'HYCOM', 'OISST', or 'COPERNICUS'"
    exit 1
fi

# Validate variable for dataset
case "$DATASET" in
    HYCOM)
        if [[ "$VARIABLE" != "T" && "$VARIABLE" != "S" ]]; then
            echo "Error: HYCOM supports variables T (temperature) and S (salinity)"
            exit 1
        fi
        ;;
    OISST)
        if [[ "$VARIABLE" != "T" ]]; then
            echo "Error: OISST only supports variable T (temperature)"
            exit 1
        fi
        ;;
    COPERNICUS)
        if [[ "$VARIABLE" != "C" ]]; then
            echo "Error: COPERNICUS only supports variable C (chlorophyll-a)"
            exit 1
        fi
        ;;
esac

# Validate depth is a number
if ! [[ "$DEPTH" =~ ^[0-9]+$ ]]; then
    echo "Error: Depth must be a non-negative integer"
    exit 1
fi

# Force surface-only datasets to depth 0
if [[ "$DATASET" == "OISST" || "$DATASET" == "COPERNICUS" ]] && [[ "$DEPTH" != "0" ]]; then
    echo "Note: $DATASET only provides surface data. Using depth 0."
    DEPTH="0"
fi

# ────────────────────────────────────────────────────────────────────────────
# Setup directories and year ranges
# ────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine output directory based on depth
if [[ "$DEPTH" == "0" ]]; then
    OUTDIR="$SCRIPT_DIR/../t_data"
else
    OUTDIR="$SCRIPT_DIR/../zt_data"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

# Set year ranges based on dataset
case "$DATASET" in
    HYCOM)
        MIN_YEAR=1993
        MAX_YEAR=2023
        ;;
    OISST)
        MIN_YEAR=1982
        MAX_YEAR=9999  # No upper limit
        ;;
    COPERNICUS)
        MIN_YEAR=1998
        MAX_YEAR=9999  # No upper limit
        ;;
esac

# ────────────────────────────────────────────────────────────────────────────
# Determine output filename pattern
# ────────────────────────────────────────────────────────────────────────────
get_output_filename() {
    local base="$1"
    base="${base%.txt}"  # Remove .txt extension
    
    case "$DATASET" in
        HYCOM)
            if [[ "$DEPTH" == "0" ]]; then
                if [[ "$VARIABLE" == "T" ]]; then
                    echo "${OUTDIR}/${base}_HYCOM_T_0.txt"
                else
                    echo "${OUTDIR}/${base}_HYCOM_S_0.txt"
                fi
            else
                echo "${OUTDIR}/${base}_HYCOM_${VARIABLE}_${DEPTH}.txt"
            fi
            ;;
        OISST)
            echo "${OUTDIR}/${base}_OISST.txt"
            ;;
        COPERNICUS)
            echo "${OUTDIR}/${base}_COPERNICUS_CHLOR.txt"
            ;;
    esac
}

# ────────────────────────────────────────────────────────────────────────────
# Process files
# ────────────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "Running $DATASET extraction for: ${VARIABLE} at depth ${DEPTH}m"
echo "Processing years: ${MIN_YEAR} to ${MAX_YEAR}"
echo "══════════════════════════════════════════════════════════════════════"

processed=0
skipped=0

for file in *.txt; do
    base=$(basename "$file")
    year=${base:4:4}  # Extract year from filename (chars 5-8)
    
    # Skip if year format is invalid
    if [[ ${#year} -ne 4 || ! $year =~ ^[0-9]{4}$ ]]; then
        continue
    fi
    
    # Skip if year is outside dataset range
    if (( year < MIN_YEAR || year > MAX_YEAR )); then
        continue
    fi
    
    # Get output filename
    out_file=$(get_output_filename "$base")
    
    # Skip if output already exists
    if [[ -e "$out_file" ]]; then
        echo "▶ skipping  $file  (output exists)"
        ((skipped++))
        continue
    fi
    
    # Process the file
    echo "▶ processing $file  (year $year)"
    
    # Build command based on whether depth is specified
    if [[ "$DEPTH" == "0" ]]; then
        python extract_ocean_data.py "$DATASET" "$VARIABLE" "$file"
    else
        python extract_ocean_data.py "$DATASET" "$VARIABLE" "$DEPTH" "$file"
    fi
    
    ((processed++))
done

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════════════"
echo "Extraction complete!"
echo "  Processed: $processed files"
echo "  Skipped:   $skipped files (output already exists)"
echo "══════════════════════════════════════════════════════════════════════"
