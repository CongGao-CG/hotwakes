#!/usr/bin/env bash
# run_extract_HYCOM_TS.sh ── run extract_HYCOM_TS.py on every best-track file
# whose storm year (chars 5-8 of the filename) ≥ 1993 and <= 2023
# If ../zt_data/<basename>_HYCOM_<var>_<depth>.txt already exists, skip processing.
#
# Usage: ./run_extract_HYCOM_TS.sh <T|S> <depth>
#   T = temperature, S = salinity
#   depth = depth level (e.g., 0, 10, 20, etc.)
#
# Example: ./run_extract_HYCOM_TS.sh T 10

set -euo pipefail
shopt -s nullglob

# Check arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <T|S> <depth>"
    echo "  T = temperature, S = salinity"
    echo "  depth = depth level (e.g., 0, 10, 20)"
    exit 1
fi

VAR_TYPE="${1^^}"  # Convert to uppercase
DEPTH="$2"

# Validate variable type
if [[ "$VAR_TYPE" != "T" && "$VAR_TYPE" != "S" ]]; then
    echo "Error: Variable type must be 'T' (temperature) or 'S' (salinity)"
    exit 1
fi

# Validate depth is a number
if ! [[ "$DEPTH" =~ ^[0-9]+$ ]]; then
    echo "Error: Depth must be a number"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTDIR="$SCRIPT_DIR/../zt_data"

# Create output directory if it doesn't exist
mkdir -p "$OUTDIR"

echo "══════════════════════════════════════════════════════════════"
echo "Running HYCOM extraction for: ${VAR_TYPE} at depth ${DEPTH}m"
echo "══════════════════════════════════════════════════════════════"

for file in *.txt; do
    base=$(basename "$file")
    year=${base:4:4}                     # chars 5-8
    
    # Skip if year format is invalid
    [[ ${#year} -ne 4 || ! $year =~ ^[0-9]{4}$ ]] && continue
    
    # Skip if year is outside range
    (( year < 1993 || year > 2023 )) && continue
    
    # Output filename includes variable type and depth
    out_file="${OUTDIR}/${base%.txt}_HYCOM_${VAR_TYPE}_${DEPTH}.txt"
    
    if [[ -e "$out_file" ]]; then
        echo "▶ skipping  $file  (output exists)"
        continue
    fi
    
    echo "▶ processing $file  (year $year)"
    python extract_HYCOM_TS.py "$file" "$VAR_TYPE" "$DEPTH"
done

echo "══════════════════════════════════════════════════════════════"
echo "Extraction complete!"
