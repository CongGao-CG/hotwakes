#!/usr/bin/env bash
# run_extract_HYCOM_surface.sh ── run extract_HYCOM_surface.py on every best-track file
# whose storm year (chars 5-8 of the filename) ≥ 1993 and <= 2023
# 
# Usage: ./run_extract_HYCOM_surface.sh SST  or  ./run_extract_HYCOM_surface.sh SSS
#
# If ../t_data/<basename>_HYCOM_T_0.txt (for SST) or _HYCOM_S_0.txt (for SSS) 
# already exists, skip processing.

set -euo pipefail
shopt -s nullglob

# Check command line argument
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <SST|SSS>"
    echo "  SST - Sea Surface Temperature"
    echo "  SSS - Sea Surface Salinity"
    exit 1
fi

VAR_TYPE=$(echo "$1" | tr '[:lower:]' '[:upper:]')  # Convert to uppercase

if [[ "$VAR_TYPE" != "SST" && "$VAR_TYPE" != "SSS" ]]; then
    echo "Error: Variable type must be 'SST' or 'SSS', not '$1'"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTDIR="$SCRIPT_DIR/../t_data"

# Set output suffix based on variable type
if [[ "$VAR_TYPE" == "SST" ]]; then
    OUT_SUFFIX="_HYCOM_T_0.txt"
    VAR_DESC="temperature"
else
    OUT_SUFFIX="_HYCOM_S_0.txt"
    VAR_DESC="salinity"
fi

echo "Processing $VAR_TYPE ($VAR_DESC) data for track files..."
echo

for file in *.txt; do
    base=$(basename "$file")
    year=${base:4:4}                     # chars 5-8
    
    [[ ${#year} -ne 4 || ! $year =~ ^[0-9]{4}$ ]] && continue
    (( year < 1993 || year > 2023 )) && continue
    
    out_file="${OUTDIR}/${base%.txt}${OUT_SUFFIX}"
    
    if [[ -e "$out_file" ]]; then
        echo "▶ skipping  $file  (output exists)"
        continue
    fi
    
    echo "▶ processing $file  (year $year) for $VAR_TYPE"
    python extract_HYCOM_surface.py "$VAR_TYPE" "$file"
done

echo
echo "Done processing $VAR_TYPE data."
