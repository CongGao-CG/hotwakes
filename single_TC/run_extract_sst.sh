#!/usr/bin/env bash
# run_extract_sst.sh  ––  call extract_sst.py on every best-track file
# whose storm year ≥ 1982.  Run this script from the single_TC directory.

set -euo pipefail
shopt -s nullglob

# Ensure we are in the directory that contains the .txt files
cd "$(dirname "$0")"

for file in *.txt; do
    # grab the *last* 4-digit number in the filename → storm year
    year=$(grep -oE '[0-9]{4}' <<<"$file" | tail -1)

    # skip if no year or year < 1982
    [[ -z $year || $year -lt 1982 ]] && continue

    echo "▶ processing $file  (year $year)"
    python extract_sst.py "$file"
done