#!/usr/bin/env bash
# run_extract_sst.sh ── run extract_sst.py on every best-track file
# whose storm year (chars 5-8 of the filename) ≥ 1982.

set -euo pipefail
shopt -s nullglob

# ensure we run in the directory that contains the .txt files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for file in *.txt; do
    base=$(basename "$file")
    year=${base:4:4}           # chars 5-8 (0-based slice 4..7)

    # guard against non-numeric or short names
    [[ ${#year} -ne 4 || ! $year =~ ^[0-9]{4}$ ]] && continue
    (( year < 1982 )) && continue

    echo "▶ processing $file  (year $year)"
    python extract_sst.py "$file"
done