#!/usr/bin/env bash
# run_extract_sst.sh ── run extract_HYCOM_sst.py on every best-track file
# whose storm year (chars 5-8 of the filename) ≥ 1993 and <= 2023
# If ../t_data/<basename>_HYCOM.txt already exists, skip processing.

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTDIR="$SCRIPT_DIR/../t_data"

for file in *.txt; do
    base=$(basename "$file")
    year=${base:4:4}                     # chars 5-8
    [[ ${#year} -ne 4 || ! $year =~ ^[0-9]{4}$ ]] && continue
    (( year < 1982 || year > 2023 )) && continue

    out_file="${OUTDIR}/${base%.txt}_HYCOM.txt"
    if [[ -e "$out_file" ]]; then
        echo "▶ skipping  $file  (output exists)"
        continue
    fi

    echo "▶ processing $file  (year $year)"
    python extract_HYCOM_sst.py "$file"
done
