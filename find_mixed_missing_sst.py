#!/usr/bin/env python3
"""
find_mixed_missing_sst.py – Locate *_SST.txt files that contain **rows with a
mix of valid and missing SST values**, and report the exact line numbers.

A token is considered *missing* if, after stripping blanks, it equals any of
    • "nan"  (case-insensitive)
    • "-999" (legacy placeholder)
    • ""     (empty string)

Output format
-------------
<filename>: line <n>  (<first 20 chars of the row…>)
<filename>: line <m>  (…)

If no mixed rows are found the script prints a confirmation message.

Usage
-----
$ python find_mixed_missing_sst.py           # scan ./t_data
$ python find_mixed_missing_sst.py /path/to/t_data
"""
import sys
from pathlib import Path
import re

MISSING_STRINGS = {"nan", "", "-999"}

def is_missing(tok: str) -> bool:
    return tok.strip().lower() in MISSING_STRINGS


def mixed_rows(path: Path):
    """Yield (line_number, line_contents) for mixed missing/valid SST rows."""
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not re.match(r"^\d{8},", line):
                continue  # skip header/meta
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 31:
                continue  # not SST-augmented
            sst_tokens = parts[-31:]
            miss_mask = [is_missing(t) for t in sst_tokens]
            if any(miss_mask) and not all(miss_mask):
                yield lineno, line.rstrip("\n")


def main():
    t_data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('t_data')
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")

    found = False
    for txt in sorted(t_data_dir.glob('*_SST.txt')):
        for lineno, content in mixed_rows(txt):
            if not found:
                print("Files and rows with mixed missing/non-missing SST values:")
                found = True
            preview = content[:40] + ('…' if len(content) > 40 else '')
            print(f"{txt.name}: line {lineno}  ({preview})")

    if not found:
        print(f"✓ No mixed rows detected in {t_data_dir}")


if __name__ == '__main__':
    main()