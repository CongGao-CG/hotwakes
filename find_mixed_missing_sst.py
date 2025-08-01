#!/usr/bin/env python3
"""
find_mixed_missing_sst.py – Locate *_OISST.txt files that contain **rows with a
mix of valid and missing SST values**, and report the exact line numbers.

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
from read_hurricane_data import read_hurricane_data

def main():
    t_data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('t_data')
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")

    found = False
    for txt in sorted(t_data_dir.glob('*_OISST.txt')):
        df = read_hurricane_data(txt, hurricane_only=False)
        df = df.iloc[:, -31:]
        if any(df.isnull().any(axis=1) & ~df.isnull().all(axis=1)):
            found = True
            print(txt) 
        
    if not found:
        print(f"✓ No mixed rows detected in {t_data_dir}")


if __name__ == '__main__':
    main()
