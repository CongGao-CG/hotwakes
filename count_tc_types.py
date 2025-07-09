#!/usr/bin/env python3
"""
count_tc_types.py – Count occurrences of each tropical-cyclone status code
(HU, TS, EX, etc.) across all *_SST.txt files in ./t_data or a specified folder.

Usage
-----
$ python count_tc_types.py            # uses ./t_data
$ python count_tc_types.py /path/to/t_data
"""
import sys
from pathlib import Path
import re
from collections import Counter


def accumulate_counts(path: Path, counts: Counter):
    with path.open() as f:
        for line in f:
            if not re.match(r"^\d{8},", line):
                continue  # skip header/meta
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4 and parts[3]:
                counts[parts[3]] += 1


def main():
    t_data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('t_data')
    if not t_data_dir.is_dir():
        sys.exit(f"✗ Directory '{t_data_dir}' not found")

    counts = Counter()
    for txt in t_data_dir.glob('*_SST.txt'):
        accumulate_counts(txt, counts)

    total = sum(counts.values())
    if not total:
        print('No status codes found.')
        return

    print('Tropical cyclone status counts:')
    for status, n in counts.most_common():
        print(f'{status:>3} : {n}')
    print('-' * 20)
    print(f'Total : {total}')


if __name__ == '__main__':
    main()