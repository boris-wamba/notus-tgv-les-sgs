"""Clean raw data files: sort a time series, or convert decimal commas.

    python scripts/prepare_data.py sort Re1600_N512_cfl03_tf9
    python scripts/prepare_data.py comma van_rees
"""
import argparse

from tgvtools import io

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("action", choices=("sort", "comma"))
ap.add_argument("file")
a = ap.parse_args()
print("written:", (io.sort_time_series if a.action == "sort" else io.convert_decimal_comma)(a.file))
