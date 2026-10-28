"""Plot time series (e.g. mean dissipation rate vs time) from several .dat files.

    python scripts/plot_series.py --out mean_kedr_dns.png \
        --series "DNS 64^3=Re1600_N64_cfl03_tf15.dat" \
        --series "DNS 512^3=Re1600_N512_cfl03_tf9.dat:0:1" \
        --reference "Van Rees=van_rees.dat:0:1"

``LABEL=path[:time_col:value_col]`` (default columns 0 and 2).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tgvtools.io import load_columns


def parse(spec: str):
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    cols = (int(parts[1]), int(parts[2])) if len(parts) == 3 else (0, 2)
    return label, parts[0], cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", default=[], metavar="LABEL=path[:tcol:ycol]")
    ap.add_argument("--reference", metavar="LABEL=path[:tcol:ycol]")
    ap.add_argument("--ylabel", default=r"$\langle 2\nu S_{ij}S_{ij}\rangle$")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7, 7))
    for spec in a.series:
        label, path, cols = parse(spec)
        t, y = load_columns(path, cols)
        ax.plot(t, y, label=label)
    if a.reference:
        label, path, cols = parse(a.reference)
        t, y = load_columns(path, cols)
        ax.plot(t, y, "k--", lw=1.5, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(a.ylabel)
    ax.legend(loc="upper left")
    fig.savefig(a.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
