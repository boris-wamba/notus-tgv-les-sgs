"""Compare modelled subgrid viscosity (or production) of LES runs with the ideal value from a filtered DNS.

    python scripts/plot_sgs_viscosity.py --out nu_sgs_dx.png --nu 6.25e-4 \
        --dns fil-dns_Re1600_N512_cfl03_tf15.dat \
        --les "SMG Delta=dx=smg-dx-Re1600_N256_cfl03_tf15.dat" \
        --les "MM Delta=dx=mm-dx-Re1600_N256_cfl03_tf15.dat"

The DNS file must follow tgvtools.io.LAYOUT_DNS_FILTERED, the LES files tgvtools.io.LAYOUT_LES.
Use --quantity production to plot 2 nu_sgs S:S against eps - eps_bar instead of nu_sgs.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tgvtools import io, sgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dns", required=True, help="filtered-DNS dissipation file")
    ap.add_argument("--les", action="append", default=[], metavar="LABEL=path")
    ap.add_argument("--nu", type=float, default=1.0 / 1600.0, help="molecular viscosity (1/Re)")
    ap.add_argument("--quantity", choices=("nu", "production"), default="nu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t, eps, eps_f = io.load_layout(a.dns, io.LAYOUT_DNS_FILTERED, "mean_kedr", "mean_kedr_filtered")
    ideal = sgs.nu_sgs_ideal(eps, eps_f, a.nu) if a.quantity == "nu" else sgs.sgs_dissipation(eps, eps_f)
    key = "nu_sgs" if a.quantity == "nu" else "production"

    fig, ax = plt.subplots(figsize=(7, 7))
    for spec in a.les:
        label, path = spec.split("=", 1)
        tl, y = io.load_layout(path, io.LAYOUT_LES, key)
        ax.plot(tl, y, marker="o", markevery=0.1, label=label)
    ax.plot(t, ideal, "k-", marker="d", markevery=0.1,
            label=r"ideal $\langle\nu_{sgs}\rangle$" if a.quantity == "nu" else r"$\epsilon-\bar\epsilon$")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\langle\nu_{sgs}\rangle$" if a.quantity == "nu" else r"$\langle 2\nu_{sgs}\bar S\bar S\rangle$")
    ax.legend()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
