"""3D kinetic-energy spectra of Notus TGV outputs (.bp files).

    python scripts/spectrum3d.py --case 64=output/case64/tgv_00000325.bp --case 128=output/case128/tgv_00000621.bp

Writes one `spectrum_N<N>.dat` (k, E) per case and `spectrum.png`. Needs the optional `adios` package.
Use --legacy to reproduce the normalisation of the internship report (2 pi k^2 weighting).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tgvtools import io, spectra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", required=True, metavar="N=file.bp")
    ap.add_argument("--length", type=float, default=2 * np.pi)
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()

    fig, ax = plt.subplots()
    for spec in a.case:
        n, path = spec.split("=", 1)
        u, v, w = io.read_adios_velocity(path)
        if a.legacy:
            d = a.length / int(n)
            cnt, ks, es = spectra.energy_spectrum_legacy(u, v, w, d, d, d)
            k, e = ks[1:-2] / cnt[1:-2], es[1:-2] / cnt[1:-2]
        else:
            k, e = spectra.energy_spectrum(u, v, w, a.length)
        np.savetxt(f"spectrum_N{n}.dat", np.c_[k, e], header="k E(k)")
        ax.loglog(k, e, label=f"N={n}")
    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.legend()
    fig.savefig("spectrum.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
