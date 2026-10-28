"""Symbolic stencils of the top-hat filter (replaces the nine simpson_semi_composite_*.py scripts).

    python -m tgvtools.filters.symbolic --kind simpson --width 8 --ndim 3
    python -m tgvtools.filters.symbolic --kind trapezoid --width 4 --ndim 2 --latex

The multi-dimensional stencil is the tensor product of the 1D rule, so its prefactor is
(1/D)^ndim with D = 24 for Simpson/8dx, giving 1/576 in 2D and 1/13824 in 3D.
"""
from __future__ import annotations

import argparse
import itertools

import sympy as sp

from .quadrature import integer_weights, normalisation, offsets


def stencil(kind: str, width: int, ndim: int):
    """Return (prefactor, expression) with expression = sum of w_p w_q ... * u[i+p, j+q, ...]."""
    if ndim not in (1, 2, 3):
        raise ValueError("ndim must be 1, 2 or 3")
    w = dict(zip(offsets(width).tolist(), integer_weights(kind, width).tolist()))
    idx = sp.symbols("i j k", integer=True)[:ndim]
    u = sp.IndexedBase("u")
    expr = 0
    for combo in itertools.product(w, repeat=ndim):
        coeff = 1
        for c in combo:
            coeff *= w[c]
        expr += coeff * u[tuple(i + c for i, c in zip(idx, combo))]
    return sp.Rational(1, normalisation(kind, width) ** ndim), expr


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("trapezoid", "simpson"), default="simpson")
    ap.add_argument("--width", type=int, default=8, help="filter width in units of dx (even)")
    ap.add_argument("--ndim", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args(argv)
    factor, expr = stencil(a.kind, a.width, a.ndim)
    print(f"{a.kind}, width {a.width} dx, {a.ndim}D: prefactor = {factor}")
    print(sp.latex(factor * expr) if a.latex else sp.pretty(factor * expr, use_unicode=True))


if __name__ == "__main__":
    main()
