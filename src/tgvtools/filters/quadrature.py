"""1D quadrature weights of the top-hat filter of width ``width * dx``.

The top-hat (box) filter  ubar(x) = 1/(m dx) * int_{x - m dx/2}^{x + m dx/2} u(s) ds
is evaluated on the grid with a composite quadrature rule on ``m`` intervals:

* trapezoid : weights 1, 2, ..., 2, 1      normalised by 1/(2m)  (exact for linear u)
* simpson   : weights 1, 4, 2, ..., 4, 1   normalised by 1/(3m)  (exact for cubic u)

``m`` must be even so that the stencil is centred on a grid node (m = 2, 4, 8, ...).
"""
from __future__ import annotations

import numpy as np

KINDS = ("trapezoid", "simpson")


def _check(kind: str, width: int) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if width < 2 or width % 2:
        raise ValueError(f"width must be an even integer >= 2 (in units of dx), got {width}")


def integer_weights(kind: str, width: int) -> np.ndarray:
    """Un-normalised integer weights, length ``width + 1`` (offsets -width/2 ... +width/2)."""
    _check(kind, width)
    w = np.ones(width + 1, dtype=int)
    if kind == "trapezoid":
        w[1:-1] = 2
    else:  # simpson
        w[1:-1:2] = 4
        w[2:-1:2] = 2
    return w


def normalisation(kind: str, width: int) -> int:
    """Denominator D such that kernel = integer_weights / D (1/24 for Simpson, width 8)."""
    _check(kind, width)
    return (2 if kind == "trapezoid" else 3) * width


def offsets(width: int) -> np.ndarray:
    return np.arange(-(width // 2), width // 2 + 1)


def kernel(kind: str, width: int) -> np.ndarray:
    """Normalised 1D kernel (sums to 1)."""
    return integer_weights(kind, width) / normalisation(kind, width)
