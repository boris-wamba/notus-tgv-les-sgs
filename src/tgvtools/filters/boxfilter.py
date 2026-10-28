"""Separable top-hat filters on periodic Cartesian fields (1D, 2D, 3D).

Replaces the explicit double/triple Python loops: the filter is applied axis by axis with
``scipy.ndimage.convolve1d`` (periodic wrap-around, as in the Taylor-Green box).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.ndimage import convolve1d

from .quadrature import kernel


def filter_field(u: np.ndarray, kind: str = "simpson", width: int = 2,
                 axes: Optional[Iterable[int]] = None) -> np.ndarray:
    """Top-hat filter of width ``width * dx`` applied on ``axes`` (default: all), periodic."""
    k = kernel(kind, width)
    out = np.asarray(u, dtype=float)
    for ax in (range(out.ndim) if axes is None else axes):
        out = convolve1d(out, k, axis=ax, mode="wrap")
    return out


def transfer_function(kind: str, width: int, theta: np.ndarray) -> np.ndarray:
    """Transfer function H(theta), theta = k*dx in [0, pi]: H = sum_p w_p cos(p theta)."""
    k = kernel(kind, width)
    p = np.arange(-(width // 2), width // 2 + 1)
    theta = np.asarray(theta, dtype=float)
    return np.sum(k[None, :] * np.cos(np.outer(theta.ravel(), p)), axis=1).reshape(theta.shape)


def three_point_kernel(coeff: float = 4.0) -> np.ndarray:
    """Kernel [1, c, 1] / (2 + c) of the small test filter used in the Notus filter test.

    coeff = 2 gives the trapezoid rule of width 2dx, coeff = 4 the Simpson rule of width 2dx.
    """
    return np.array([1.0, coeff, 1.0]) / (2.0 + coeff)


def highpass(u: np.ndarray, coeff: float = 4.0, axes: Optional[Iterable[int]] = None) -> np.ndarray:
    """u - filtered(u) with the separable [1, c, 1] kernel (vectorised, periodic).

    In 2D this is the stencil of ``filtre_notus_2D``: diagonals 1, direct neighbours c, centre c^2,
    normalised by (2 + c)^2, subtracted from u.
    """
    k = three_point_kernel(coeff)
    smooth = np.asarray(u, dtype=float)
    for ax in (range(smooth.ndim) if axes is None else axes):
        smooth = convolve1d(smooth, k, axis=ax, mode="wrap")
    return np.asarray(u, dtype=float) - smooth
