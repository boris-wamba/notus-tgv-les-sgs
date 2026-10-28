"""Shell-averaged kinetic-energy spectra of 3D periodic velocity fields (vectorised)."""
from __future__ import annotations

import numpy as np


def _wavenumbers(n: int) -> np.ndarray:
    return np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers (cycles per box)


def energy_spectrum(u, v, w, length: float = 2 * np.pi):
    """E(k) such that sum_k E(k) = <(u^2 + v^2 + w^2)/2> (Parseval, checked in the tests).

    Shells are integer wavenumber bins |k| in [n - 1/2, n + 1/2). Returns (k, E) with
    k = 2 pi n / length for n = 1 ... N/2 (the mean mode n = 0 is dropped).
    """
    shape = u.shape
    if not (u.shape == v.shape == w.shape) or u.ndim != 3:
        raise ValueError("u, v, w must be 3D arrays of equal shape")
    e3 = np.zeros(shape)
    for comp in (u, v, w):
        e3 += 0.5 * np.abs(np.fft.fftn(comp) / comp.size) ** 2
    kx = _wavenumbers(shape[0])[:, None, None]
    ky = _wavenumbers(shape[1])[None, :, None]
    kz = _wavenumbers(shape[2])[None, None, :]
    shell = np.rint(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)).astype(int)
    e = np.bincount(shell.ravel(), weights=e3.ravel())
    nmax = min(shape) // 2
    n = np.arange(1, nmax + 1)
    return 2 * np.pi * n / length, e[1:nmax + 1]


def energy_spectrum_legacy(u, v, w, dx: float, dy: float, dz: float):
    """Vectorised version of the original ``compute_spectrum_fanav_3D`` (same output, no loops).

    Kept to reproduce the spectra of the internship report: energy is multiplied by 2 pi k^2
    and averaged over the number of points per shell. Returns (n_points, k_sum, E_sum);
    divide k_sum and E_sum by n_points to obtain the per-shell averages.
    """
    nx, ny, nz = u.shape
    ek = np.zeros(u.shape)
    for comp in (u, v, w):
        ek += np.fft.fftshift(np.abs(np.fft.fftn(comp) / comp.size) ** 2)
    ic, jc, kc = nx // 2, ny // 2, nz // 2
    nbox = int(np.ceil(np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) / 2.0) + 1)
    i = np.arange(nx)[:, None, None] - ic
    j = np.arange(ny)[None, :, None] - jc
    k = np.arange(nz)[None, None, :] - kc
    wn = np.rint(np.sqrt(i ** 2 + j ** 2 + k ** 2)).astype(int)
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, dx))[:, None, None]
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, dy))[None, :, None]
    kz = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nz, dz))[None, None, :]
    ksph = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    count = np.bincount(wn.ravel(), minlength=nbox).astype(float)
    k_sum = np.bincount(wn.ravel(), weights=ksph.ravel(), minlength=nbox)
    e_sum = 1e-50 + np.bincount(wn.ravel(), weights=(2 * np.pi * ksph ** 2 * ek).ravel(), minlength=nbox)
    return count, k_sum, 0.5 * e_sum


def reference_slope(k: np.ndarray, e: np.ndarray, exponent: float = -5.0 / 3.0) -> np.ndarray:
    """Power law e_max (k / k_max)^exponent anchored on the spectrum maximum above k index N/4."""
    start = len(e) // 4
    imax = start + int(np.argmax(e[start:]))
    return e[imax] * (k / k[imax]) ** exponent
