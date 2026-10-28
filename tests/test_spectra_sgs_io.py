import numpy as np
import pytest

from tgvtools import io, sgs, spectra


def _loop_reference(U, V, W, dx, dy, dz):
    """Original triple-loop implementation (compute_spectrum_fanav_3D, fast=False)."""
    nx, ny, nz = U.shape
    ek = sum(np.fft.fftshift(abs(np.fft.fftn(c) / c.size) ** 2) for c in (U, V, W))
    ic, jc, kc = nx // 2, ny // 2, nz // 2
    nbox = int(np.ceil(np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) / 2.0) + 1)
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, dx))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, dy))
    kz = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(nz, dz))
    cnt, ks, es = np.zeros(nbox), np.zeros(nbox), np.zeros(nbox) + 1e-50
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                wn = int(np.round(np.sqrt((i - ic) ** 2 + (j - jc) ** 2 + (k - kc) ** 2)))
                kk = np.sqrt(kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2)
                cnt[wn] += 1
                ks[wn] += kk
                es[wn] += 2 * np.pi * kk ** 2 * ek[i, j, k]
    return cnt, ks, 0.5 * es


def test_legacy_vectorised_matches_loops():
    rng = np.random.default_rng(0)
    u, v, w = (rng.standard_normal((12, 12, 12)) for _ in range(3))
    dx = 2 * np.pi / 12
    for a, b in zip(spectra.energy_spectrum_legacy(u, v, w, dx, dx, dx), _loop_reference(u, v, w, dx, dx, dx)):
        assert np.allclose(a, b)


def test_parseval():
    rng = np.random.default_rng(1)
    n = 16
    u, v, w = (rng.standard_normal((n, n, n)) for _ in range(3))
    for c in (u, v, w):
        c -= c.mean()
    k, e = spectra.energy_spectrum(u, v, w)
    # shells above N/2 (corner modes) are not returned: compare with all shells instead
    full = sum(0.5 * np.abs(np.fft.fftn(c) / c.size) ** 2 for c in (u, v, w)).sum()
    assert full == pytest.approx(0.5 * np.mean(u ** 2 + v ** 2 + w ** 2))
    assert e.sum() <= full + 1e-12


def test_single_mode_lands_in_right_shell():
    n = 32
    x = 2 * np.pi * np.arange(n) / n
    X, _, _ = np.meshgrid(x, x, x, indexing="ij")
    u = np.cos(3 * X)
    zero = np.zeros_like(u)
    k, e = spectra.energy_spectrum(u, zero, zero)
    assert k[np.argmax(e)] == pytest.approx(3.0)
    assert e.sum() == pytest.approx(0.5 * np.mean(u ** 2))


def test_nu_sgs_ideal_equals_both_original_expressions():
    rng = np.random.default_rng(2)
    nu = 1 / 1600
    ef = rng.uniform(1e-3, 1e-2, 50)
    eps = ef * rng.uniform(1.0, 1.5, 50)
    original_a = (eps - ef) / (2 * (ef / (2 * nu)))  # nu_exact_N256fdns.py
    original_b = nu * ((eps / (2 * nu)) / (ef / (2 * nu)) - 1)  # plot-sgs-viscosity.py
    assert np.allclose(sgs.nu_sgs_ideal(eps, ef, nu), original_a)
    assert np.allclose(sgs.nu_sgs_ideal(eps, ef, nu), original_b)


def test_peak_error():
    t = np.linspace(0, 15, 301)
    ref = np.exp(-((t - 9) / 3) ** 2)
    assert sgs.relative_peak_error(t, 0.7 * ref, t, ref) == pytest.approx(-0.3)


def test_sort_and_comma(tmp_path):
    f = tmp_path / "raw"
    f.write_text("# c\n2.0 5.0\n1.0 3.0\n")
    out = io.sort_time_series(f)
    t, y = io.load_columns(out, (0, 1))
    assert list(t) == [1.0, 2.0] and list(y) == [3.0, 5.0]
    g = tmp_path / "vr"
    g.write_text("1,5 2,5\n")
    assert io.convert_decimal_comma(g).read_text() == "1.5 2.5\n"
