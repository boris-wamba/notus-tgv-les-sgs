import numpy as np
import pytest
import sympy as sp

from tgvtools.filters import filter_field, highpass, kernel, three_point_kernel
from tgvtools.filters.quadrature import integer_weights, normalisation
from tgvtools.filters.symbolic import stencil


@pytest.mark.parametrize("kind", ["trapezoid", "simpson"])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_kernel_normalised_and_symmetric(kind, width):
    k = kernel(kind, width)
    assert k.sum() == pytest.approx(1.0)
    assert np.allclose(k, k[::-1])


def test_simpson_width8_matches_original_scripts():
    assert list(integer_weights("simpson", 8)) == [1, 4, 2, 4, 2, 4, 2, 4, 1]
    assert normalisation("simpson", 8) == 24


@pytest.mark.parametrize("width", [2, 4, 8])
def test_simpson_exact_for_cubics(width):
    dx, n = 0.1, 200
    x = np.arange(n) * dx
    h = width * dx / 2
    got = np.convolve(x ** 3, kernel("simpson", width), mode="same")
    exact = x ** 3 + x * h ** 2  # mean of (x+s)^3 over [-h, h]
    m = width // 2
    assert np.allclose(got[m:-m], exact[m:-m])


def test_trapezoid_exact_for_linear():
    x = np.arange(100) * 0.1
    got = np.convolve(3 * x + 1, kernel("trapezoid", 4), mode="same")
    assert np.allclose(got[2:-2], (3 * x + 1)[2:-2])


@pytest.mark.parametrize("kind,expected", [("trapezoid", 2.0), ("simpson", 4.0)])
def test_convergence_order(kind, expected):
    """Fixed physical width h = pi/8, grids 64/128/256: error ratio ~ 2^p, p = 2 (trap.) or 4 (Simpson)."""
    errs = []
    for n, m in ((64, 8), (128, 16), (256, 32)):
        x = 2 * np.pi * np.arange(n) / n
        h = np.pi / 8
        exact = np.sin(x) * np.sin(h) / h
        errs.append(np.max(np.abs(filter_field(np.sin(x), kind, m) - exact)))
    order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    assert all(abs(o - expected) < 0.3 for o in order)


def test_3d_filter_is_tensor_product():
    rng = np.random.default_rng(1)
    u = rng.standard_normal((16, 16, 16))
    full = filter_field(u, "simpson", 4)
    seq = u
    for ax in range(3):
        seq = filter_field(seq, "simpson", 4, axes=[ax])
    assert np.allclose(full, seq)


def test_filter_preserves_mean():
    u = np.random.default_rng(2).standard_normal((12, 12))
    assert filter_field(u, "simpson", 8).mean() == pytest.approx(u.mean())


@pytest.mark.parametrize("ndim,factor", [(1, sp.Rational(1, 24)), (2, sp.Rational(1, 576)), (3, sp.Rational(1, 13824))])
def test_symbolic_prefactors_match_original_scripts(ndim, factor):
    assert stencil("simpson", 8, ndim)[0] == factor


def test_symbolic_stencil_matches_numeric_kernel():
    factor, expr = stencil("simpson", 4, 2)
    i, j = sp.symbols("i j", integer=True)
    u = sp.IndexedBase("u")
    poly = sp.Poly(sp.expand(factor * expr), *[u[i + p, j + q] for p in range(-2, 3) for q in range(-2, 3)])
    k = kernel("simpson", 4)
    for p in range(-2, 3):
        for q in range(-2, 3):
            assert float(poly.coeff_monomial(u[i + p, j + q])) == pytest.approx(k[p + 2] * k[q + 2])


def test_notus_highpass_is_u_minus_simpson2():
    u = np.random.default_rng(3).standard_normal((20, 20))
    assert np.allclose(highpass(u, 4.0), u - filter_field(u, "simpson", 2))
    assert np.allclose(three_point_kernel(2.0), kernel("trapezoid", 2))
