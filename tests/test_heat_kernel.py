import numpy as np

from bsheat.formula import bs_call
from bsheat.heat_kernel import bs_call_heat_kernel, convolve, heat_kernel

K, r, sigma, T = 100.0, 0.05, 0.25, 1.0


def test_kernel_has_unit_mass():
    x = np.linspace(-8, 8, 32001)
    assert abs(np.trapezoid(heat_kernel(x, 0.3), x) - 1.0) < 1e-10


def test_convolution_of_gaussian():
    # Gaussian data e^{-x^2/(4 t0)} / sqrt(4 pi t0) evolves to the kernel at t0 + tau
    t0, tau = 0.1, 0.25
    f = lambda s: heat_kernel(s, t0)
    for x in (-1.0, 0.0, 0.7):
        assert abs(convolve(f, x, tau) - heat_kernel(x, t0 + tau)) < 1e-12


def test_quadrature_matches_formula():
    S = np.array([60.0, 80.0, 100.0, 120.0, 160.0])
    for tau in (0.05, 1.0, 2.0):
        diff = bs_call_heat_kernel(S, K, r, sigma, tau) - bs_call(S, K, r, sigma, tau)
        assert np.max(np.abs(diff)) < 1e-8
