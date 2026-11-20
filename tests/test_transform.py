import numpy as np

from bsheat.formula import bs_call
from bsheat.transform import constants, to_heat, from_heat, u_from_V, u_exact, u0

K, r, sigma, T = 100.0, 0.05, 0.25, 1.0
k, alpha, beta = constants(r, sigma)


def test_constants():
    assert np.isclose(k, 1.6)
    assert np.isclose(alpha, -0.3)
    assert np.isclose(beta, -1.69)


def test_roundtrip():
    S, t = np.array([50.0, 100.0, 150.0]), np.array([0.0, 0.3, 0.9])
    x, tau = to_heat(S, t, K, sigma, T)
    S2, t2 = from_heat(x, tau, K, sigma, T)
    assert np.allclose(S, S2) and np.allclose(t, t2)


def test_u_from_formula_equals_closed_form():
    x = np.linspace(-1, 1, 41)
    tau = 0.02
    S, t = from_heat(x, tau, K, sigma, T)
    u = u_from_V(bs_call(S, K, r, sigma, T - t), x, tau, K, r, sigma)
    assert np.max(np.abs(u - u_exact(x, tau, k))) < 1e-10


def test_u_satisfies_heat_equation():
    """Central-difference residual u_tau - u_xx of the transformed BS price."""
    def u_of(x, tau):
        S, t = from_heat(x, tau, K, sigma, T)
        return u_from_V(bs_call(S, K, r, sigma, T - t), x, tau, K, r, sigma)

    x = np.linspace(-0.8, 0.8, 33)
    h, d = 1e-3, 1e-4
    for tau in (0.005, 0.02, 0.03):
        u_t = (u_of(x, tau + d) - u_of(x, tau - d)) / (2 * d)
        u_xx = (u_of(x + h, tau) - 2 * u_of(x, tau) + u_of(x - h, tau)) / h**2
        rel = np.max(np.abs(u_t - u_xx)) / np.max(np.abs(u_xx))
        assert rel < 1e-4, (tau, rel)


def test_initial_data():
    x = np.array([-0.5, 0.0, 0.5])
    assert np.allclose(u0(x, k), [0.0, 0.0, np.exp(1.3 * 0.5) - np.exp(0.3 * 0.5)])
