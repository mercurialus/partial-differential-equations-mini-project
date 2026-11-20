"""Double-barrier knock-out call (B1 < K < B2) by separation of variables.

In heat coordinates the option lives on a < x < b with
    a = ln(B1/K), b = ln(B2/K), l = b - a,  u(a, tau) = u(b, tau) = 0,
and u(x, 0) = u0(x) (the call initial data restricted to [a, b]).
Separation u = X(x) T(tau) gives X'' + lambda X = 0, X(a) = X(b) = 0, so
X_n = sin(n pi (x - a) / l), lambda_n = (n pi / l)^2, and

    u(x, tau) = sum_n c_n sin(n pi (x - a)/l) exp(-(n pi / l)^2 tau),
    c_n = (2/l) int_a^b u0(x) sin(n pi (x - a)/l) dx.

u0 = 0 for x < 0, so the integral runs over [max(a, 0), b].
"""
import numpy as np
from scipy.integrate import quad

from .transform import constants, u0, V_from_u
from .fd import crank_nicolson_heat


def barrier_interval(K, B1, B2):
    a, b = np.log(B1 / K), np.log(B2 / K)
    return a, b, b - a


def sine_coefficients(a, b, k, n_terms):
    """c_1..c_n by adaptive quadrature. With y = x - a the integrand is
    u0(y + a) sin(w y), w = n pi / l, which quad treats with its
    dedicated oscillatory rule (weight='sin')."""
    l = b - a
    y0 = max(a, 0.0) - a
    c = np.empty(n_terms)
    for n in range(1, n_terms + 1):
        w = n * np.pi / l
        val, _ = quad(lambda y: float(u0(y + a, k)), y0, l, weight="sin", wvar=w,
                      epsabs=1e-14, limit=200)
        c[n - 1] = 2.0 / l * val
    return c


def series_u(x, tau, a, b, coeffs):
    """Partial sum of the sine series at points x (1-D array), time tau."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    l = b - a
    n = np.arange(1, len(coeffs) + 1)
    modes = np.sin(np.outer(x - a, n) * np.pi / l)          # (len(x), N)
    return modes @ (coeffs * np.exp(-(n * np.pi / l) ** 2 * tau))


def series_partial_sums(x, tau, a, b, coeffs):
    """All partial sums at once: returns array (N, len(x)) whose row N-1 is
    the N-term approximation."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    l = b - a
    n = np.arange(1, len(coeffs) + 1)
    terms = np.sin(np.outer(n, x - a) * np.pi / l) * (coeffs * np.exp(-(n * np.pi / l) ** 2 * tau))[:, None]
    return np.cumsum(terms, axis=0)


def barrier_call_series(S, K, r, sigma, tau_years, B1, B2, n_terms=200, coeffs=None):
    """Price of the double-barrier knock-out call from the Fourier sine series."""
    k, _, _ = constants(r, sigma)
    a, b, _ = barrier_interval(K, B1, B2)
    if coeffs is None:
        coeffs = sine_coefficients(a, b, k, n_terms)
    x = np.log(np.atleast_1d(np.asarray(S, dtype=float)) / K)
    tau = 0.5 * sigma**2 * tau_years
    return V_from_u(series_u(x, tau, a, b, coeffs), x, tau, K, r, sigma)


def barrier_call_cn(K, r, sigma, tau_years, B1, B2, M=2000, n_steps=2000):
    """Crank-Nicolson solution of the same barrier problem on a uniform grid
    of [a, b]. Returns (S_grid, V_grid)."""
    k, _, _ = constants(r, sigma)
    a, b, _ = barrier_interval(K, B1, B2)
    x = np.linspace(a, b, M + 1)
    init = u0(x, k)
    init[0] = init[-1] = 0.0                  # knocked out at the barriers
    zero = lambda s: 0.0
    tau = 0.5 * sigma**2 * tau_years
    u = crank_nicolson_heat(init, x[1] - x[0], tau, n_steps, zero, zero, n_implicit=8)
    return K * np.exp(x), V_from_u(u, x, tau, K, r, sigma)


def terms_needed(K, r, sigma, tau_years, B1, B2, tol=1e-4, n_ref=1000, n_points=201,
                 coeffs=None):
    """Smallest N such that the N-term series is within tol (in price units)
    of the n_ref-term reference at every point of a grid in (B1, B2)."""
    k, _, _ = constants(r, sigma)
    a, b, _ = barrier_interval(K, B1, B2)
    if coeffs is None:
        coeffs = sine_coefficients(a, b, k, n_ref)
    x = np.linspace(a, b, n_points)[1:-1]
    tau = 0.5 * sigma**2 * tau_years
    scale = K * np.exp(-0.5 * (k - 1) * x - 0.25 * (k + 1) ** 2 * tau)  # V = scale * u
    partial = series_partial_sums(x, tau, a, b, coeffs) * scale
    err = np.max(np.abs(partial - partial[-1]), axis=1)
    # first N from which the error stays below tol
    bad = np.nonzero(err >= tol)[0]
    return int(bad[-1] + 2) if bad.size else 1
