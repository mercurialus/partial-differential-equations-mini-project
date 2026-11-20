"""Solution of the heat equation on the whole line by convolution with the
fundamental solution

    Phi(x, tau) = (4 pi tau)^{-1/2} exp(-x^2 / (4 tau)),
    u(x, tau)   = integral of u0(s) Phi(x - s, tau) ds.

Substituting s = x + 2 sqrt(tau) z turns the integral into
    u(x, tau) = pi^{-1/2} integral of u0(x + 2 sqrt(tau) z) e^{-z^2} dz,
which scipy.integrate.quad handles well on a semi-infinite interval.
"""
import numpy as np
from scipy.integrate import quad

from .transform import constants, u0, V_from_u


def heat_kernel(x, tau):
    """Fundamental solution Phi(x, tau) of u_tau = u_xx."""
    return np.exp(-np.asarray(x) ** 2 / (4.0 * tau)) / np.sqrt(4.0 * np.pi * tau)


def convolve(f, x, tau, support_left=-np.inf):
    """u(x, tau) = int f(s) Phi(x - s, tau) ds for scalar x, tau > 0.

    support_left: f vanishes for s < support_left (used to skip the zero part
    of the call payoff, whose support is s >= 0).
    """
    h = 2.0 * np.sqrt(tau)
    # e^{-z^2} < 1e-300 for |z| > 27, so a finite window is exact in double
    # precision and avoids inf * 0 when f grows exponentially.
    Z = 27.0
    z_left = max((support_left - x) / h, -Z) if np.isfinite(support_left) else -Z
    if z_left >= Z:
        return 0.0
    val, _ = quad(lambda z: f(x + h * z) * np.exp(-z * z), z_left, Z,
                  epsabs=1e-13, epsrel=1e-12, limit=400)
    return val / np.sqrt(np.pi)


def bs_call_heat_kernel(S, K, r, sigma, tau_years):
    """Call price obtained by numerically convolving u0 with the heat kernel
    and transforming back. tau_years = T - t > 0."""
    k, _, _ = constants(r, sigma)
    S_arr = np.atleast_1d(np.asarray(S, dtype=float))
    tau = 0.5 * sigma**2 * tau_years
    out = np.empty_like(S_arr)
    for i, s in enumerate(S_arr):
        x = np.log(s / K)
        u = convolve(lambda y: float(u0(y, k)), x, tau, support_left=0.0)
        out[i] = V_from_u(u, x, tau, K, r, sigma)
    return out if np.ndim(S) else float(out[0])
