"""Change of variables taking Black-Scholes to the heat equation.

Step 1 (dimensionless variables):
    S = K e^x,  t = T - 2 tau / sigma^2,  V(S, t) = K v(x, tau)
    =>  v_tau = v_xx + (k - 1) v_x - k v,     k = 2 r / sigma^2.
Step 2 (remove first- and zeroth-order terms):
    v = e^{alpha x + beta tau} u,  alpha = -(k - 1)/2,  beta = -(k + 1)^2 / 4
    =>  u_tau = u_xx,
    u(x, 0) = max(e^{(k+1)x/2} - e^{(k-1)x/2}, 0).
"""
import numpy as np
from scipy.stats import norm


def constants(r, sigma):
    """Return (k, alpha, beta)."""
    k = 2.0 * r / sigma**2
    alpha = -0.5 * (k - 1.0)
    beta = -0.25 * (k + 1.0) ** 2
    return k, alpha, beta


def to_heat(S, t, K, sigma, T):
    """(S, t) -> (x, tau) with x = ln(S/K), tau = sigma^2 (T - t) / 2."""
    return np.log(np.asarray(S, dtype=float) / K), 0.5 * sigma**2 * (T - np.asarray(t, dtype=float))


def from_heat(x, tau, K, sigma, T):
    """(x, tau) -> (S, t), inverse of to_heat."""
    return K * np.exp(np.asarray(x, dtype=float)), T - 2.0 * np.asarray(tau, dtype=float) / sigma**2


def u0(x, k):
    """Heat-equation initial data of the call: max(e^{(k+1)x/2} - e^{(k-1)x/2}, 0)."""
    x = np.asarray(x, dtype=float)
    return np.maximum(np.exp(0.5 * (k + 1) * x) - np.exp(0.5 * (k - 1) * x), 0.0)


def V_from_u(u, x, tau, K, r, sigma):
    """V = K e^{alpha x + beta tau} u."""
    _, alpha, beta = constants(r, sigma)
    return K * np.exp(alpha * np.asarray(x) + beta * np.asarray(tau)) * u


def u_from_V(V, x, tau, K, r, sigma):
    """u = (V / K) e^{-alpha x - beta tau}."""
    _, alpha, beta = constants(r, sigma)
    return np.asarray(V) / K * np.exp(-alpha * np.asarray(x) - beta * np.asarray(tau))


def u_exact(x, tau, k):
    """Closed-form solution of u_tau = u_xx with the call initial data (tau > 0):
        u = e^{(k+1)x/2 + (k+1)^2 tau/4} N(d1) - e^{(k-1)x/2 + (k-1)^2 tau/4} N(d2),
        d1 = x / sqrt(2 tau) + (k+1) sqrt(2 tau) / 2,
        d2 = x / sqrt(2 tau) + (k-1) sqrt(2 tau) / 2.
    """
    x = np.asarray(x, dtype=float)
    tau = np.asarray(tau, dtype=float)
    s = np.sqrt(2.0 * tau)
    d1 = x / s + 0.5 * (k + 1) * s
    d2 = x / s + 0.5 * (k - 1) * s
    return (np.exp(0.5 * (k + 1) * x + 0.25 * (k + 1) ** 2 * tau) * norm.cdf(d1)
            - np.exp(0.5 * (k - 1) * x + 0.25 * (k - 1) ** 2 * tau) * norm.cdf(d2))


def u_right_bc(x, tau, k):
    """Large-x boundary value of u from V ~ S - K e^{-r(T-t)}, i.e.
    v ~ e^x - e^{-k tau}, u = v e^{-alpha x - beta tau}."""
    alpha = -0.5 * (k - 1.0)
    beta = -0.25 * (k + 1.0) ** 2
    return (np.exp(x) - np.exp(-k * tau)) * np.exp(-alpha * x - beta * tau)
