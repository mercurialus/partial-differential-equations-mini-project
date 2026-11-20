"""Closed-form Black-Scholes prices for European calls and puts.

Notation: S spot, K strike, r risk-free rate, sigma volatility,
tau = T - t time to expiry in years.

    C = S N(d1) - K e^{-r tau} N(d2),
    P = K e^{-r tau} N(-d2) - S N(-d1),
    d1,2 = [ln(S/K) + (r +- sigma^2/2) tau] / (sigma sqrt(tau)).
"""
import numpy as np
from scipy.stats import norm


def norm_cdf(z):
    """Standard normal CDF N(z) (thin wrapper so the source of N is explicit)."""
    return norm.cdf(z)


def d1_d2(S, K, r, sigma, tau):
    """Return (d1, d2). Requires S > 0 and tau > 0."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / sq
    return d1, d1 - sq


def _broadcast(S, tau):
    """Writable float arrays of a common shape (at least 1-D)."""
    S_b, tau_b = np.broadcast_arrays(np.atleast_1d(np.asarray(S, dtype=float)),
                                     np.atleast_1d(np.asarray(tau, dtype=float)))
    return S_b.copy(), tau_b.copy()


def _finish(out, S, tau):
    """Return a Python float for scalar inputs, an array otherwise."""
    return float(out[0]) if np.ndim(S) == 0 and np.ndim(tau) == 0 else out


def bs_call(S, K, r, sigma, tau):
    """European call price. Handles the edge cases S = 0 (price 0) and
    tau = 0 (price = payoff max(S - K, 0)) explicitly."""
    S_b, tau_b = _broadcast(S, tau)
    out = np.maximum(S_b - K, 0.0)            # value at expiry
    inside = (S_b > 0) & (tau_b > 0)
    if np.any(inside):
        s, t = S_b[inside], tau_b[inside]
        d1, d2 = d1_d2(s, K, r, sigma, t)
        out[inside] = s * norm_cdf(d1) - K * np.exp(-r * t) * norm_cdf(d2)
    out[(S_b <= 0)] = 0.0
    return _finish(out, S, tau)


def bs_put(S, K, r, sigma, tau):
    """European put price (same edge-case handling as bs_call)."""
    S_b, tau_b = _broadcast(S, tau)
    out = np.maximum(K - S_b, 0.0)
    inside = (S_b > 0) & (tau_b > 0)
    if np.any(inside):
        s, t = S_b[inside], tau_b[inside]
        d1, d2 = d1_d2(s, K, r, sigma, t)
        out[inside] = K * np.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)
    zero_S = (S_b <= 0) & (tau_b > 0)
    out[zero_S] = K * np.exp(-r * tau_b[zero_S])
    return _finish(out, S, tau)


def parity_gap(S, K, r, sigma, tau):
    """Put-call parity residual C - P - (S - K e^{-r tau}); zero in exact arithmetic."""
    return (bs_call(S, K, r, sigma, tau) - bs_put(S, K, r, sigma, tau)
            - (np.asarray(S, dtype=float) - K * np.exp(-r * np.asarray(tau, dtype=float))))
