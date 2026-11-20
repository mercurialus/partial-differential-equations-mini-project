"""Monte Carlo pricing under geometric Brownian motion (risk-neutral):
    S_T = S_0 exp((r - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),
    price = e^{-rT} E[max(S_T - K, 0)].
The 95% confidence interval is mean +- 1.96 * (sample std) / sqrt(n).
"""
import numpy as np


def mc_call(S0, K, r, sigma, tau, n_paths=1_000_000, seed=0):
    """Return (estimate, ci_low, ci_high, std_error)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * tau + sigma * np.sqrt(tau) * Z)
    disc = np.exp(-r * tau) * np.maximum(ST - K, 0.0)
    mean = disc.mean()
    se = disc.std(ddof=1) / np.sqrt(n_paths)
    return mean, mean - 1.96 * se, mean + 1.96 * se, se
