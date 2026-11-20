import numpy as np

from bsheat.fd import price_call_fd
from bsheat.formula import bs_call

K, r, sigma, T = 100.0, 0.05, 0.25, 1.0
S = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
exact = bs_call(S, K, r, sigma, T)


def err(dx, lam):
    V = price_call_fd(S, K, r, sigma, T, dx=dx, lam=lam, method="explicit")
    return np.max(np.abs(V - exact))


def test_explicit_converges_lambda_04():
    e = [err(dx, 0.4) for dx in (0.05, 0.025, 0.0125)]
    assert e[-1] < 5e-3
    assert e[0] > e[1] > e[2]


def test_explicit_diverges_lambda_06():
    e = err(0.0125, 0.6)
    assert not np.isfinite(e) or e > 1e3


def test_crank_nicolson_accurate():
    V = price_call_fd(S, K, r, sigma, T, dx=0.0125, method="cn", n_steps=400)
    assert np.max(np.abs(V - exact)) < 5e-3
