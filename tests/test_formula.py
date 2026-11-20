import numpy as np
import pytest

from bsheat.formula import bs_call, bs_put, parity_gap

K, r, sigma, T = 100.0, 0.05, 0.25, 1.0


def test_put_call_parity():
    S = np.linspace(10, 300, 50)
    for tau in (0.01, 0.5, 1.0, 3.0):
        assert np.max(np.abs(parity_gap(S, K, r, sigma, tau))) < 1e-10


def test_limit_small_S():
    assert bs_call(1e-6, K, r, sigma, T) < 1e-12
    assert bs_call(0.0, K, r, sigma, T) == 0.0


def test_limit_large_S():
    S = 1e4
    assert bs_call(S, K, r, sigma, T) == pytest.approx(S - K * np.exp(-r * T), rel=1e-12)


def test_limit_expiry():
    S = np.array([80.0, 99.0, 100.0, 101.0, 120.0])
    payoff = np.maximum(S - K, 0)
    assert np.max(np.abs(bs_call(S, K, r, sigma, 1e-10) - payoff)) < 1e-3
    assert np.array_equal(bs_call(S, K, r, sigma, 0.0), payoff)


def test_known_value():
    # textbook check value: S=K=100, r=0.05, sigma=0.25, T=1 -> 12.3360
    assert bs_call(100.0, K, r, sigma, T) == pytest.approx(12.3360, abs=1e-4)
    assert bs_put(100.0, K, r, sigma, T) == pytest.approx(7.4590, abs=1e-4)
