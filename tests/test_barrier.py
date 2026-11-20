import numpy as np

from bsheat.barrier import barrier_call_cn, barrier_call_series, terms_needed

K, r, sigma, B1, B2 = 100.0, 0.05, 0.25, 80.0, 130.0


def test_series_matches_crank_nicolson():
    for tau in (0.1, 0.5, 1.0):
        S_grid, V_cn = barrier_call_cn(K, r, sigma, tau, B1, B2, M=2000, n_steps=2000)
        V_ser = barrier_call_series(S_grid, K, r, sigma, tau, B1, B2, n_terms=400)
        assert np.max(np.abs(V_ser - V_cn)) < 1e-3, tau


def test_zero_at_barriers_and_positive_inside():
    V = barrier_call_series([B1, 100.0, 115.0, B2], K, r, sigma, 0.5, B1, B2)
    assert abs(V[0]) < 1e-12 and abs(V[-1]) < 1e-12
    assert V[1] > 0 and V[2] > 0


def test_more_terms_near_expiry():
    near = terms_needed(K, r, sigma, 0.01, B1, B2)
    far = terms_needed(K, r, sigma, 1.0, B1, B2)
    assert near > far
