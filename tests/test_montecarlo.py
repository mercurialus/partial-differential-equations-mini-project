import numpy as np

from bsheat.formula import bs_call
from bsheat.montecarlo import mc_call

K, r, sigma, T = 100.0, 0.05, 0.25, 1.0


def test_formula_inside_confidence_interval():
    # A 95% interval misses the true value in about 1 run out of 20, so a
    # fixed seed is used here; the coverage test below checks the 95%.
    for S0 in (80.0, 90.0, 100.0, 110.0, 120.0):
        est, lo, hi, se = mc_call(S0, K, r, sigma, T, n_paths=400_000, seed=0)
        assert lo <= bs_call(S0, K, r, sigma, T) <= hi
        assert se < 0.05


def test_confidence_interval_coverage():
    exact = bs_call(100.0, K, r, sigma, T)
    hits = [lo <= exact <= hi for lo, hi in
            (mc_call(100.0, K, r, sigma, T, n_paths=2_000, seed=s)[1:3] for s in range(400))]
    assert 0.92 <= np.mean(hits) <= 0.98
