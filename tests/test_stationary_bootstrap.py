"""Stationary bootstrap tests."""

from __future__ import annotations

import numpy as np

from src.simulation.stationary_bootstrap import (
    StationaryBootstrapSimulator,
    geometric_success_probability,
    politis_white_mean_block_length,
)


def test_p_equals_one_over_L():
    assert geometric_success_probability(10) == 0.1


def test_wrap_and_joint_continuation():
    n = 12
    hist = np.column_stack([np.arange(n), np.arange(n) + 100])
    sim = StationaryBootstrapSimulator(expected_block_length=20).fit(hist)
    out = sim.simulate(30, 40, random_seed=3)
    assert out.shape == (30, 40, 2)
    # within a continued step, EUR = USD + 100
    assert np.allclose(out[:, :, 1], out[:, :, 0] + 100)
    # wrapping: values remain in the historical set
    assert set(out[:, :, 0].ravel().tolist()).issubset(set(range(n)))


def test_short_blocks_more_restarts_than_long_blocks():
    hist = np.column_stack([np.arange(200), np.arange(200)])
    s5 = StationaryBootstrapSimulator(5).fit(hist).simulate(5, 400, random_seed=1)
    s20 = StationaryBootstrapSimulator(20).fit(hist).simulate(5, 400, random_seed=1)
    def restarts(arr):
        d = np.diff(arr[0, :, 0])
        # non +1 (or wrap 199->0) counts as restart
        cont = (d == 1) | (d == -199)
        return int((~cont).sum())
    assert restarts(s5) > restarts(s20)


def test_politis_white_positive():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    l = politis_white_mean_block_length(np.abs(x))
    assert l >= 2
