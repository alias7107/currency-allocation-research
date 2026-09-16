"""IID joint bootstrap tests."""

from __future__ import annotations

import numpy as np

from src.simulation.iid_bootstrap import IIDBootstrapSimulator


def _toy():
    return np.array([[0.01, 0.02], [-0.03, -0.01], [0.00, 0.04], [0.05, 0.06]])


def test_joint_rows_are_resampled_together():
    hist = _toy()
    sim = IIDBootstrapSimulator().fit(hist)
    out = sim.simulate(2000, 5, random_seed=1)
    # every simulated pair must equal some historical row
    for p in range(50):
        for t in range(5):
            assert any(np.allclose(out[p, t], row) for row in hist)


def test_shape_and_no_independent_scramble():
    hist = _toy()
    sim = IIDBootstrapSimulator().fit(hist)
    out = sim.simulate(100, 10, random_seed=0)
    assert out.shape == (100, 10, 2)
    # USD 0.01 is only paired with EUR 0.02 in history
    mask = np.isclose(out[:, :, 0], 0.01)
    assert np.allclose(out[:, :, 1][mask], 0.02)


def test_reproducible_seed():
    hist = _toy()
    a = IIDBootstrapSimulator().fit(hist).simulate(20, 8, random_seed=7)
    b = IIDBootstrapSimulator().fit(hist).simulate(20, 8, random_seed=7)
    c = IIDBootstrapSimulator().fit(hist).simulate(20, 8, random_seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
