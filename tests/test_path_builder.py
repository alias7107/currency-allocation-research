"""FX path construction tests."""

from __future__ import annotations

import numpy as np

from src.simulation.path_builder import implied_log_returns, returns_to_levels, sanity_check_paths


def test_level_identity_and_positivity():
    r = np.array([[[0.01, -0.02], [0.0, 0.03], [-0.05, 0.01]]])
    s0 = np.array([450.0, 500.0])
    levels = returns_to_levels(r, s0)
    assert levels.shape == (1, 4, 2)
    assert np.allclose(levels[0, 0], s0)
    assert np.allclose(implied_log_returns(levels), r)
    chk = sanity_check_paths(levels, r)
    assert chk["all_positive"]
    assert chk["return_identity_ok"]
    assert not chk["has_nan"]


def test_no_clipping_of_large_moves():
    r = np.array([[[2.0, 2.0]]])  # huge log return
    levels = returns_to_levels(r, np.array([100.0, 100.0]))
    assert levels[0, 1, 0] == np.exp(2.0) * 100.0
