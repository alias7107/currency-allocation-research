"""Reproducibility of the joint simulation interface."""

from __future__ import annotations

import numpy as np

from src.simulation.iid_bootstrap import IIDBootstrapSimulator
from src.simulation.stationary_bootstrap import StationaryBootstrapSimulator
from src.simulation.student_t import GaussianSimulator, StudentTSimulator


HIST = np.array(
    [
        [0.001, 0.002],
        [-0.004, -0.003],
        [0.010, 0.008],
        [-0.002, 0.001],
        [0.003, 0.000],
    ],
    dtype=float,
)


def test_all_models_same_seed_same_output():
    for cls, kwargs in (
        (IIDBootstrapSimulator, {}),
        (StationaryBootstrapSimulator, {"expected_block_length": 3}),
        (StudentTSimulator, {}),
        (GaussianSimulator, {}),
    ):
        a = cls(**kwargs).fit(HIST).simulate(6, 9, random_seed=42)
        b = cls(**kwargs).fit(HIST).simulate(6, 9, random_seed=42)
        assert np.allclose(a, b), cls.__name__
        assert a.shape == (6, 9, 2)


def test_no_nan_inf_in_bootstrap():
    out = StationaryBootstrapSimulator(5).fit(HIST).simulate(20, 30, random_seed=0)
    assert np.isfinite(out).all()
