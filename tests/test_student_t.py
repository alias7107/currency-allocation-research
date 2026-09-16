"""Student-t joint simulator tests."""

from __future__ import annotations

import numpy as np

from src.simulation.student_t import StudentTSimulator, fit_bivariate_student_t, mvt_logpdf


def test_mle_recovers_df_order_of_magnitude():
    rng = np.random.default_rng(0)
    df = 6.0
    mu = np.array([0.0, 0.0])
    scale = np.array([[1.0, 0.5], [0.5, 1.2]])
    n = 4000
    z = rng.standard_normal((n, 2))
    w = rng.chisquare(df, size=n)
    chol = np.linalg.cholesky(scale)
    x = mu + (z @ chol.T) / np.sqrt(w / df)[:, None]
    params = fit_bivariate_student_t(x)
    assert params.success or params.df > 2
    assert 3.0 < params.df < 15.0
    # scale correlation sign preserved
    sc = params.scale[0, 1] / np.sqrt(params.scale[0, 0] * params.scale[1, 1])
    assert sc > 0.2


def test_simulate_shape_and_finite():
    rng = np.random.default_rng(1)
    x = rng.standard_t(5, size=(300, 2))
    sim = StudentTSimulator().fit(x)
    out = sim.simulate(11, 7, random_seed=2)
    assert out.shape == (11, 7, 2)
    assert np.isfinite(out).all()


def test_logpdf_finite():
    x = np.zeros((5, 2))
    ll = mvt_logpdf(x, np.zeros(2), np.eye(2), 8.0)
    assert np.isfinite(ll).all()
