import numpy as np

from src.household.cashflows import nominal_cashflows
from src.household.portfolio import kzt_wealth
from src.household.profiles import P1_LOCAL_CONTROL
from src.household.settlement import simulate_household


def test_rebalance_restores_target_weights():
    fx = np.ones((2, 3, 2))
    fx[:, :, 0] = 400.0
    fx[:, :, 1] = 500.0
    cf = nominal_cashflows(P1_LOCAL_CONTROL, 1.0, 0.8, 400.0, 500.0)
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    run = simulate_household(fx, cf, w, W0=2.4)
    a_k, a_u, a_e = run.holdings_end
    S_u, S_e = 400.0, 500.0
    W = kzt_wealth(np.stack([a_k, a_u, a_e], axis=-1), S_u, S_e)
    assert np.allclose(a_k, w[0] * W)
    assert np.allclose(a_u * S_u, w[1] * W)
    assert np.allclose(a_e * S_e, w[2] * W)
