import numpy as np

from src.household.portfolio import initial_holdings, kzt_wealth


def test_initial_portfolio_values_to_W0():
    w = np.array([0.2, 0.5, 0.3])
    W0 = 4.0
    S_u, S_e = 450.0, 500.0
    h = initial_holdings(W0, w, S_u, S_e)
    recovered = kzt_wealth(h, S_u, S_e)
    assert np.isclose(recovered, W0)
    assert np.isclose(h[0], 0.2 * W0)
    assert np.isclose(h[1], 0.5 * W0 / S_u)
    assert np.isclose(h[2], 0.3 * W0 / S_e)
