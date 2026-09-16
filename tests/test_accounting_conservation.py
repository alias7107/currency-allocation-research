import numpy as np

from src.household.cashflows import nominal_cashflows
from src.household.profiles import P4_MULTI_CURRENCY
from src.household.settlement import simulate_household


def test_conservation_and_no_negative():
    rng = np.random.default_rng(0)
    fx = np.stack(
        [400.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(8, 24)), axis=1)),
         500.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(8, 24)), axis=1))],
        axis=-1,
    )
    cf = nominal_cashflows(P4_MULTI_CURRENCY, 1.0, 0.95, 400.0, 500.0)
    w = np.array([0.5, 0.3, 0.2])
    run = simulate_household(fx, cf, w, W0=2.85)
    assert run.max_conservation_error < 1e-8
    assert not run.negative_holdings
    assert not run.has_nan
    assert np.all(run.terminal_wealth >= -1e-12)
