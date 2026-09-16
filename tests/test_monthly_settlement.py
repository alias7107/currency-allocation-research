import numpy as np

from src.household.cashflows import daily_paths_to_monthly_rates, nominal_cashflows
from src.household.profiles import P1_LOCAL_CONTROL
from src.household.settlement import simulate_household


def test_monthly_rate_shape():
    daily = np.ones((3, 253, 2))
    daily[:, :, 0] = np.linspace(1, 2, 253)
    m = daily_paths_to_monthly_rates(daily, 21)
    assert m.shape == (3, 12, 2)
    assert np.allclose(m[0, 0], daily[0, 21])
    assert np.allclose(m[0, -1], daily[0, 252])


def test_p1_surplus_never_shortfalls():
    fx = np.ones((5, 12, 2)) * np.array([400.0, 500.0])
    fx[:, :, 0] *= np.linspace(1, 1.5, 12)
    cf = nominal_cashflows(P1_LOCAL_CONTROL, 1.0, 0.80, 400.0, 500.0)
    run = simulate_household(fx, cf, np.array([1.0, 0.0, 0.0]), W0=0.80)
    assert np.allclose(run.monthly_shortfall, 0)
    # surplus 0.20 per month * 12 + W0
    assert np.allclose(run.terminal_wealth, 0.80 + 12 * 0.20)
