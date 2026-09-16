import numpy as np

from src.household.cashflows import kzt_equivalent_income, nominal_cashflows
from src.household.profiles import P2_FOREIGN_INCOME, P4_MULTI_CURRENCY


def test_initial_kzt_equivalent_matches_I0_E0():
    S_u, S_e = 500.0, 550.0
    I0, E0 = 1.0, 0.95
    cf = nominal_cashflows(P4_MULTI_CURRENCY, I0, E0, S_u, S_e)
    inc0 = kzt_equivalent_income(cf, np.array([S_u]), np.array([S_e]))[0]
    exp0 = cf.E_kzt + cf.E_usd * S_u + cf.E_eur * S_e
    assert np.isclose(inc0, I0)
    assert np.isclose(exp0, E0)
    assert np.isclose(cf.I_usd, 0.5 * I0 / S_u)
    assert np.isclose(cf.I_eur, 0.5 * I0 / S_e)


def test_p2_usd_income_nominal():
    cf = nominal_cashflows(P2_FOREIGN_INCOME, 1.0, 0.8, 400.0, 500.0)
    assert cf.I_kzt == 0
    assert np.isclose(cf.I_usd, 1.0 / 400.0)
    assert cf.E_kzt == 0.8
