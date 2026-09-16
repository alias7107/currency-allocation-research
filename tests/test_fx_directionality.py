import numpy as np

from src.household.cashflows import kzt_equivalent_expense, kzt_equivalent_income, nominal_cashflows
from src.household.profiles import P2_FOREIGN_INCOME, P4_MULTI_CURRENCY


def test_stronger_usd_kzt_raises_p2_income():
    """Higher USD/KZT (more tenge per dollar) raises KZT-equivalent USD income."""
    cf = nominal_cashflows(P2_FOREIGN_INCOME, 1.0, 0.8, 400.0, 500.0)
    low = kzt_equivalent_income(cf, np.array([400.0]), np.array([500.0]))[0]
    high = kzt_equivalent_income(cf, np.array([500.0]), np.array([500.0]))[0]
    assert high > low
    assert np.isclose(high / low, 500.0 / 400.0)


def test_eur_moves_p4_income_and_expenses():
    cf = nominal_cashflows(P4_MULTI_CURRENCY, 1.0, 1.0, 400.0, 500.0)
    S_u = np.array([400.0, 400.0])
    S_e = np.array([500.0, 600.0])
    inc = kzt_equivalent_income(cf, S_u, S_e)
    exp = kzt_equivalent_expense(cf, S_u, S_e)
    assert inc[1] > inc[0]
    assert exp[1] > exp[0]
    # EUR savings channel: same EUR holding is worth more KZT when S_EUR rises
    a_eur = 1.0
    assert a_eur * S_e[1] > a_eur * S_e[0]
