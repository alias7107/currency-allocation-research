import numpy as np

from src.household.cashflows import nominal_cashflows
from src.household.profiles import P1_LOCAL_CONTROL
from src.household.risk_metrics import cumulative_shortfall, empirical_cvar
from src.household.settlement import simulate_household


def test_structural_deficit_shortfall_after_buffer():
    fx = np.ones((1, 6, 2)) * np.array([400.0, 500.0])
    cf = nominal_cashflows(P1_LOCAL_CONTROL, 1.0, 1.05, 400.0, 500.0)
    # W0 = 1 * 1.05; monthly gap 0.05 → 21 months theoretically; 6 months still funded
    run = simulate_household(fx, cf, np.array([1.0, 0.0, 0.0]), W0=1.05)
    assert np.allclose(run.monthly_shortfall, 0.0)
    run2 = simulate_household(fx, cf, np.array([1.0, 0.0, 0.0]), W0=0.10)
    L = cumulative_shortfall(run2.monthly_shortfall)
    assert L[0] > 0


def test_cvar_largest_k():
    x = np.arange(100.0)
    # N=100, alpha=0.95, k=5, mean of 99..95
    assert np.isclose(empirical_cvar(x, 0.95), np.mean([99, 98, 97, 96, 95]))
