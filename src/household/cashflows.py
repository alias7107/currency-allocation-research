"""Normalize nominal FX cash flows and convert daily FX paths to monthly rates.

Monthly conversion
------------------
The FX engine simulates weekdays. The household settles once per month.
**21 simulated weekdays = 1 household month** (no calendar dates are assigned).

252 days → 12 months, 756 → 36, 1260 → 60.

For month m = 1..M the settlement FX rate is the **level at the end of the
m-th 21-day block** (index 21*m on a path that includes t=0 at index 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config
from src.household.profiles import HouseholdProfile


@dataclass(frozen=True)
class NominalCashFlows:
    I_kzt: float
    I_usd: float
    I_eur: float
    E_kzt: float
    E_usd: float
    E_eur: float
    I0: float
    E0: float
    S_usd0: float
    S_eur0: float


def nominal_cashflows(
    profile: HouseholdProfile,
    I0: float,
    E0: float,
    S_usd0: float,
    S_eur0: float,
) -> NominalCashFlows:
    """Foreign amounts are KZT-equivalent shares divided by initial FX levels."""
    if S_usd0 <= 0 or S_eur0 <= 0:
        raise ValueError("initial FX levels must be positive")
    return NominalCashFlows(
        I_kzt=profile.qI_kzt * I0,
        I_usd=profile.qI_usd * I0 / S_usd0,
        I_eur=profile.qI_eur * I0 / S_eur0,
        E_kzt=profile.qE_kzt * E0,
        E_usd=profile.qE_usd * E0 / S_usd0,
        E_eur=profile.qE_eur * E0 / S_eur0,
        I0=float(I0),
        E0=float(E0),
        S_usd0=float(S_usd0),
        S_eur0=float(S_eur0),
    )


def kzt_equivalent_income(cf: NominalCashFlows, S_usd: np.ndarray, S_eur: np.ndarray) -> np.ndarray:
    return cf.I_kzt + cf.I_usd * S_usd + cf.I_eur * S_eur


def kzt_equivalent_expense(cf: NominalCashFlows, S_usd: np.ndarray, S_eur: np.ndarray) -> np.ndarray:
    return cf.E_kzt + cf.E_usd * S_usd + cf.E_eur * S_eur


def daily_paths_to_monthly_rates(
    daily_levels: np.ndarray,
    weekdays_per_month: int | None = None,
) -> np.ndarray:
    """Map daily FX levels (n_paths, n_days+1, 2) to month-end rates (n_paths, n_months, 2).

    ``daily_levels[:, 0]`` is S0. Month m uses ``daily_levels[:, (m+1)*21]``.
    """
    w = weekdays_per_month or config.HH_WEEKDAYS_PER_MONTH
    levels = np.asarray(daily_levels, dtype=float)
    if levels.ndim != 3 or levels.shape[2] != 2:
        raise ValueError("daily_levels must have shape (n_paths, n_days+1, 2)")
    n_steps = levels.shape[1] - 1
    if n_steps % w != 0:
        raise ValueError(f"horizon days {n_steps} is not a multiple of {w}")
    n_months = n_steps // w
    idx = np.arange(1, n_months + 1) * w
    return levels[:, idx, :]
