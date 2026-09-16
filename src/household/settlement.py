"""Monthly cash-flow settlement, shortfall, rebalancing, and conversion costs.

Event order (each month m, each path)
-------------------------------------
1. Observe month-end FX (S_USD, S_EUR).
2. Mark savings to KZT.
3. Receive nominal income in original currencies.
4. Pay expenses in original currencies (direct; no conversion).
5. Fund remaining gaps by **pro-rata liquidation** of leftover holdings
   by current KZT-equivalent value.
6. If wealth is insufficient, record KZT-equivalent shortfall and
   set holdings to zero. No borrowing, no negative holdings.
7. Optionally rebalance remaining wealth to target weights.

Conversion cost (Phase 2C)
--------------------------
``conversion_cost`` is a **scenario-based proportional FX conversion cost**
on the KZT-equivalent value of chargeable transactions — not an empirical
Kazakhstan retail spread estimate.

Chargeable events:
- Gap funding via liquidation after native currency matching:
  fee = c * take, with take grossed up so take + fee <= W
  (after native pay, remaining gaps are currency-mismatched by construction,
  so the full KZT-equivalent take is treated as one conversion through the
  KZT numeraire — no double charge for synthetic USD→KZT→EUR).
- Voluntary target-weight rebalancing: fee = c * one-way buy volume in KZT.

Rebalancing frequency
---------------------
- ``monthly``: every household month (baseline).
- ``quarterly``: after months 3, 6, 9, ... (1-based month index divisible by 3).
- ``none``: initial t=0 allocation only; thereafter holdings evolve via FX,
  native income/expenses, and necessary liquidation. New surplus remains in
  the currencies in which income arrived (no voluntary target restoration).

Conservation (after cash-flow / liquidation, before optional rebalance):
    wealth_before + income_kzt_eq - funded_expenses - liq_conversion_cost
        = wealth_after
where funded_expenses = expense_kzt_eq - shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config
from src.household.cashflows import NominalCashFlows, kzt_equivalent_expense, kzt_equivalent_income
from src.household.portfolio import (
    initial_holdings,
    rebalance_holdings_with_cost,
)


@dataclass
class HouseholdRun:
    monthly_shortfall: np.ndarray  # (n_paths, n_months)
    terminal_wealth: np.ndarray
    wealth_path: np.ndarray | None
    holdings_end: tuple[np.ndarray, np.ndarray, np.ndarray]
    max_conservation_error: float
    negative_holdings: bool
    has_nan: bool
    cumulative_conversion_cost: np.ndarray | None = None
    cumulative_conversion_volume: np.ndarray | None = None


def _should_rebalance(month_index0: int, frequency: str) -> bool:
    freq = (frequency or "monthly").lower()
    if freq in ("none", "no", "never"):
        return False
    if freq == "monthly":
        return True
    if freq == "quarterly":
        return (month_index0 + 1) % 3 == 0
    raise ValueError(f"unknown rebalance_frequency: {frequency}")


def simulate_household(
    monthly_fx: np.ndarray,
    cf: NominalCashFlows,
    weights: np.ndarray,
    W0: float,
    record_paths: bool = False,
    example_ids: tuple[int, ...] | None = None,
    rebalance_frequency: str | None = None,
    conversion_cost: float | None = None,
) -> HouseholdRun:
    """``monthly_fx`` shape (n_paths, n_months, 2) = [USD/KZT, EUR/KZT] month-end."""
    fx = np.asarray(monthly_fx, dtype=float)
    n_paths, n_months, _ = fx.shape
    w = np.asarray(weights, dtype=float)
    c = float(config.HH_CONVERSION_COST if conversion_cost is None else conversion_cost)
    if c < 0:
        raise ValueError("conversion_cost must be non-negative")
    freq = (
        config.HH_REBALANCE_FREQUENCY
        if rebalance_frequency is None
        else rebalance_frequency
    )

    S0u, S0e = cf.S_usd0, cf.S_eur0
    a0 = initial_holdings(W0, w, S0u, S0e)
    a_kzt = np.full(n_paths, a0[0])
    a_usd = np.full(n_paths, a0[1])
    a_eur = np.full(n_paths, a0[2])
    sf = np.zeros((n_paths, n_months))
    cost_cum = np.zeros(n_paths)
    vol_cum = np.zeros(n_paths)
    max_err = 0.0
    wealth_path = np.zeros((n_paths, n_months + 1)) if record_paths else None
    if wealth_path is not None:
        wealth_path[:, 0] = W0

    trace_rows = [] if example_ids is not None else None

    for m in range(n_months):
        S_u = fx[:, m, 0]
        S_e = fx[:, m, 1]
        wealth_before = a_kzt + a_usd * S_u + a_eur * S_e
        inc = kzt_equivalent_income(cf, S_u, S_e)
        exp = kzt_equivalent_expense(cf, S_u, S_e)

        a_kzt = a_kzt + cf.I_kzt
        a_usd = a_usd + cf.I_usd
        a_eur = a_eur + cf.I_eur

        pay_k = np.minimum(a_kzt, cf.E_kzt)
        pay_u = np.minimum(a_usd, cf.E_usd)
        pay_e = np.minimum(a_eur, cf.E_eur)
        a_kzt = a_kzt - pay_k
        a_usd = a_usd - pay_u
        a_eur = a_eur - pay_e
        gap_k = cf.E_kzt - pay_k
        gap_u = cf.E_usd - pay_u
        gap_e = cf.E_eur - pay_e
        G = gap_k + gap_u * S_u + gap_e * S_e
        W = a_kzt + a_usd * S_u + a_eur * S_e
        W = np.maximum(W, 0.0)

        # Gap funding with optional proportional conversion cost on take
        if c > 0:
            max_fund = W / (1.0 + c)
        else:
            max_fund = W
        take = np.minimum(G, max_fund)
        liq_fee = c * take
        liq = take + liq_fee
        shortfall = np.maximum(G - take, 0.0)
        frac = np.divide(liq, W, out=np.zeros_like(W), where=W > 1e-18)
        a_kzt = a_kzt * (1.0 - frac)
        a_usd = a_usd * (1.0 - frac)
        a_eur = a_eur * (1.0 - frac)
        dead = shortfall > config.HH_WEALTH_DEPLETE_TOL
        a_kzt = np.where(dead, 0.0, np.maximum(a_kzt, 0.0))
        a_usd = np.where(dead, 0.0, np.maximum(a_usd, 0.0))
        a_eur = np.where(dead, 0.0, np.maximum(a_eur, 0.0))

        wealth_after = a_kzt + a_usd * S_u + a_eur * S_e
        funded = exp - shortfall
        err = np.abs(wealth_before + inc - funded - liq_fee - wealth_after)
        max_err = max(max_err, float(np.max(err)) if err.size else 0.0)
        sf[:, m] = shortfall
        cost_cum = cost_cum + liq_fee
        vol_cum = vol_cum + take

        reb_fee = np.zeros(n_paths)
        if _should_rebalance(m, freq):
            a_kzt, a_usd, a_eur, reb_fee = rebalance_holdings_with_cost(
                a_kzt, a_usd, a_eur, w, S_u, S_e, c
            )
            cost_cum = cost_cum + reb_fee
            vol_cum = vol_cum + np.divide(
                reb_fee, c, out=np.zeros_like(reb_fee), where=c > 0
            )
        wealth_reb = a_kzt + a_usd * S_u + a_eur * S_e
        # Rebalance: wealth drops only by reb_fee
        max_err = max(
            max_err,
            float(np.max(np.abs(wealth_reb + reb_fee - wealth_after))),
        )

        if wealth_path is not None:
            wealth_path[:, m + 1] = wealth_reb
        if trace_rows is not None:
            for pid in example_ids:
                trace_rows.append(
                    {
                        "path_id": int(pid),
                        "month": m + 1,
                        "usd_kzt": float(S_u[pid]),
                        "eur_kzt": float(S_e[pid]),
                        "income_kzt_equivalent": float(inc[pid]),
                        "expenses_kzt_equivalent": float(exp[pid]),
                        "wealth_before_cashflow": float(wealth_before[pid]),
                        "funded_expenses": float(funded[pid]),
                        "monthly_shortfall": float(shortfall[pid]),
                        "liq_conversion_cost": float(liq_fee[pid]),
                        "rebalance_conversion_cost": float(reb_fee[pid]),
                        "wealth_after_cashflow": float(wealth_after[pid]),
                        "wealth_after_rebalance": float(wealth_reb[pid]),
                        "kzt_holding": float(a_kzt[pid]),
                        "usd_holding": float(a_usd[pid]),
                        "eur_holding": float(a_eur[pid]),
                    }
                )

    neg = bool(np.any(a_kzt < -1e-12) or np.any(a_usd < -1e-12) or np.any(a_eur < -1e-12))
    tw = a_kzt + a_usd * fx[:, -1, 0] + a_eur * fx[:, -1, 1]
    has_nan = bool(
        np.isnan(sf).any()
        or np.isnan(tw).any()
        or np.isinf(sf).any()
        or np.isinf(tw).any()
        or np.isnan(cost_cum).any()
    )
    run = HouseholdRun(
        monthly_shortfall=sf,
        terminal_wealth=tw,
        wealth_path=wealth_path,
        holdings_end=(a_kzt, a_usd, a_eur),
        max_conservation_error=max_err,
        negative_holdings=neg,
        has_nan=has_nan,
        cumulative_conversion_cost=cost_cum,
        cumulative_conversion_volume=vol_cum,
    )
    if trace_rows is not None:
        run.trace_rows = trace_rows  # type: ignore[attr-defined]
    return run
