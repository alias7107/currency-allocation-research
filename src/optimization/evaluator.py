"""Evaluate household path losses for candidate currency weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.household.cashflows import NominalCashFlows, daily_paths_to_monthly_rates
from src.household.profiles import HouseholdProfile, strategy_catalog
from src.household.risk_metrics import (
    cumulative_shortfall,
    empirical_cvar,
    empirical_var,
    max_monthly_shortfall,
    summarize_run,
)
from src.household.settlement import simulate_household
from src.simulation.path_builder import returns_to_levels
from src.simulation.stationary_bootstrap import StationaryBootstrapSimulator
from src.simulation.validation import load_baseline_bundle


@dataclass
class EvalResult:
    weights: np.ndarray
    path_losses: np.ndarray
    terminal_wealth: np.ndarray
    monthly_shortfall: np.ndarray
    cvar95: float
    var95: float
    shortfall_probability: float
    expected_cumulative_shortfall: float
    mean_max_monthly_shortfall: float
    wealth_depletion_probability: float
    mean_terminal_wealth: float
    median_terminal_wealth: float
    p05_terminal_wealth: float
    p95_terminal_wealth: float


def load_or_generate_fx_levels(
    n_paths: int | None = None,
    seed: int | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (daily_levels, s0). Prefer Phase 2B1 cache when n_paths/seed match."""
    n_paths = n_paths or config.OPT_N_PATHS
    seed = seed if seed is not None else config.OPT_RANDOM_SEED
    cache = config.HH_DIR / "cache" / "sb20_daily_levels.npy"
    if (
        use_cache
        and n_paths == config.HH_N_PATHS
        and seed == config.HH_RANDOM_SEED
        and cache.exists()
    ):
        levels = np.load(cache)
        s0 = levels[0, 0, :].copy()
        return levels, s0
    bundle = load_baseline_bundle()
    sim = StationaryBootstrapSimulator(expected_block_length=config.OPT_BLOCK_LENGTH)
    sim.fit(bundle.returns)
    r = sim.simulate(n_paths, config.HORIZON_5Y_DAYS, random_seed=seed)
    levels = returns_to_levels(r, bundle.s0)
    return levels, bundle.s0


def monthly_fx_for_horizon(daily_levels: np.ndarray, horizon_years: int) -> np.ndarray:
    days = horizon_years * config.HORIZON_1Y_DAYS
    return daily_paths_to_monthly_rates(daily_levels[:, : days + 1, :])


def primary_cashflows(profile: HouseholdProfile, s0: np.ndarray) -> tuple[NominalCashFlows, float, float]:
    from src.household.cashflows import nominal_cashflows

    I0 = config.OPT_I0
    E0 = config.OPT_EXPENSE_INCOME_RATIO * I0
    W0 = config.OPT_BUFFER_MONTHS * E0
    cf = nominal_cashflows(profile, I0, E0, float(s0[0]), float(s0[1]))
    return cf, E0, W0


def evaluate_weights(
    monthly_fx: np.ndarray,
    cf: NominalCashFlows,
    W0: float,
    weights: np.ndarray,
    store_monthly: bool = False,
    rebalance_frequency: str | None = None,
    conversion_cost: float | None = None,
) -> EvalResult:
    w = np.asarray(weights, dtype=float).reshape(3)
    run = simulate_household(
        monthly_fx,
        cf,
        w,
        W0,
        rebalance_frequency=rebalance_frequency,
        conversion_cost=conversion_cost,
    )
    L = cumulative_shortfall(run.monthly_shortfall)
    stats = summarize_run(run.monthly_shortfall, run.terminal_wealth, cf.E0, monthly_fx.shape[1])
    ev = EvalResult(
        weights=w,
        path_losses=L,
        terminal_wealth=run.terminal_wealth,
        monthly_shortfall=run.monthly_shortfall if store_monthly else run.monthly_shortfall,
        cvar95=stats["cvar95_cumulative_shortfall"],
        var95=stats["var95_cumulative_shortfall"],
        shortfall_probability=stats["shortfall_probability"],
        expected_cumulative_shortfall=stats["expected_cumulative_shortfall"],
        mean_max_monthly_shortfall=stats["mean_max_monthly_shortfall"],
        wealth_depletion_probability=stats["wealth_depletion_probability"],
        mean_terminal_wealth=stats["mean_terminal_wealth"],
        median_terminal_wealth=stats["median_terminal_wealth"],
        p05_terminal_wealth=stats["p05_terminal_wealth"],
        p95_terminal_wealth=stats["p95_terminal_wealth"],
    )
    # Attach cost diagnostics when present (not part of dataclass fields)
    if run.cumulative_conversion_cost is not None:
        ev.mean_cumulative_conversion_cost = float(np.mean(run.cumulative_conversion_cost))  # type: ignore[attr-defined]
        ev.mean_cumulative_conversion_volume = float(np.mean(run.cumulative_conversion_volume))  # type: ignore[attr-defined]
        ev.path_conversion_cost = run.cumulative_conversion_cost  # type: ignore[attr-defined]
    return ev


def evaluate_grid(
    monthly_fx: np.ndarray,
    cf: NominalCashFlows,
    W0: float,
    grid: np.ndarray,
    rebalance_frequency: str | None = None,
    conversion_cost: float | None = None,
    grid_is_bp: bool = False,
) -> pd.DataFrame:
    """Evaluate decimal or integer-bp weight grid; always stores bp keys."""
    from src.optimization.grid import TOTAL_BP, bp_to_weights, weights_to_bp

    rows = []
    grid = np.asarray(grid)
    for row in grid:
        if grid_is_bp:
            bp = np.asarray(row, dtype=np.int64).reshape(3)
            w = bp_to_weights(bp)
        else:
            w = np.asarray(row, dtype=float).reshape(3)
            bp = weights_to_bp(w).reshape(3)
        ev = evaluate_weights(
            monthly_fx,
            cf,
            W0,
            w,
            rebalance_frequency=rebalance_frequency,
            conversion_cost=conversion_cost,
        )
        rec = {
            "w_kzt_bp": int(bp[0]),
            "w_usd_bp": int(bp[1]),
            "w_eur_bp": int(bp[2]),
            "w_kzt": float(bp[0]) / TOTAL_BP,
            "w_usd": float(bp[1]) / TOTAL_BP,
            "w_eur": float(bp[2]) / TOTAL_BP,
            "cvar95": ev.cvar95,
            "var95": ev.var95,
            "shortfall_probability": ev.shortfall_probability,
            "expected_cumulative_shortfall": ev.expected_cumulative_shortfall,
            "mean_max_monthly_shortfall": ev.mean_max_monthly_shortfall,
            "wealth_depletion_probability": ev.wealth_depletion_probability,
            "mean_terminal_wealth": ev.mean_terminal_wealth,
            "median_terminal_wealth": ev.median_terminal_wealth,
            "p05_terminal_wealth": ev.p05_terminal_wealth,
            "p95_terminal_wealth": ev.p95_terminal_wealth,
        }
        if hasattr(ev, "mean_cumulative_conversion_cost"):
            rec["mean_cumulative_conversion_cost"] = ev.mean_cumulative_conversion_cost
            rec["mean_cumulative_conversion_volume"] = ev.mean_cumulative_conversion_volume
        rows.append(rec)
    return pd.DataFrame(rows)


def verify_phase2b1_benchmarks(
    daily_levels: np.ndarray,
    s0: np.ndarray,
    phase2b1_table: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce Phase 2B1 metrics for primary env benchmarks; raise if mismatch."""
    rows = []
    ratio = config.OPT_EXPENSE_INCOME_RATIO
    B = config.OPT_BUFFER_MONTHS
    for years in config.OPT_HORIZON_YEARS:
        monthly = monthly_fx_for_horizon(daily_levels, years)
        for profile in __import__("src.household.profiles", fromlist=["PRIMARY_PROFILES"]).PRIMARY_PROFILES:
            cf, E0, W0 = primary_cashflows(profile, s0)
            for name, w in strategy_catalog(profile):
                ev = evaluate_weights(monthly, cf, W0, w)
                ref = phase2b1_table[
                    (phase2b1_table["profile"] == profile.name)
                    & (phase2b1_table["horizon_years"] == years)
                    & (np.isclose(phase2b1_table["expense_income_ratio"], ratio))
                    & (phase2b1_table["buffer_months"] == B)
                    & (phase2b1_table["strategy"] == name)
                ]
                if ref.empty:
                    status = "missing_in_phase2b1"
                    ok = False
                    ref_cvar = np.nan
                else:
                    ref_cvar = float(ref["cvar95_cumulative_shortfall"].iloc[0])
                    ref_p = float(ref["shortfall_probability"].iloc[0])
                    ok = np.isclose(
                        ev.cvar95,
                        ref_cvar,
                        rtol=config.OPT_BENCHMARK_REPRO_RTOL,
                        atol=config.OPT_BENCHMARK_REPRO_ATOL,
                    ) and np.isclose(
                        ev.shortfall_probability,
                        ref_p,
                        rtol=config.OPT_BENCHMARK_REPRO_RTOL,
                        atol=1e-4,
                    )
                    status = "ok" if ok else "MISMATCH"
                rows.append(
                    {
                        "profile": profile.name,
                        "horizon_years": years,
                        "strategy": name,
                        "opt_cvar95": ev.cvar95,
                        "phase2b1_cvar95": ref_cvar,
                        "opt_shortfall_prob": ev.shortfall_probability,
                        "status": status,
                        "ok": ok,
                    }
                )
    out = pd.DataFrame(rows)
    bad = out[~out["ok"]]
    if len(bad):
        raise RuntimeError(
            "Phase 2B1 benchmark reproduction failed for:\n"
            + bad.to_string(index=False)
        )
    return out
