"""Stress calibration across profiles × E/I × buffers × horizons × benchmarks."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.household.cashflows import daily_paths_to_monthly_rates, nominal_cashflows
from src.household.profiles import PRIMARY_PROFILES, strategy_catalog
from src.household.risk_metrics import summarize_run
from src.household.settlement import simulate_household
from src.simulation.path_builder import returns_to_levels
from src.simulation.stationary_bootstrap import StationaryBootstrapSimulator
from src.simulation.validation import load_baseline_bundle
from src.utils import write_csv


plt.rcParams.update(
    {"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3}
)


def _dirs() -> None:
    for p in (config.HH_FIGURES, config.HH_TABLES, config.HH_DIAGNOSTICS):
        p.mkdir(parents=True, exist_ok=True)


def generate_common_fx_paths() -> tuple[np.ndarray, np.ndarray]:
    """One SB20 draw of 10,000 × 1260 days; reused for all households/strategies."""
    bundle = load_baseline_bundle()
    sim = StationaryBootstrapSimulator(expected_block_length=config.HH_BLOCK_LENGTH)
    sim.fit(bundle.returns)
    r = sim.simulate(
        config.HH_N_PATHS,
        config.HORIZON_5Y_DAYS,
        random_seed=config.HH_RANDOM_SEED,
    )
    levels = returns_to_levels(r, bundle.s0)
    cache = config.HH_DIR / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "sb20_daily_levels.npy", levels)
    return levels, bundle.s0


def run_stress_calibration(levels: np.ndarray, s0: np.ndarray) -> tuple[pd.DataFrame, dict]:
    S_u0, S_e0 = float(s0[0]), float(s0[1])
    rows = []
    audit = {
        "max_conservation_error": 0.0,
        "negative_holdings": False,
        "has_nan": False,
    }
    trace = None
    for years in config.HH_HORIZON_YEARS:
        days = years * config.HORIZON_1Y_DAYS
        monthly = daily_paths_to_monthly_rates(levels[:, : days + 1, :])
        n_months = monthly.shape[1]
        for profile in PRIMARY_PROFILES:
            for ratio in config.HH_EXPENSE_INCOME_RATIOS:
                E0 = ratio * config.HH_I0
                cf = nominal_cashflows(profile, config.HH_I0, E0, S_u0, S_e0)
                for B in config.HH_BUFFER_MONTHS:
                    W0 = B * E0
                    for strat_name, w in strategy_catalog(profile):
                        want_trace = (
                            trace is None
                            and profile.name.startswith("P4")
                            and years == 1
                            and abs(ratio - 0.95) < 1e-12
                            and B == 6
                            and strat_name == "B4_equal"
                        )
                        run = simulate_household(
                            monthly,
                            cf,
                            w,
                            W0,
                            record_paths=False,
                            example_ids=config.HH_EXAMPLE_PATH_IDS if want_trace else None,
                        )
                        audit["max_conservation_error"] = max(
                            audit["max_conservation_error"], run.max_conservation_error
                        )
                        audit["negative_holdings"] = audit["negative_holdings"] or run.negative_holdings
                        audit["has_nan"] = audit["has_nan"] or run.has_nan
                        if want_trace and hasattr(run, "trace_rows"):
                            trace = pd.DataFrame(run.trace_rows)
                        stats = summarize_run(run.monthly_shortfall, run.terminal_wealth, E0, n_months)
                        rows.append(
                            {
                                "profile": profile.name,
                                "horizon_years": years,
                                "expense_income_ratio": ratio,
                                "buffer_months": B,
                                "strategy": strat_name,
                                "w_kzt": float(w[0]),
                                "w_usd": float(w[1]),
                                "w_eur": float(w[2]),
                                **stats,
                            }
                        )
    table = pd.DataFrame(rows)
    write_csv(table, config.HH_TABLES / "stress_calibration.csv")
    if trace is not None:
        write_csv(trace, config.HH_DIAGNOSTICS / "example_path_accounting.csv")
    return table, audit


def _save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(config.HH_FIGURES / f"{stem}.png")
    fig.savefig(config.HH_FIGURES / f"{stem}.pdf")
    plt.close(fig)


def make_figures(table: pd.DataFrame) -> None:
    t = table.copy()
    # A: P(shortfall) vs buffer, 3y, E/I=0.95, all-KZT
    sub = t[
        (t.horizon_years == 3)
        & (t.expense_income_ratio == 0.95)
        & (t.strategy == "B1_all_kzt")
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for prof, g in sub.groupby("profile"):
        g = g.sort_values("buffer_months")
        ax.plot(g.buffer_months, g.shortfall_probability, marker="o", label=prof)
    ax.set_xlabel("Liquidity buffer (months of E0)")
    ax.set_ylabel("P(cumulative shortfall > 0)")
    ax.set_title("Figure A. Shortfall probability vs buffer (3y, E/I=0.95, all-KZT)")
    ax.legend(fontsize=8)
    _save(fig, "figA_shortfall_vs_buffer")

    # B: CVaR
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for prof, g in sub.groupby("profile"):
        g = g.sort_values("buffer_months")
        ax.plot(g.buffer_months, g.cvar95_over_annual_expenses, marker="o", label=prof)
    ax.set_xlabel("Liquidity buffer (months of E0)")
    ax.set_ylabel("CVaR95 / annual expenses")
    ax.set_title("Figure B. CVaR95 vs buffer (3y, E/I=0.95, all-KZT)")
    ax.legend(fontsize=8)
    _save(fig, "figB_cvar_vs_buffer")

    # C: by E/I, 3y, B=6, all-KZT
    subc = t[(t.horizon_years == 3) & (t.buffer_months == 6) & (t.strategy == "B1_all_kzt")]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for prof, g in subc.groupby("profile"):
        g = g.sort_values("expense_income_ratio")
        ax.plot(g.expense_income_ratio, g.shortfall_probability, marker="o", label=prof)
    ax.set_xlabel("E0 / I0")
    ax.set_ylabel("P(shortfall > 0)")
    ax.set_title("Figure C. Shortfall probability vs expense/income (3y, B=6, all-KZT)")
    ax.legend(fontsize=8)
    _save(fig, "figC_shortfall_vs_ratio")

    # D: terminal wealth — need a re-sim for histograms? Table only has summaries.
    # Skip raw hist unless we re-run one config — store p05/median/p95 as bars.
    subd = t[
        (t.horizon_years == 3)
        & (t.expense_income_ratio == 0.95)
        & (t.buffer_months == 6)
        & (t.profile.isin(["P1_local_control", "P2_foreign_income_mismatch", "P4_full_multi_currency"]))
        & (t.strategy == "B1_all_kzt")
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(subd))
    ax.bar(x - 0.2, subd.p05_terminal_wealth, width=0.2, label="p5")
    ax.bar(x, subd.median_terminal_wealth, width=0.2, label="median")
    ax.bar(x + 0.2, subd.p95_terminal_wealth, width=0.2, label="p95")
    ax.set_xticks(x)
    ax.set_xticklabels(subd.profile.str.replace("_", "\n"), fontsize=8)
    ax.set_ylabel("Terminal wealth (normalized KZT)")
    ax.set_title("Figure D. Terminal wealth quantiles (3y, E/I=0.95, B=6, all-KZT)")
    ax.legend()
    _save(fig, "figD_terminal_wealth_quantiles")

    # E: strategies for P2 and P4
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, prof in zip(axes, ("P2_foreign_income_mismatch", "P4_full_multi_currency")):
        g = t[
            (t.profile == prof)
            & (t.horizon_years == 3)
            & (t.expense_income_ratio == 0.95)
            & (t.buffer_months == 6)
        ]
        ax.bar(g.strategy, g.shortfall_probability)
        ax.set_title(prof)
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        ax.set_ylabel("P(shortfall > 0)")
    fig.suptitle("Figure E. Strategies at 3y, E/I=0.95, B=6")
    _save(fig, "figE_strategies_P2_P4")


def select_baseline_candidate(table: pd.DataFrame) -> dict:
    """Apply the pre-specified experimental-design rule (not an optimum)."""
    notes = []
    p1 = table[(table.profile == "P1_local_control") & (table.strategy == "B1_all_kzt")]
    candidates = []
    for (ratio, B, years), _ in table.groupby(["expense_income_ratio", "buffer_months", "horizon_years"]):
        p1p = p1[
            (p1.expense_income_ratio == ratio)
            & (p1.buffer_months == B)
            & (p1.horizon_years == years)
        ]
        if p1p.empty:
            continue
        p1_prob = float(p1p.shortfall_probability.iloc[0])
        fx = table[
            (table.profile.isin(
                ["P2_foreign_income_mismatch", "P3_mixed_income", "P4_full_multi_currency"]
            ))
            & (table.expense_income_ratio == ratio)
            & (table.buffer_months == B)
            & (table.horizon_years == years)
        ]
        hit = fx[(fx.shortfall_probability >= 0.05) & (fx.shortfall_probability <= 0.40)]
        if p1_prob <= 0.05 and len(hit):
            candidates.append(
                {
                    "expense_income_ratio": ratio,
                    "buffer_months": int(B),
                    "horizon_years": int(years),
                    "p1_all_kzt_shortfall_prob": p1_prob,
                    "n_fx_benchmarks_in_5_40": int(len(hit)),
                    "example_fx_row": hit.sort_values("shortfall_probability").iloc[-1].to_dict(),
                }
            )
    # Prefer 3-year horizon as the paper-relevant medium term if available
    preferred = [c for c in candidates if c["horizon_years"] == 3]
    pool = preferred or candidates
    # Among remaining, prefer E/I=0.95 (near break-even) then larger buffer still informative
    pool = sorted(
        pool,
        key=lambda c: (
            abs(c["expense_income_ratio"] - 0.95),
            abs(c["buffer_months"] - 6),
            -c["n_fx_benchmarks_in_5_40"],
        ),
    )
    chosen = pool[0] if pool else None
    if chosen is None:
        notes.append("No (ratio, B, horizon) jointly met P1<=5% and an FX profile in 5–40%.")
    return {"rule_satisfied": chosen is not None, "candidate": chosen, "n_candidates": len(candidates), "notes": notes}


def write_calibration_report(table: pd.DataFrame, audit: dict, selection: dict) -> None:
    p1k = table[(table.profile == "P1_local_control") & (table.strategy == "B1_all_kzt")]
    zero = table[table.shortfall_probability < 0.01]
    sure = table[table.shortfall_probability > 0.99]
    mid = table[(table.shortfall_probability >= 0.05) & (table.shortfall_probability <= 0.40)]
    cand = selection.get("candidate")
    cand_txt = json.dumps(cand, indent=2, default=str) if cand else "None"

    p1_ei_lt1 = p1k[p1k.expense_income_ratio < 1.0]
    p1_max_prob_surplus = float(p1_ei_lt1.shortfall_probability.max()) if len(p1_ei_lt1) else np.nan

    by_prof = (
        table.groupby("profile")[["shortfall_probability", "cvar95_over_annual_expenses"]]
        .mean()
        .round(4)
        .to_string()
    )
    text = f"""# Phase 2B1 Household Stress-Calibration Report

FX DGP: joint stationary bootstrap **L={config.HH_BLOCK_LENGTH}**, seed **{config.HH_RANDOM_SEED}**,
n_paths=**{config.HH_N_PATHS}**. Monthly settlement: **{config.HH_WEEKDAYS_PER_MONTH} weekdays = 1 month**.
I0=**{config.HH_I0}** normalized KZT-equivalent. Conversion cost 0. Monthly target rebalancing.
CVaR95 = mean of the largest k=ceil(0.05 N) path losses (k=500 at N=10,000).

Accounting audit: max conservation error **{audit['max_conservation_error']:.3e}**;
negative holdings **{audit['negative_holdings']}**; NaN/Inf **{audit['has_nan']}**.

## 1. Essentially zero shortfall risk

Configurations with P(L>0) < 1%: **{len(zero)}** of {len(table)}.
Typical: **E0/I0 = 0.80** (structural surplus 0.20 per month) and/or large B with
local-currency matched cash flows (P1 all-KZT). Surplus households never need
the buffer for P1 all-KZT.

P1 all-KZT when E0<I0: max shortfall probability **{p1_max_prob_surplus:.4f}**
(should be ~0 if accounting is FX-insulated).

## 2. Near-certain shortfall

P(L>0) > 99%: **{len(sure)}** rows. Typical: **E0/I0 = 1.05** (structural deficit
0.05/month). Buffer B months of expenses lasts about B×E0/0.05 = **21B months**
for a pure KZT deficit, so 5-year (60m) P1 all-KZT is insolvent for small B.

## 3. Intermediate downside risk

P(L>0) in [5%, 40%]: **{len(mid)}** rows. These are the informative experimental
design region.

## 4. Does P1 all-KZT behave as expected?

P1 has 100% KZT income and expenses. All-KZT savings should not create FX-driven
shortfall when E0 ≤ I0. See max probability above. If E0=1.05, shortfall is
**cash-flow insolvency**, not FX. P1 portfolios with USD/EUR can still show FX risk.

## 5. Which profiles are most FX-sensitive?

Mean P(shortfall) and mean CVaR95/annual expenses across the full grid:

{by_prof}

P2 (USD income, KZT expenses) loads entirely on USD/KZT. P4 adds EUR income,
EUR expenses, and EUR savings. P3 is a 50/50 blend. P1 is the control.

## 6. Expense/income ratio vs liquidity buffer

E/I **shifts the mean surplus/deficit** and can force P(shortfall) to 0 or 1
regardless of currency weights. Buffer B **delays** depletion under a deficit
and **absorbs** FX-driven income shortfalls under near-break-even (0.95).
Both matter; E/I=1.05 dominates unless B is large relative to horizon.

## 7. Do strategy rankings already differ across P1–P4?

Yes, qualitatively (no optimization):
- P1: all-KZT / expense-matched / income-matched coincide and should dominate
  FX-heavy strategies if E0≤I0.
- P2: income-matched = all-USD; expense-matched = all-KZT. Ranking depends on
  whether USD income volatility or KZT expense matching matters more at the
  chosen B and horizon — see Figure E.
- P4: expense-matched (0.8, 0, 0.2) vs income-matched (0, 0.5, 0.5) are distinct.

## 8–10. Baseline candidate (not an optimum)

Pre-specified rule: P1 all-KZT P(shortfall) ≤ 5% **and** at least one P2/P3/P4
benchmark in 5–40%. Preference among matches: 3-year horizon, E/I near 0.95, B near 6.

Rule satisfied: **{selection['rule_satisfied']}** ({selection['n_candidates']} matching cells).

Candidate:

```
{cand_txt}
```

{selection.get('notes') or ''}

Do **not** interpret this as an optimal currency allocation. Phase 2B2 may use
this cell as the household environment in which to compare weights.

Configurations tested: {len(table)}.
"""
    (config.HH_DIAGNOSTICS / "calibration_report.md").write_text(text, encoding="utf-8")


def run_phase2b1() -> dict:
    _dirs()
    levels, s0 = generate_common_fx_paths()
    table, audit = run_stress_calibration(levels, s0)
    make_figures(table)
    selection = select_baseline_candidate(table)
    write_calibration_report(table, audit, selection)
    p1 = table[(table.profile == "P1_local_control") & (table.strategy == "B1_all_kzt")]
    summary = {
        "n_configs": int(len(table)),
        "s0_usd": float(s0[0]),
        "s0_eur": float(s0[1]),
        "max_conservation_error": audit["max_conservation_error"],
        "negative_holdings": audit["negative_holdings"],
        "has_nan": audit["has_nan"],
        "p1_surplus_max_shortfall_prob": float(
            p1[p1.expense_income_ratio < 1].shortfall_probability.max()
        ),
        "rule_satisfied": selection["rule_satisfied"],
        "candidate": selection["candidate"],
    }
    (config.HH_DIAGNOSTICS / "phase2b1_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
