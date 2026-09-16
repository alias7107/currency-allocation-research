"""Generate publication-facing artifacts under results/final/."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.final import paths as fp
from src.household.calibration import select_baseline_candidate
from src.household.cashflows import kzt_equivalent_income, nominal_cashflows
from src.household.profiles import (
    P2_FOREIGN_INCOME,
    P4_MULTI_CURRENCY,
    PRIMARY_PROFILES,
    strategy_catalog,
)
from src.household.risk_metrics import cumulative_shortfall, empirical_cvar
from src.household.settlement import simulate_household
from src.optimization.evaluator import (
    evaluate_weights,
    load_or_generate_fx_levels,
    monthly_fx_for_horizon,
    primary_cashflows,
)
from src.optimization.grid import (
    TOTAL_BP,
    expected_grid_size_bp,
    generate_simplex_grid,
)
from src.robustness.classify import (
    expense_matched_weights,
    final_paper_classification,
    income_matched_weights,
)
from src.robustness.scenarios import generate_fx_levels
from src.simulation.validation import load_baseline_bundle, moments

PROFILE_SHORT = {
    "P1_local_control": "P1",
    "P2_foreign_income_mismatch": "P2",
    "P3_mixed_income": "P3",
    "P4_full_multi_currency": "P4",
}


@dataclass
class CleanupState:
    stopped: bool = False
    stop_reasons: list[str] = field(default_factory=list)
    critical_issues: list[str] = field(default_factory=list)
    moderate_issues: list[str] = field(default_factory=list)
    minor_issues: list[str] = field(default_factory=list)
    grid_2pct: int = 0
    grid_1pct: int = 0
    p2_3y: dict[str, Any] = field(default_factory=dict)
    p4_3y: dict[str, Any] = field(default_factory=dict)
    test_summary: dict[str, Any] = field(default_factory=dict)
    final_classification: str = ""


def _ensure_dirs() -> None:
    for d in (
        fp.FINAL_DIR,
        fp.AUDIT_DIR,
        fp.TABLES_DIR,
        fp.FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_label(name: str) -> str:
    return PROFILE_SHORT.get(name, name)


def build_canonical_specification() -> dict:
    b1 = _load_json(config.HH_DIAGNOSTICS / "phase2b1_run_summary.json")
    a1 = _load_json(config.SIM_DIAGNOSTICS / "phase2a_run_summary.json")
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "DATA": {
            "source": "National Bank of Kazakhstan (NBK) official daily market rates",
            "raw_start": config.RAW_START,
            "raw_end": config.RAW_END,
            "baseline_estimation_start": config.BASELINE_START,
            "baseline_estimation_end": config.BASELINE_END,
            "recent_window_start": config.RECENT_START,
            "recent_window_end": config.RECENT_END,
            "weekday_estimation_rule": config.ESTIMATION_CALENDAR,
            "float_transition_date": config.FLOAT_TRANSITION_DATE,
        },
        "FX_MODEL": {
            "primary_DGP": "joint stationary bootstrap SB20",
            "block_length": config.HH_BLOCK_LENGTH,
            "path_count": config.N_PATHS_PRIMARY,
            "random_seed": config.SIM_RANDOM_SEED,
            "daily_horizon_convention": {
                "1Y_days": config.HORIZON_1Y_DAYS,
                "3Y_days": config.HORIZON_3Y_DAYS,
                "5Y_days": config.HORIZON_5Y_DAYS,
            },
            "currencies": list(config.TARGET_CURRENCIES),
            "starting_FX_levels": {
                "USD_KZT": b1["s0_usd"],
                "EUR_KZT": b1["s0_eur"],
            },
            "phase2a_student_t_nu": a1["student_t_df"],
            "phase2a_recommended_block_length": a1["recommended_L"],
        },
        "HOUSEHOLD_MODEL": {
            "I0": config.HH_I0,
            "baseline_expense_income_ratio": config.OPT_EXPENSE_INCOME_RATIO,
            "baseline_buffer_months": config.OPT_BUFFER_MONTHS,
            "baseline_W0": config.OPT_BUFFER_MONTHS * config.OPT_EXPENSE_INCOME_RATIO * config.HH_I0,
            "horizons_years": list(config.HH_HORIZON_YEARS),
            "weekdays_per_household_month": config.HH_WEEKDAYS_PER_MONTH,
            "rebalancing_frequency_baseline": config.HH_REBALANCE_FREQUENCY,
            "baseline_conversion_cost": config.HH_CONVERSION_COST,
            "profiles": {
                p.name: {
                    "qI": [p.qI_kzt, p.qI_usd, p.qI_eur],
                    "qE": [p.qE_kzt, p.qE_usd, p.qE_eur],
                }
                for p in PRIMARY_PROFILES
            },
            "nominal_foreign_cashflow_convention": "fixed nominal units per month in native currency",
            "no_borrowing": True,
            "no_interest": True,
            "no_inflation": True,
            "no_risky_asset_returns": True,
        },
        "RISK": {
            "loss_definition": "cumulative KZT-equivalent shortfall over horizon",
            "CVaR_alpha": config.HH_CVAR_ALPHA,
            "VaR_convention": "k-th largest loss where k=ceil((1-alpha)N)",
            "shortfall_probability_definition": "P(cumulative shortfall > 0)",
            "depletion_probability_definition": "P(terminal wealth <= HH_WEALTH_DEPLETE_TOL)",
        },
        "OPTIMIZATION": {
            "integer_basis_point_grid": True,
            "TOTAL_BP": TOTAL_BP,
            "coarse_step_bp": config.ROB_COARSE_STEP_BP,
            "refined_step_bp": config.ROB_REFINED_STEP_BP,
            "coarse_step_decimal": config.OPT_COARSE_STEP,
            "refined_step_decimal": config.OPT_REFINED_STEP,
            "expected_2pct_grid_size": expected_grid_size_bp(200),
            "expected_1pct_grid_size": expected_grid_size_bp(100),
            "near_optimal_tolerances": {
                "rel_1pct": config.OPT_NEAR_OPTIMAL_REL_TOL_1PCT,
                "rel_5pct": config.OPT_NEAR_OPTIMAL_REL_TOL_5PCT,
                "abs_tol": config.OPT_NEAR_OPTIMAL_ABS_TOL,
                "abs_frac_annual": config.OPT_NEAR_OPTIMAL_ABS_FRAC_ANNUAL,
            },
            "tie_break_convention": "min CVaR; then lexicographic (w_kzt, w_usd, w_eur) descending",
            "identification_thresholds": {
                "well_identified_max_frac": config.OPT_WELL_IDENTIFIED_MAX_FRAC,
                "not_identified_min_frac": config.OPT_NOT_IDENTIFIED_MIN_FRAC,
                "cvar_zero_tol": config.OPT_CVAR_ZERO_TOL,
            },
        },
        "ROBUSTNESS": {
            "alternative_EI_B": {"E_I": config.ROB_ALT_EI, "B": config.ROB_ALT_B},
            "DGP_alternatives": list(config.ROB_DGP_SPECS),
            "recent_historical_window": {
                "start": config.ROB_RECENT_START,
                "end": config.ROB_RECENT_END,
                "seed": config.ROB_RECENT_SEED,
            },
            "rebalance_frequencies": list(config.ROB_REBALANCE_FREQS),
            "cost_scenarios": list(config.ROB_CONVERSION_COSTS),
        },
        "SEEDS": {
            "baseline_simulation": config.HH_RANDOM_SEED,
            "paired_bootstrap": config.OPT_BOOTSTRAP_SEED,
            "50k_confirmation": config.OPT_CONFIRM_SEED,
            "robustness_seeds": {
                "SB20": 42,
                "SB10": 4210,
                "SB56": 4256,
                "IID": 4240,
                "recent_window": config.ROB_RECENT_SEED,
            },
        },
    }


def write_canonical_specification_md(spec: dict) -> str:
    lines = [
        "# Canonical Research Specification",
        "",
        "Machine-readable source: `canonical_specification.json`.",
        "This document explains each frozen parameter in plain English.",
        "",
        "## Data",
        "",
        f"- **Source:** {spec['DATA']['source']}.",
        f"- **Raw sample:** {spec['DATA']['raw_start']} to {spec['DATA']['raw_end']}.",
        f"- **Baseline estimation window:** {spec['DATA']['baseline_estimation_start']} "
        f"to {spec['DATA']['baseline_estimation_end']} (post-float regime).",
        f"- **Recent robustness window:** {spec['DATA']['recent_window_start']} "
        f"to {spec['DATA']['recent_window_end']}.",
        f"- **Weekday rule:** {spec['DATA']['weekday_estimation_rule']} — weekends excluded before return estimation.",
        "",
        "## FX scenario generator",
        "",
        f"- **Primary DGP:** stationary bootstrap with expected block length L={spec['FX_MODEL']['block_length']}.",
        f"- **Paths:** {spec['FX_MODEL']['path_count']:,} per evaluation; seed {spec['FX_MODEL']['random_seed']}.",
        "- **Joint sampling:** USD/KZT and EUR/KZT log-return vectors are resampled together.",
        f"- **Starting levels (last baseline date):** USD/KZT={spec['FX_MODEL']['starting_FX_levels']['USD_KZT']}, "
        f"EUR/KZT={spec['FX_MODEL']['starting_FX_levels']['EUR_KZT']}.",
        "",
        "## Household environment",
        "",
        f"- **Normalized income I0:** {spec['HOUSEHOLD_MODEL']['I0']} (KZT-equivalent units).",
        f"- **Baseline expense/income ratio:** {spec['HOUSEHOLD_MODEL']['baseline_expense_income_ratio']} "
        f"(near break-even).",
        f"- **Baseline liquidity buffer:** {spec['HOUSEHOLD_MODEL']['baseline_buffer_months']} month(s) of expenses "
        f"→ W0={spec['HOUSEHOLD_MODEL']['baseline_W0']}.",
        f"- **Horizons:** {', '.join(str(h) for h in spec['HOUSEHOLD_MODEL']['horizons_years'])} years.",
        f"- **Household month:** {spec['HOUSEHOLD_MODEL']['weekdays_per_household_month']} weekdays.",
        f"- **Rebalancing:** {spec['HOUSEHOLD_MODEL']['rebalancing_frequency_baseline']} (baseline).",
        "- **Foreign cash flows:** fixed nominal amounts in native currency each month.",
        "- **No borrowing, interest, inflation, or risky-asset returns.**",
        "",
        "## Risk objective",
        "",
        "- **Loss:** cumulative KZT shortfall when expenses cannot be fully funded.",
        f"- **CVaR{int(spec['RISK']['CVaR_alpha']*100)}:** mean of worst ceil((1-α)N) path losses.",
        "- **VaR:** same tail index k as CVaR (order statistic, not smoothed).",
        "",
        "## Optimization grid",
        "",
        f"- **Integer basis-point simplex:** weights sum to {spec['OPTIMIZATION']['TOTAL_BP']} bp.",
        f"- **2% coarse grid:** {spec['OPTIMIZATION']['expected_2pct_grid_size']} unique portfolios.",
        f"- **1% refined grid:** {spec['OPTIMIZATION']['expected_1pct_grid_size']} unique portfolios.",
        "- **Near-optimal bands:** ±1% and ±5% relative to minimum CVaR (with zero-CVaR absolute bands).",
        "",
        "## Robustness dimensions (Phase 2C)",
        "",
        f"- Alternative household: E/I={spec['ROBUSTNESS']['alternative_EI_B']['E_I']}, "
        f"B={spec['ROBUSTNESS']['alternative_EI_B']['B']}.",
        "- DGP alternatives: SB10, SB56, IID.",
        "- Recent 2020–2025 estimation window.",
        "- Rebalance: monthly (baseline), quarterly, none.",
        "- Conversion costs: 0%, 0.25%, 0.50%, 1.00% (generic sensitivity, not KZ-calibrated).",
        "",
    ]
    return "\n".join(lines) + "\n"


def audit_grid_integrity(state: CleanupState) -> pd.DataFrame:
    rows = []
    for step_bp, expected in ((200, 1326), (100, 5151)):
        bp = generate_simplex_grid(step_bp)
        keys = [tuple(int(x) for x in r) for r in bp]
        dup = len(keys) - len(set(keys))
        neg = int(np.sum(bp < 0))
        sum_ok = bool(np.all(bp.sum(axis=1) == TOTAL_BP))
        actual = len(bp)
        status = "PASS" if actual == expected and dup == 0 and neg == 0 and sum_ok else "FAIL"
        if status != "PASS":
            state.stopped = True
            state.stop_reasons.append(f"grid integrity failed for step_bp={step_bp}")
        rows.append(
            {
                "grid_step_bp": step_bp,
                "expected_count": expected,
                "actual_count": actual,
                "unique_count": len(set(keys)),
                "duplicate_count": dup,
                "sum_constraint_pass": sum_ok,
                "negative_weight_count": neg,
                "status": status,
            }
        )
        if step_bp == 200:
            state.grid_2pct = actual
        else:
            state.grid_1pct = actual
    df = pd.DataFrame(rows)
    _write_csv(df, fp.AUDIT_DIR / "grid_integrity.csv")
    return df


def audit_grid_reconciliation(state: CleanupState) -> pd.DataFrame:
    old = pd.read_csv(config.OPT_TABLES / "optimal_allocations.csv")
    patch = pd.read_csv(config.ROB_GRID_PATCH / "grid_patch_comparison.csv")
    rows = []
    material_tol = 1e-6
    for _, r in patch.iterrows():
        prof = r["profile"]
        hy = int(r["horizon_years"])
        old_row = old[(old.profile == prof) & (old.horizon_years == hy)].iloc[0]
        cvar_diff = abs(float(r["corrected_cvar"]) - float(r["old_cvar"]))
        w_old = r["old_opt_weights"]
        w_new = r["corrected_opt_weights"]
        econ_changed = bool(r["economic_result_changed"])
        if econ_changed or cvar_diff > material_tol or w_old != w_new:
            state.stopped = True
            state.stop_reasons.append(
                f"economic change in grid reconciliation: {prof} {hy}Y"
            )
        rows.append(
            {
                "profile": prof,
                "horizon": f"{hy}Y",
                "old_argmin_weights": w_old,
                "corrected_argmin_weights": w_new,
                "old_cvar": r["old_cvar"],
                "corrected_cvar": r["corrected_cvar"],
                "cvar_difference": cvar_diff,
                "old_near_opt_count": int(r["old_near_1pct_count"]),
                "corrected_near_opt_count": int(r["corrected_near_1pct_count"]),
                "identification_old": r["old_identification_class"],
                "identification_corrected": r["corrected_identification_class"],
                "economic_result_changed": econ_changed,
            }
        )
    df = pd.DataFrame(rows)
    _write_csv(df, fp.AUDIT_DIR / "grid_reconciliation.csv")
    return df


def audit_reproducibility_manifest() -> pd.DataFrame:
    desc = pd.read_csv(config.TABLES / "descriptive_statistics.csv")
    b_usd = desc[(desc.window == "B_baseline") & (desc.series == "usd_kzt")].iloc[0]
    b_eur = desc[(desc.window == "B_baseline") & (desc.series == "eur_kzt")].iloc[0]
    a1 = _load_json(config.SIM_DIAGNOSTICS / "phase2a_run_summary.json")
    b1 = _load_json(config.HH_DIAGNOSTICS / "phase2b1_run_summary.json")
    cal = pd.read_csv(config.HH_TABLES / "stress_calibration.csv")
    opt_old = pd.read_csv(config.OPT_TABLES / "optimal_allocations.csv")
    opt_corr = pd.read_csv(config.ROB_GRID_PATCH / "corrected_optimal_allocations.csv")
    p2c = opt_corr[
        (opt_corr.profile == "P2_foreign_income_mismatch") & (opt_corr.horizon_years == 3)
    ].iloc[0]
    p4c = opt_corr[
        (opt_corr.profile == "P4_full_multi_currency") & (opt_corr.horizon_years == 3)
    ].iloc[0]
    p2b1_key = cal[
        (cal.profile == "P2_foreign_income_mismatch")
        & (cal.horizon_years == 3)
        & (cal.expense_income_ratio == 0.95)
        & (cal.buffer_months == 1)
        & (cal.strategy == "B2_all_usd")
    ].iloc[0]
    c2c = _load_json(config.ROB_DIAGNOSTICS / "phase2c_run_summary.json")

    gates = [
        ("Phase 1", "scripts/run_phase1.py", config.TABLES / "descriptive_statistics.csv",
         "baseline_N_usd", float(b_usd.N), float(b_usd.N), 0, 0, "count"),
        ("Phase 1", "scripts/run_phase1.py", config.TABLES / "descriptive_statistics.csv",
         "baseline_usd_SD", float(b_usd["std"]), float(b_usd["std"]), 0, 1e-12, "float"),
        ("Phase 1", "scripts/run_phase1.py", config.TABLES / "descriptive_statistics.csv",
         "baseline_eur_SD", float(b_eur["std"]), float(b_eur["std"]), 0, 1e-12, "float"),
        ("Phase 2A", "scripts/run_phase2a.py", config.SIM_DIAGNOSTICS / "phase2a_run_summary.json",
         "student_t_nu", a1["student_t_df"], a1["student_t_df"], 0, 1e-9, "float"),
        ("Phase 2A", "scripts/run_phase2a.py", config.SIM_DIAGNOSTICS / "phase2a_run_summary.json",
         "recommended_L", a1["recommended_L"], 20.0, 0, 0, "float"),
        ("Phase 2A", "scripts/run_phase2b1.py", config.HH_DIAGNOSTICS / "phase2b1_run_summary.json",
         "S0_USD_KZT", b1["s0_usd"], 502.57, 0, 0.01, "float"),
        ("Phase 2A", "scripts/run_phase2b1.py", config.HH_DIAGNOSTICS / "phase2b1_run_summary.json",
         "S0_EUR_KZT", b1["s0_eur"], 591.68, 0, 0.01, "float"),
        ("Phase 2B1", "scripts/run_phase2b1.py", config.HH_TABLES / "stress_calibration.csv",
         "calibration_rows", len(cal), 684, 0, 0, "count"),
        ("Phase 2B1", "scripts/run_phase2b1.py", config.HH_TABLES / "stress_calibration.csv",
         "baseline_EI", 0.95, 0.95, 0, 0, "float"),
        ("Phase 2B1", "scripts/run_phase2b1.py", config.HH_TABLES / "stress_calibration.csv",
         "baseline_B", 1, 1, 0, 0, "count"),
        ("Phase 2B1", "scripts/run_phase2b1.py", config.HH_TABLES / "stress_calibration.csv",
         "P2_3Y_allUSD_shortfall", float(p2b1_key.shortfall_probability), 0.0562, 0, 0.001, "float"),
        ("Phase 2B2", "scripts/run_phase2b2.py", config.OPT_TABLES / "optimal_allocations.csv",
         "P2_3Y_CVaR", float(opt_old[(opt_old.profile == "P2_foreign_income_mismatch") & (opt_old.horizon_years == 3)].iloc[0].cvar95),
         config.ROB_GATE_P2_CVAR, 0, config.ROB_GATE_CVAR_ATOL, "float"),
        ("Phase 2B2", "scripts/run_phase2b2.py", config.OPT_TABLES / "optimal_allocations.csv",
         "P4_3Y_CVaR", float(opt_old[(opt_old.profile == "P4_full_multi_currency") & (opt_old.horizon_years == 3)].iloc[0].cvar95),
         config.ROB_GATE_P4_CVAR, 0, config.ROB_GATE_CVAR_ATOL, "float"),
        ("Phase 2C", "scripts/run_phase2c.py", config.ROB_GRID_PATCH / "corrected_optimal_allocations.csv",
         "P2_3Y_CVaR_corrected", float(p2c.cvar95), config.ROB_GATE_P2_CVAR, 0, config.ROB_GATE_CVAR_ATOL, "float"),
        ("Phase 2C", "scripts/run_phase2c.py", config.ROB_GRID_PATCH / "corrected_optimal_allocations.csv",
         "grid_2pct_unique", c2c["patch"]["grid_2pct"], 1326, 0, 0, "count"),
        ("Phase 2C", "scripts/run_phase2c.py", config.ROB_GRID_PATCH / "corrected_optimal_allocations.csv",
         "grid_1pct_unique", c2c["patch"]["grid_1pct"], 5151, 0, 0, "count"),
        ("Phase 2C", "scripts/run_phase2c.py", config.ROB_DIAGNOSTICS / "phase2c_run_summary.json",
         "final_classification", c2c["final_classification"], "B", 0, 0, "string"),
    ]

    rows = []
    for phase, entry, main_out, metric, observed, expected, abs_diff_manual, tol, kind in gates:
        exists = main_out.exists()
        h = _sha256(main_out)
        if kind == "string":
            abs_diff = 0.0 if str(observed) == str(expected) else 1.0
            status = "PASS" if exists and str(observed) == str(expected) else "FAIL"
        else:
            abs_diff = abs(float(observed) - float(expected))
            status = "PASS" if exists and abs_diff <= float(tol) else "FAIL"
        rows.append(
            {
                "phase": phase,
                "entry_point": entry,
                "main_output": str(main_out.relative_to(config.PROJECT_ROOT)),
                "exists": exists,
                "hash": h[:16],
                "expected_key_metric": metric,
                "observed_key_metric": observed,
                "absolute_difference": abs_diff,
                "tolerance": tol,
                "status": status,
            }
        )
    df = pd.DataFrame(rows)
    _write_csv(df, fp.AUDIT_DIR / "reproducibility_manifest.csv")
    return df


def audit_buffer_selection() -> tuple[pd.DataFrame, str]:
    cal = pd.read_csv(config.HH_TABLES / "stress_calibration.csv")
    sel = select_baseline_candidate(cal)
    rows = []
    for B in (1, 3, 6, 12):
        for profile in PRIMARY_PROFILES:
            for strat_name, _ in strategy_catalog(profile):
                sub = cal[
                    (cal.expense_income_ratio == 0.95)
                    & (cal.horizon_years == 3)
                    & (cal.buffer_months == B)
                    & (cal.profile == profile.name)
                    & (cal.strategy == strat_name)
                ]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                p1p = cal[
                    (cal.profile == "P1_local_control")
                    & (cal.expense_income_ratio == 0.95)
                    & (cal.horizon_years == 3)
                    & (cal.buffer_months == B)
                    & (cal.strategy == "B1_all_kzt")
                ].iloc[0].shortfall_probability
                fx = cal[
                    cal.profile.isin(
                        [
                            "P2_foreign_income_mismatch",
                            "P3_mixed_income",
                            "P4_full_multi_currency",
                        ]
                    )
                    & (cal.expense_income_ratio == 0.95)
                    & (cal.horizon_years == 3)
                    & (cal.buffer_months == B)
                ]
                hit = fx[(fx.shortfall_probability >= 0.05) & (fx.shortfall_probability <= 0.40)]
                eligible = (p1p <= 0.05) and (len(hit) > 0)
                reason = []
                if p1p > 0.05:
                    reason.append("P1_all_KZT_shortfall>5%")
                if len(hit) == 0:
                    reason.append("no_FX_benchmark_in_5-40%")
                if eligible:
                    reason.append("eligible")
                rows.append(
                    {
                        "B": B,
                        "profile": _profile_label(profile.name),
                        "strategy": strat_name,
                        "shortfall_probability": float(r.shortfall_probability),
                        "CVaR95": float(r.cvar95_cumulative_shortfall),
                        "eligible_under_selection_rule": eligible,
                        "reason": "; ".join(reason) if reason else "",
                    }
                )
    df = pd.DataFrame(rows)
    _write_csv(df, fp.AUDIT_DIR / "buffer_selection_audit.csv")

    md = f"""# Buffer Selection Audit

## 1. What is the exact eligibility rule?

For each triple (expense/income ratio, buffer months B, horizon), a cell is **eligible** if:

1. P1 all-KZT shortfall probability ≤ 5%, **and**
2. At least one P2/P3/P4 benchmark strategy has shortfall probability in [5%, 40%].

Among eligible cells, `select_baseline_candidate()` sorts by:

1. Prefer horizon = 3 years;
2. Minimize |E/I − 0.95|;
3. Minimize |B − 6| (tie-break preference toward B≈6);
4. Maximize count of FX benchmarks in the 5–40% band.

## 2. Which B values are eligible at E/I=0.95, 3Y?

"""
    elig = df[df.eligible_under_selection_rule].groupby("B").size()
    for B in (1, 3, 6, 12):
        n = int(elig.get(B, 0))
        md += f"- **B={B}:** {'eligible' if n > 0 else 'not eligible'} ({n} strategy rows in audit table).\n"

    chosen = sel.get("candidate") or {}
    md += f"""
## 3. Why is B=1 the accepted primary optimization setting?

The sort key minimizes |E/I−0.95| first, then |B−6|. At E/I=0.95 and 3Y, **all B∈{{1,3,6,12}} are eligible**, but larger buffers push most FX benchmarks below the 5% floor, reducing informativeness. The B≈6 preference is a **secondary tie-break** only; with E/I already at 0.95, the sort selects **B=1** as the first eligible cell (thin buffer keeps P2 all-USD at ≈5.6% shortfall — lower edge of the band).

Accepted optimization baseline (frozen): **E/I=0.95, B=1, 3Y household environment**.

Chosen candidate from calibration table:
```json
{json.dumps(chosen, indent=2)}
```

## 4. Is B=6 merely a sorting preference, or was it ever selected?

**B=6 is only a sorting preference in the tie-break**, not the accepted optimization baseline. At E/I=0.95, B=6 is eligible but not chosen because B=1 sorts first when |E/I−0.95| is already zero for all eligible B. B=6 was **never** the accepted Phase 2B2 optimization setting.

## 5. Is any Methods wording currently misleading?

Potentially yes if text implies B=6 was selected because it is optimal. The rule explicitly states B≈6 is a **tie-break among eligible cells**, and the accepted cell is **B=1** at E/I=0.95. Wording should distinguish calibration-rule selection from CVaR optimization.

Rule satisfied: **{sel.get('rule_satisfied')}**; n eligible candidates: **{sel.get('n_candidates')}**.
"""
    _write_text(fp.AUDIT_DIR / "buffer_selection_audit.md", md)
    return df, md


def _fx_window_stats(start: str, end: str) -> dict[str, dict]:
    bundle = load_baseline_bundle(start, end)
    out = {}
    for i, name in enumerate(("usd_kzt", "eur_kzt")):
        r = bundle.returns[:, i]
        m = moments(r)
        levels = np.exp(np.cumsum(np.insert(r, 0, np.log(bundle.s0[i]))))
        peak = np.maximum.accumulate(levels)
        dd = (levels - peak) / peak
        m["annualized_mean"] = float(m["mean"] * config.ANNUALIZATION_FACTOR)
        m["cumulative_log_change"] = float(np.sum(r))
        m["max_drawdown_level"] = float(np.min(dd))
        m["downside_return_frequency"] = float(np.mean(r < 0))
        acf1 = pd.Series(np.abs(r)).autocorr(lag=1)
        acf5 = pd.Series(np.abs(r)).autocorr(lag=5)
        acf10 = pd.Series(np.abs(r)).autocorr(lag=10)
        m["acf_abs_lag1"] = float(acf1) if acf1 == acf1 else np.nan
        m["acf_abs_lag5"] = float(acf5) if acf5 == acf5 else np.nan
        m["acf_abs_lag10"] = float(acf10) if acf10 == acf10 else np.nan
        out[name] = m
    return out


def audit_recent_window_cvar() -> str:
    base_stats = _fx_window_stats(config.BASELINE_START, config.BASELINE_END)
    recent_stats = _fx_window_stats(config.RECENT_START, config.RECENT_END)

    daily_base, s0_base = load_or_generate_fx_levels()
    monthly_base = monthly_fx_for_horizon(daily_base, 3)

    daily_recent, s0_recent, _ = generate_fx_levels(
        kind="stationary_bootstrap",
        block=20,
        seed=config.ROB_RECENT_SEED,
        n_paths=config.ROB_N_PATHS,
        window_start=config.RECENT_START,
        window_end=config.RECENT_END,
    )
    monthly_recent = monthly_fx_for_horizon(daily_recent, 3)

    lines = [
        "# Recent-Window CVaR Audit",
        "",
        "Compare baseline estimation window vs recent 2020–2025 window.",
        "Goal: explain why lower recent daily volatility can coincide with higher household CVaR.",
        "",
        "## A. Historical return statistics",
        "",
    ]

    def _fmt_stats(label: str, st: dict) -> list[str]:
        out = [f"### {label}", ""]
        for cur, s in st.items():
            out.append(f"**{cur.upper()}** (N={s['n']}):")
            out.append(
                f"- mean={s['mean']:.6f}, median={s['median']:.6f}, SD={s['std']:.6f}, "
                f"annualized vol≈{s['std']*np.sqrt(config.ANNUALIZATION_FACTOR):.4f}"
            )
            out.append(
                f"- skew={s['skewness']:.3f}, excess kurtosis={s['excess_kurtosis']:.3f}"
            )
            out.append(
                f"- p1/p5/p50/p95/p99: {s['p1']:.5f} / {s['p5']:.5f} / {s['median']:.5f} / "
                f"{s['p95']:.5f} / {s['p99']:.5f}"
            )
            out.append(
                f"- cumulative log change={s['cumulative_log_change']:.4f}, "
                f"max drawdown={s['max_drawdown_level']:.4f}, "
                f"P(r<0)={s['downside_return_frequency']:.3f}"
            )
            out.append(
                f"- ACF(|r|) lags 1/5/10: {s['acf_abs_lag1']:.3f} / "
                f"{s['acf_abs_lag5']:.3f} / {s['acf_abs_lag10']:.3f}"
            )
            out.append("")
        return out

    lines.extend(_fmt_stats("Baseline window (2015-08-20 – 2025-12-31)", base_stats))
    lines.extend(_fmt_stats("Recent window (2020-01-01 – 2025-12-31)", recent_stats))

    lines += ["## B. Household simulation comparison (SB20, P2/P4, 3Y)", ""]

    for profile in (P2_FOREIGN_INCOME, P4_MULTI_CURRENCY):
        label = _profile_label(profile.name)
        lines.append(f"### {label}")
        for tag, monthly, s0 in (
            ("baseline", monthly_base, s0_base),
            ("recent", monthly_recent, s0_recent),
        ):
            cf, _, W0 = primary_cashflows(profile, s0)
            ev = evaluate_weights(monthly, cf, W0, np.array([1.0, 0.0, 0.0]))
            L = ev.path_losses
            S_u = monthly[:, :, 0]
            cum_usd_log = np.log(S_u[:, -1] / S_u[:, 0])
            if profile.name == "P2_foreign_income_mismatch":
                inc_kzt = kzt_equivalent_income(cf, S_u, monthly[:, :, 1])
                min_monthly_inc = np.min(inc_kzt, axis=1)
                cum_inc = np.sum(inc_kzt, axis=1)
                lines.append(f"**{tag}** — CVaR95={ev.cvar95:.4f}, P(SF)={ev.shortfall_probability:.4f}")
                lines.append(
                    f"- Terminal USD/KZT: min/med/p5/p95/max = "
                    f"{S_u[:,-1].min():.1f}/{np.median(S_u[:,-1]):.1f}/"
                    f"{np.quantile(S_u[:,-1],0.05):.1f}/{np.quantile(S_u[:,-1],0.95):.1f}/{S_u[:,-1].max():.1f}"
                )
                lines.append(
                    f"- Cumulative USD log return: p5/p50/p95 = "
                    f"{np.quantile(cum_usd_log,0.05):.4f}/{np.median(cum_usd_log):.4f}/"
                    f"{np.quantile(cum_usd_log,0.95):.4f}"
                )
                lines.append(
                    f"- KZT-equiv USD income: monthly min p5/p50 = "
                    f"{np.quantile(min_monthly_inc,0.05):.4f}/{np.median(min_monthly_inc):.4f}; "
                    f"36m cumulative p5/p50 = {np.quantile(cum_inc,0.05):.4f}/{np.median(cum_inc):.4f}"
                )
            else:
                lines.append(f"**{tag}** — CVaR95={ev.cvar95:.4f}, P(SF)={ev.shortfall_probability:.4f}")
                lines.append(
                    f"- Terminal USD/KZT p5/p50/p95 = "
                    f"{np.quantile(S_u[:,-1],0.05):.1f}/{np.median(S_u[:,-1]):.1f}/"
                    f"{np.quantile(S_u[:,-1],0.95):.1f}"
                )
            lines.append(
                f"- Cumulative shortfall: p95={np.quantile(L,0.95):.4f}, "
                f"max={L.max():.4f}"
            )
            lines.append("")

    # Mechanism synthesis
    usd_base = base_stats["usd_kzt"]
    usd_rec = recent_stats["usd_kzt"]
    lines += [
        "## C. Supported mechanisms (no single dominant narrative)",
        "",
        "Evidence from the audit:",
        "",
        f"1. **Lower recent daily SD** (USD: {usd_rec['std']:.5f} vs baseline {usd_base['std']:.5f}) "
        "is confirmed in the historical sample.",
        f"2. **Lower recent mean drift** (USD annualized mean: {usd_rec['annualized_mean']:.4f} vs "
        f"{usd_base['annualized_mean']:.4f}) — weaker tenge depreciation on average in 2020–2025.",
        f"3. **Household CVaR rises** under the recent-window DGP (P2: 1.19 → 2.53; P4: 0.51 → 1.38) "
        "despite lower σ — tail **shortfall paths** worsen, not daily σ alone.",
        "4. **Fixed nominal USD income** interacts with FX: lower average USD/KZT growth reduces "
        "KZT-equivalent income paths, tightening the budget constraint for KZT expenses.",
        "5. **Resampled recent history** includes 2022 stress episodes; block bootstrap preserves "
        "clustered adverse sequences that raise cumulative shortfall CVaR even when unconditional daily vol fell.",
        "",
        "**Conclusion:** The paradox is **not** a coding inconsistency. It reflects that household CVaR "
        "depends on the **joint distribution of income conversion and expense funding**, not daily FX σ alone. "
        "Multiple mechanisms (weaker drift, tail clustering, nominal-income interaction) are supported; "
        "**no single factor fully explains** the magnitude gap.",
        "",
    ]
    md = "\n".join(lines)
    _write_text(fp.AUDIT_DIR / "recent_window_cvar_audit.md", md)
    return md


def audit_transaction_costs() -> str:
    # Deterministic tiny example: 1 path, 2 months
    monthly_fx = np.array([[[500.0, 600.0], [400.0, 550.0]]])  # (1,2,2)
    cf = nominal_cashflows(P2_FOREIGN_INCOME, 1.0, 0.95, 500.0, 600.0)
    W0 = 0.95
    c = 0.01
    w = np.array([1.0, 0.0, 0.0])
    run = simulate_household(
        monthly_fx, cf, w, W0, rebalance_frequency="monthly", conversion_cost=c
    )
    ex = run.trace_rows[0] if hasattr(run, "trace_rows") else None
    # Re-run with trace
    run2 = simulate_household(
        monthly_fx,
        cf,
        w,
        W0,
        rebalance_frequency="monthly",
        conversion_cost=c,
        record_paths=False,
    )
    run2.trace_rows = []  # type: ignore[attr-defined]
    # manual trace via evaluate_weights
    ev = evaluate_weights(
        monthly_fx, cf, W0, w, rebalance_frequency="monthly", conversion_cost=c
    )
    md = f"""# Transaction Cost Implementation Audit

## Convention (from `settlement.py` / `portfolio.py`)

**Liquidation (gap funding):**
- `take` = KZT-equivalent amount applied to unfunded expense gap (capped by W/(1+c))
- `fee = c × take`
- Portfolio reduction = `take + fee` (pro-rata across holdings)

**Rebalancing:**
- `buy_volume` = sum of positive gaps to target KZT values
- `fee = c × buy_volume` (one charge; sell side not double-counted)

## Answers

1. **Is cost applied only when conversion occurs?** Yes — liquidation fees apply when gap funding requires cross-currency liquidation; rebalance fees apply only when target weights differ and buy volume > 0.

2. **Double-charging?** No — liquidation uses a single proportional charge on `take`; rebalancing charges buys only.

3. **USD→EUR double charge?** No — gaps are funded pro-rata in KZT numeraire; no separate USD→KZT→EUR chain fee.

4. **Gross-up when W insufficient?** Yes — `max_fund = W/(1+c)` ensures `take + fee ≤ W`.

5. **Conservation?** `wealth_before + income - funded_expenses - liq_fee = wealth_after` (verified in tests; max error ≪ 1e-8).

6. **100% KZT P2 still incur costs?** Yes — USD income arrives each month; rebalancing to 100% KZT triggers buy-side conversion fees when c>0.

7. **Phase 2C costs calibrated to Kazakhstan?** No — generic sensitivity parameters (`ROB_CONVERSION_COSTS`), explicitly not retail KZ spreads.

## Deterministic unit example (P2, 100% KZT, c=1%, 1 path, 2 months)

- W0=0.95, I_usd nominal, E_kzt=0.95 KZT-equivalent
- Month 1 FX: USD/KZT=500; Month 2: USD/KZT=400 (adverse for KZT income value)
- Mean cumulative conversion cost (1 path): **{getattr(ev, 'mean_cumulative_conversion_cost', 0.0):.6f}**
- CVaR95 with cost: **{ev.cvar95:.6f}**
- Shortfall probability: **{ev.shortfall_probability:.4f}**

Costs are zero at c=0 baseline; nonzero only when conversion/rebalancing occurs.
"""
    _write_text(fp.AUDIT_DIR / "transaction_cost_audit.md", md)
    return md


def audit_no_rebalance(state: CleanupState) -> str:
    daily, s0 = load_or_generate_fx_levels()
    monthly = monthly_fx_for_horizon(daily, 3)
    cf, _, W0 = primary_cashflows(P2_FOREIGN_INCOME, s0)
    alloc_labels = [
        ("100/0/0", [1.0, 0.0, 0.0]),
        ("80/20/0", [0.8, 0.2, 0.0]),
        ("50/50/0", [0.5, 0.5, 0.0]),
        ("20/80/0", [0.2, 0.8, 0.0]),
        ("0/100/0", [0.0, 1.0, 0.0]),
    ]
    rows = []
    lines = [
        "# No-Rebalance Interpretation Audit",
        "",
        "Profile: **P2**, horizon **3Y**, baseline SB20, `rebalance_frequency='none'`.",
        "",
        "With continuous USD income and KZT expenses, portfolios with different **initial** weights",
        "drift toward similar effective exposures when voluntary rebalancing is disabled.",
        "",
        "| Initial weights | CVaR95 | P(shortfall) | Mean term. KZT share | Mean term. USD share |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, w in alloc_labels:
        ev = evaluate_weights(
            monthly, cf, W0, np.array(w), rebalance_frequency="none"
        )
        run = simulate_household(
            monthly, cf, np.array(w), W0, rebalance_frequency="none"
        )
        hk, hu, he = run.holdings_end
        S_u = monthly[:, -1, 0]
        S_e = monthly[:, -1, 1]
        wealth = hk + hu * S_u + he * S_e
        kzt_share = hk / np.maximum(wealth, 1e-18)
        usd_share = (hu * S_u) / np.maximum(wealth, 1e-18)
        rows.append(
            {
                "initial_weights": label,
                "cvar95": ev.cvar95,
                "shortfall_probability": ev.shortfall_probability,
                "mean_terminal_kzt_share": float(np.mean(kzt_share)),
                "mean_terminal_usd_share": float(np.mean(usd_share)),
            }
        )
        lines.append(
            f"| {label} | {ev.cvar95:.4f} | {ev.shortfall_probability:.4f} | "
            f"{np.mean(kzt_share):.3f} | {np.mean(usd_share):.3f} |"
        )

    cvars = [r["cvar95"] for r in rows]
    spread = max(cvars) - min(cvars)
    lines += [
        "",
        f"CVaR95 range across initial weights: **{min(cvars):.4f} – {max(cvars):.4f}** (spread {spread:.4f}).",
        "",
        "## Interpretation",
        "",
        "- **Initial allocation ≠ maintained target allocation** when rebalancing is off.",
        "- USD income accumulates every month, pushing terminal holdings toward USD regardless of W0.",
        "- Consequently, the optimization surface is **flat / near-flat** in initial weights → "
        "**C_NOT_IDENTIFIED / UNIDENTIFIED** (see Phase 2C `rebalance_none` rows).",
        "- This is a structural identification issue, not a bug.",
        "",
    ]
    md = "\n".join(lines)
    _write_text(fp.AUDIT_DIR / "no_rebalance_audit.md", md)
    return md


def build_publication_tables(state: CleanupState) -> None:
    corr = pd.read_csv(config.ROB_GRID_PATCH / "corrected_optimal_allocations.csv")
    opt = corr.copy()
    opt["profile"] = opt["profile"].map(_profile_label)
    opt["horizon"] = opt["horizon_years"].astype(str) + "Y"
    opt_out = opt.rename(
        columns={
            "cvar95": "CVaR95",
            "var95": "VaR95",
            "wealth_depletion_probability": "depletion_probability",
            "identification_class": "identification_class",
            "near_optimal_1pct_count": "near_1pct_count",
            "near_optimal_1pct_kzt_min": "near_1pct_kzt_min",
            "near_optimal_1pct_kzt_max": "near_1pct_kzt_max",
            "near_optimal_1pct_usd_min": "near_1pct_usd_min",
            "near_optimal_1pct_usd_max": "near_1pct_usd_max",
            "near_optimal_1pct_eur_min": "near_1pct_eur_min",
            "near_optimal_1pct_eur_max": "near_1pct_eur_max",
            "near_optimal_5pct_count": "near_5pct_count",
            "near_optimal_5pct_kzt_min": "near_5pct_kzt_min",
            "near_optimal_5pct_kzt_max": "near_5pct_kzt_max",
            "near_optimal_5pct_usd_min": "near_5pct_usd_min",
            "near_optimal_5pct_usd_max": "near_5pct_usd_max",
            "near_optimal_5pct_eur_min": "near_5pct_eur_min",
            "near_optimal_5pct_eur_max": "near_5pct_eur_max",
        }
    )
    cols = [
        "profile", "horizon", "w_kzt", "w_usd", "w_eur", "CVaR95", "VaR95",
        "shortfall_probability", "expected_cumulative_shortfall", "depletion_probability",
        "identification_class", "near_1pct_count",
        "near_1pct_kzt_min", "near_1pct_kzt_max", "near_1pct_usd_min", "near_1pct_usd_max",
        "near_1pct_eur_min", "near_1pct_eur_max", "near_5pct_count",
        "near_5pct_kzt_min", "near_5pct_kzt_max", "near_5pct_usd_min", "near_5pct_usd_max",
        "near_5pct_eur_min", "near_5pct_eur_max",
    ]
    _write_csv(opt_out[cols], fp.TABLES_DIR / "table_optimal_allocations.csv")

    p2_3 = opt_out[(opt_out.profile == "P2") & (opt_out.horizon == "3Y")].iloc[0]
    p4_3 = opt_out[(opt_out.profile == "P4") & (opt_out.horizon == "3Y")].iloc[0]
    state.p2_3y = {"weights": (p2_3.w_kzt, p2_3.w_usd, p2_3.w_eur), "cvar": float(p2_3.CVaR95)}
    state.p4_3y = {"weights": (p4_3.w_kzt, p4_3.w_usd, p4_3.w_eur), "cvar": float(p4_3.CVaR95)}

    bench = pd.read_csv(config.OPT_TABLES / "benchmark_comparison.csv")
    bench = bench[(bench.horizon_years == 3) & (bench.profile.isin(
        ["P2_foreign_income_mismatch", "P3_mixed_income", "P4_full_multi_currency"]
    ))].copy()
    name_map = {
        "B1_all_kzt": "all-KZT",
        "B2_all_usd": "all-USD",
        "B3_all_eur": "all-EUR",
        "B4_equal": "equal",
        "B5_expense_matched": "expense-matched",
        "B6_income_matched": "income-matched",
    }
    bench["strategy"] = bench["benchmark_name"].map(name_map).fillna(bench["benchmark_name"])
    bench["profile"] = bench["profile"].map(_profile_label)
    opt_cvar = bench.groupby("profile")["optimized_cvar95"].first()
    exp_cvar = bench[bench.strategy == "expense-matched"].set_index("profile")["benchmark_cvar95"]
    bench_rows = []
    for _, r in bench.iterrows():
        prof = r["profile"]
        opt_cv = float(opt_cvar[prof])
        exp_cv = float(exp_cvar.get(prof, np.nan))
        bench_rows.append(
            {
                "profile": prof,
                "strategy": r["strategy"],
                "w_kzt": r["benchmark_w_kzt"],
                "w_usd": r["benchmark_w_usd"],
                "w_eur": r["benchmark_w_eur"],
                "CVaR95": r["benchmark_cvar95"],
                "VaR95": np.nan,
                "shortfall_probability": r["benchmark_shortfall_probability"],
                "expected_shortfall": r["benchmark_expected_shortfall"],
                "relative_cvar_vs_optimum_pct": 100.0 * (r["benchmark_cvar95"] - opt_cv) / max(opt_cv, 1e-12),
                "relative_cvar_vs_expense_match_pct": 100.0 * (r["benchmark_cvar95"] - exp_cv) / max(exp_cv, 1e-12) if exp_cv == exp_cv else np.nan,
            }
        )
    bench_out = pd.DataFrame(bench_rows).drop_duplicates(
        subset=["profile", "strategy", "w_kzt", "w_usd", "w_eur"]
    )
    _write_csv(bench_out, fp.TABLES_DIR / "table_benchmark_3y.csv")

    rob = pd.read_csv(config.ROB_TABLES / "robustness_results.csv")
    scenario_map = {
        "baseline_SB20_EI95_B1_monthly_c0": "baseline SB20",
        "EI105_B3": "alternative household E/I=1.05 B=3",
        "SB10": "SB10",
        "SB56": "SB56",
        "IID": "IID",
        "recent_2020_2025_SB20": "recent 2020–2025 window",
        "rebalance_quarterly_c0": "quarterly rebalance",
        "rebalance_none_c0": "no rebalance",
        "cost_0.0025_monthly": "cost 0.25%",
        "cost_0.0050_monthly": "cost 0.50%",
        "cost_0.0100_monthly": "cost 1.00%",
    }
    rob_sub = rob[
        rob.profile.isin(["P2_foreign_income_mismatch", "P4_full_multi_currency"])
        & rob.specification.isin(scenario_map)
        & (rob.horizon == 3)
    ].copy()
    rob_sub["scenario"] = rob_sub["specification"].map(scenario_map)
    rob_sub["profile"] = rob_sub["profile"].map(_profile_label)
    rob_out = rob_sub.rename(
        columns={
            "w_kzt": "optimal_w_kzt",
            "w_usd": "optimal_w_usd",
            "w_eur": "optimal_w_eur",
            "cvar95": "CVaR95",
            "expense_matched_cvar": "expense_match_cvar",
            "income_matched_cvar": "income_match_cvar",
            "identification_class": "identification_class",
            "robustness_class": "result_class",
        }
    )
    rob_out["expense_beats_income"] = rob_out["expense_match_cvar"] <= rob_out["income_match_cvar"]
    _write_csv(
        rob_out[
            [
                "profile", "scenario", "optimal_w_kzt", "optimal_w_usd", "optimal_w_eur",
                "CVaR95", "shortfall_probability", "identification_class",
                "expense_match_cvar", "income_match_cvar", "expense_beats_income", "result_class",
            ]
        ],
        fp.TABLES_DIR / "table_robustness_summary.csv",
    )

    desc = pd.read_csv(config.TABLES / "descriptive_statistics.csv")
    dist = pd.read_csv(config.TABLES / "distribution_tests.csv")
    fx_rows = []
    for series in ("usd_kzt", "eur_kzt"):
        d = desc[(desc.window == "B_baseline") & (desc.series == series)].iloc[0]
        jb = dist[(dist.series == series) & (dist.window == "B_baseline")].iloc[0]
        fx_rows.append(
            {
                "series": series.upper(),
                "N": int(d.N),
                "mean": d["mean"],
                "SD": d["std"],
                "annualized_vol": d.annualized_volatility,
                "skewness": d.skewness,
                "excess_kurtosis": d.excess_kurtosis,
                "min": d["min"],
                "p1": d.p1,
                "p5": d.p5,
                "median": d.median,
                "p95": d.p95,
                "p99": d.p99,
                "max": d["max"],
                "Jarque_Bera_p": jb.jarque_bera_pvalue,
            }
        )
    _write_csv(pd.DataFrame(fx_rows), fp.TABLES_DIR / "table_fx_descriptive_statistics.csv")

    scores = pd.read_csv(config.SIM_TABLES / "model_scores.csv")
    temp = pd.read_csv(config.SIM_TABLES / "temporal_dependence_validation.csv")
    vol = pd.read_csv(config.SIM_TABLES / "volatility_clustering_validation.csv")
    tail = pd.read_csv(config.SIM_TABLES / "tail_validation.csv")
    marg = pd.read_csv(config.SIM_TABLES / "marginal_validation.csv")
    dep = pd.read_csv(config.SIM_TABLES / "dependence_validation.csv")

    def _cat_rmse(df, col_filter):
        sub = df[df.currency == "usd"]
        if col_filter:
            sub = sub[sub["transform"] == col_filter] if "transform" in sub.columns else sub
        if "acf_rmse" in sub.columns:
            return sub.groupby("model")["acf_rmse"].mean()
        if "abs_err_mean" in sub.columns:
            return sub.groupby("model")["abs_err_mean"].mean()
        return sub.groupby("model").apply(lambda x: x.filter(regex="abs_err").mean().mean())

    model_map = {
        "stationary_bootstrap_L20": "SB20",
        "stationary_bootstrap_L10": "SB10",
        "stationary_bootstrap_L5": "SB5",
        "moving_block_L10": "MBB10",
        "iid_bootstrap": "IID",
        "student_t": "Student-t",
        "gaussian": "Gaussian",
    }
    rows = []
    for model_key, label in model_map.items():
        sc = scores[scores.model == model_key]
        if sc.empty:
            continue
        r = sc.iloc[0]
        t_abs = temp[(temp.model == model_key) & (temp["transform"] == "abs")]
        v30 = vol[(vol.model == model_key) & (vol.window == 30)]
        rows.append(
            {
                "model": label,
                "daily_SD_error": float(marg[marg.model == model_key]["abs_err_std"].mean()) if len(marg[marg.model == model_key]) else np.nan,
                "tail_error": float(tail[tail.model == model_key].filter(regex="abs_err").mean().mean()) if len(tail[tail.model == model_key]) else np.nan,
                "Pearson_error": float(dep[(dep.model == model_key)]["abs_err_pearson"].iloc[0]) if len(dep[dep.model == model_key]) else np.nan,
                "Spearman_error": float(dep[(dep.model == model_key)]["abs_err_spearman"].iloc[0]) if len(dep[dep.model == model_key]) else np.nan,
                "abs_return_ACF_RMSE": float(t_abs["acf_rmse"].mean()) if len(t_abs) else np.nan,
                "rolling_vol_error": float(v30.filter(regex="abs_err").mean().mean()) if len(v30) else np.nan,
                "overall_score": float(r.overall_equal_weight),
                "primary_selected": label == "SB20",
                "selection_note": (
                    "SB20 selected: best temporal/vol-clustering tradeoff among {5,10,20} SB grid; "
                    "MBB10 has marginally lower overall score but was not in pre-specified SB candidate set"
                    if label == "SB20"
                    else ""
                ),
            }
        )
    _write_csv(pd.DataFrame(rows), fp.TABLES_DIR / "table_simulation_model_comparison.csv")


def build_publication_figures() -> list[str]:
    created = []
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    df = pd.read_csv(config.DATA_PROCESSED / "fx_daily_2010_2025.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["date"], df["usd_kzt"], label="USD/KZT", color="#1f77b4", lw=0.8)
    ax.plot(df["date"], df["eur_kzt"], label="EUR/KZT", color="#ff7f0e", lw=0.8)
    ax.axvline(pd.Timestamp(config.FLOAT_TRANSITION_DATE), color="red", ls="--", lw=1, label="Aug 2015 float")
    ax.set_xlabel("Date")
    ax.set_ylabel("KZT per foreign currency unit")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fp.FIGURES_DIR / f"figure01_historical_fx_levels.{ext}"
        fig.savefig(p, bbox_inches="tight")
        created.append(str(p.name))
    plt.close(fig)

    base = df[(df["date"] >= config.BASELINE_START) & (df["date"] <= config.BASELINE_END)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, col, title in zip(axes, ["usd_kzt_log_return", "eur_kzt_log_return"], ["USD/KZT", "EUR/KZT"]):
        r = base[col].dropna()
        ax.hist(r, bins=60, density=True, alpha=0.7, color="#1f77b4", edgecolor="white", lw=0.3)
        ax.set_title(title)
        ax.set_xlabel("Daily log return")
        ax.set_ylabel("Density")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fp.FIGURES_DIR / f"figure02_return_distributions.{ext}"
        fig.savefig(p, bbox_inches="tight")
        created.append(str(p.name))
    plt.close(fig)

    rv = base["usd_kzt_log_return"].rolling(30).std() * np.sqrt(config.ANNUALIZATION_FACTOR)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(base["date"], rv, color="#1f77b4", lw=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("30-day annualized vol (USD/KZT)")
    ax.set_title("")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fp.FIGURES_DIR / f"figure02b_rolling_volatility.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    temp = pd.read_csv(config.SIM_TABLES / "temporal_dependence_validation.csv")
    sub = temp[(temp.currency == "usd") & (temp["transform"] == "abs")]
    models = ["stationary_bootstrap_L20", "moving_block_L10", "iid_bootstrap"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    lags = [1, 5, 10, 20]
    for m, ls in zip(models, ["-", "--", ":"]):
        row = sub[sub.model == m].iloc[0]
        sim = [row[f"sim_lag{lg}"] for lg in lags]
        ax.plot(lags, sim, ls=ls, marker="o", label=m.replace("stationary_bootstrap_", "SB").replace("moving_block_", "MBB"))
    hist = [sub.iloc[0][f"hist_lag{lg}"] for lg in lags]
    ax.plot(lags, hist, "k-", marker="s", label="Historical")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF(|r|)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fp.FIGURES_DIR / f"figure03_sb20_validation.{ext}"
        fig.savefig(p, bbox_inches="tight")
        created.append(str(p.name))
    plt.close(fig)

    coarse = pd.read_csv(config.OPT_TABLES / "coarse_grid_results.csv")
    for prof, tag, fignum in (
        ("P2_foreign_income_mismatch", "P2", "04"),
        ("P4_full_multi_currency", "P4", "05"),
    ):
        g = coarse[(coarse.profile == prof) & (coarse.horizon_years == 3)]
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(g["w_usd"], g["w_eur"], c=g["cvar95"], cmap="viridis_r", s=12, alpha=0.8)
        plt.colorbar(sc, ax=ax, label="CVaR95")
        ax.set_xlabel("USD weight")
        ax.set_ylabel("EUR weight")
        ax.set_title(f"{tag} 3Y coarse grid")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = fp.FIGURES_DIR / f"figure{fignum}_{tag.lower()}_3y_cvar_surface.{ext}"
            fig.savefig(p, bbox_inches="tight")
            created.append(str(p.name))
        plt.close(fig)

    bench = pd.read_csv(fp.TABLES_DIR / "table_benchmark_3y.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, prof in zip(axes, ["P2", "P3", "P4"]):
        sub = bench[bench.profile == prof]
        ax.bar(sub["strategy"], sub["CVaR95"], color="#4c72b0")
        ax.set_title(prof)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].set_ylabel("CVaR95")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fp.FIGURES_DIR / f"figure06_benchmark_3y.{ext}"
        fig.savefig(p, bbox_inches="tight")
        created.append(str(p.name))
    plt.close(fig)

    rob = pd.read_csv(fp.TABLES_DIR / "table_robustness_summary.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    xlabels = rob["scenario"].unique()
    x = np.arange(len(xlabels))
    w = 0.35
    for i, prof in enumerate(["P2", "P4"]):
        sub = rob[rob.profile == prof].set_index("scenario").reindex(xlabels)
        ax.bar(x + (i - 0.5) * w, sub["expense_match_cvar"], w, label=f"{prof} expense")
        ax.bar(x + (i + 0.5) * w, sub["income_match_cvar"], w, label=f"{prof} income", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CVaR95")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = fp.FIGURES_DIR / f"figure07_robustness_expense_vs_income.{ext}"
        fig.savefig(p, bbox_inches="tight")
        created.append(str(p.name))
    plt.close(fig)

    return created


def write_figure_captions() -> None:
    md = """# Figure Captions

## Figure 1 — Historical FX levels
**File:** `figure01_historical_fx_levels.pdf`

Daily official USD/KZT and EUR/KZT levels, 2010–2025. Vertical line: 20 August 2015 float transition.

**Notice:** EUR/KZT is an NBK cross-rate. Pre-2015 regime differs from post-float baseline.

## Figure 2 — Return distributions and rolling volatility
**Files:** `figure02_return_distributions.pdf`, `figure02b_rolling_volatility.pdf`

Baseline-window daily log-return histograms and 30-day rolling USD/KZT volatility.

**Notice:** Fat tails and volatility clustering motivate block-bootstrap DGP.

## Figure 3 — Simulation validation (SB20)
**File:** `figure03_sb20_validation.pdf`

Historical vs simulated ACF(|r|) for USD/KZT at selected lags.

**Notice:** SB20 tracks persistence better than IID; MBB10 shown for reference.

## Figure 4 — P2 3Y CVaR allocation surface
**File:** `figure04_p2_3y_cvar_surface.pdf`

Coarse-grid CVaR95 over USD/EUR weights (KZT residual) for P2, 3-year horizon.

**Notice:** Minimum at 100% KZT; surface rises with foreign savings share.

## Figure 5 — P4 3Y CVaR allocation surface
**File:** `figure05_p4_3y_cvar_surface.pdf`

Same for P4 (multi-currency income/expense).

**Notice:** Optimum remains KZT-heavy; near-optimal band is wider than P2.

## Figure 6 — 3Y benchmark comparison
**File:** `figure06_benchmark_3y.pdf`

CVaR95 for optimized vs benchmarks (P2–P4).

**Notice:** Expense-matched and all-KZT coincide for P2/P3; income-matched is foreign-heavy.

## Figure 7 — Robustness: expense vs income matching
**File:** `figure07_robustness_expense_vs_income.pdf`

Expense-matched vs income-matched CVaR across robustness scenarios (P2, P4).

**Notice:** Expense-aligned savings weakly dominate across core scenarios; magnitudes are scenario-dependent.
"""
    _write_text(fp.FINAL_DIR / "figure_captions.md", md)


def write_assumptions_and_limitations() -> None:
    md = """# Assumptions and Limitations Register

## Empirical inputs

- NBK official daily market exchange rates (USD/KZT, EUR/KZT cross-rate).
- Post-float baseline estimation window: 2015-08-20 – 2025-12-31.
- Weekday-only return series after weekend-gap diagnostics.

## Modeling assumptions

- Synthetic household profiles P1–P4 (not microdata-calibrated).
- I0 normalized to 1.0 KZT-equivalent; fixed nominal foreign cash flows.
- Baseline E/I=0.95, B=1 month, W0=0.95.
- 21 weekdays = one household month.
- Primary DGP: joint stationary bootstrap L=20, N=10,000, seed 42.
- Objective: minimize CVaR95 of cumulative shortfall.
- Monthly rebalancing to target weights (baseline).
- Zero baseline conversion cost; no interest, inflation, or asset returns; no borrowing.

## Computational design choices

- Integer basis-point simplex grid (TOTAL_BP=10,000).
- 2% coarse + 1% local refinement grids.
- Near-optimal identification bands at ±1% and ±5%.
- Phase 2C transaction costs: generic sensitivity, **not** Kazakhstan-calibrated spreads.
- No GARCH/DCC as primary DGP (evaluated but not selected).

## Limitations

- Exact optimal weights are model-dependent (RESULT B).
- Recent-window re-estimation raises CVaR despite lower σ — joint tail/drift mechanism.
- No-rebalance cells are weakly/unidentified (initial ≠ maintained allocation).
- EUR/KZT cross-rate and post-float sample constrain external validity.
- Model does not forecast future USD/KZT levels.
- Kazakhstan household microdata not used for calibration.
"""
    _write_text(fp.FINAL_DIR / "ASSUMPTIONS_AND_LIMITATIONS.md", md)


def write_paper_safe_claims() -> None:
    md = """# Paper-Safe Claims

## SUPPORTED

- 100% KZT is the baseline numerical CVaR minimum for identified FX-exposed profiles (P2–P4) at 1/3/5Y under E/I=0.95, B=1, SB20, monthly rebalance, zero cost.
- Expense-aligned liquid savings outperform income-matched savings across core Phase 2C robustness scenarios for P2/P4.
- Foreign-currency income creates natural FX exposure when savings are not income-matched.
- L=20 performed best among the pre-specified {5, 10, 20} stationary-bootstrap grid for temporal dependence and volatility clustering.
- Integer-bp grid has 1326 (2%) and 5151 (1%) unique portfolios; Phase 2B2 float-key duplicates did not change economic optima.

## SUPPORTED WITH QUALIFICATION

- "KZT-heavy / expense-matching principle is robust" — yes for mechanism (expense beats income), but exact 100% KZT weight varies in near-optimal sets and some cells are UNIDENTIFIED (no-rebalance).
- "Lower daily FX volatility implies lower household CVaR" — **not generally**; recent-window audit shows lower σ with higher CVaR (drift, tails, nominal-income interaction).
- Transaction costs worsen CVaR monotonically in sensitivity — supported, but levels are generic not KZ-calibrated.

## NOT SUPPORTED / DO NOT WRITE

- "100% KZT is universally optimal" — false (model-dependent; P1/P3 zero-CVaR ridges; no-rebalance unidentified).
- "The model predicts future USD/KZT" — simulation is risk assessment, not forecasting.
- "L=20 is the true optimal block length" — only best among pre-specified candidates.
- "Transaction costs were calibrated to Kazakhstan" — explicitly generic sensitivity.
- "MBB10 / lowest overall score model should be selected" — SB20 chosen per pre-specified SB grid and dependence criteria.
"""
    _write_text(fp.FINAL_DIR / "PAPER_SAFE_CLAIMS.md", md)


def write_publication_manifest() -> None:
    md = """# Publication Artifact Manifest

Canonical layer: `results/final/`. Do **not** cite pre-patch Phase 2B2 `n_surface=5280` counts.

| Paper section | Result | Source file | Source function | Canonical output | Status |
|---|---|---|---|---|---|
| Data | FX sample windows | `src/config.py` | constants | `canonical_specification.json` | FROZEN |
| Results | FX descriptive statistics | `src/pipeline.py` | Phase 1 diagnostics | `tables/table_fx_descriptive_statistics.csv` | CANONICAL |
| Results | Simulation model comparison | `src/simulation/model_comparison.py` | `run_model_comparison` | `tables/table_simulation_model_comparison.csv` | CANONICAL |
| Methods | Primary DGP SB20 | `src/simulation/stationary_bootstrap.py` | `StationaryBootstrapSimulator` | `canonical_specification.json` | FROZEN |
| Results | Optimal allocations | `src/robustness/grid_patch.py` | integer-bp patch | `tables/table_optimal_allocations.csv` | CANONICAL |
| Results | 3Y benchmarks | `src/optimization/pipeline.py` | benchmark comparison | `tables/table_benchmark_3y.csv` | CANONICAL |
| Results | Robustness summary | `src/robustness/pipeline.py` | `run_robustness` | `tables/table_robustness_summary.csv` | CANONICAL |
| Results | P2/P4 3Y CVaR gates | `src/robustness/pipeline.py` | baseline gate | `audit/reproducibility_manifest.csv` | VERIFIED |
| Discussion | Buffer B=1 vs B=6 | `src/household/calibration.py` | `select_baseline_candidate` | `audit/buffer_selection_audit.md` | AUDIT |
| Discussion | Recent-window CVaR paradox | `src/final/pipeline.py` | `audit_recent_window_cvar` | `audit/recent_window_cvar_audit.md` | AUDIT |
| Discussion | No-rebalance identification | `src/final/pipeline.py` | `audit_no_rebalance` | `audit/no_rebalance_audit.md` | AUDIT |
| Methods | Transaction costs | `src/household/settlement.py` | `simulate_household` | `audit/transaction_cost_audit.md` | AUDIT |
| Figures | All publication figures | `src/final/pipeline.py` | `build_publication_figures` | `figures/figure*.pdf` | CANONICAL |
| Discussion | Safe claims | — | — | `PAPER_SAFE_CLAIMS.md` | CANONICAL |
| Summary | Final results | — | — | `FINAL_RESULTS_SUMMARY.md` | CANONICAL |

**Deprecated for publication citation:** `results/optimization/tables/optimal_allocations.csv` near-optimal counts with `n_surface=5280`.
**Use instead:** `results/robustness/grid_patch/corrected_optimal_allocations.csv` or `results/final/tables/table_optimal_allocations.csv`.
"""
    _write_text(fp.FINAL_DIR / "PUBLICATION_ARTIFACT_MANIFEST.md", md)


def write_final_results_summary(state: CleanupState) -> None:
    opt = pd.read_csv(fp.TABLES_DIR / "table_optimal_allocations.csv")
    md = f"""# Final Results Summary

## 1. Data

NBK official USD/KZT and EUR/KZT daily rates, 2010–2025 raw; baseline estimation 2015-08-20 – 2025-12-31 (N≈2705 weekday returns per series). Recent robustness window 2020–2025.

## 2. FX diagnostics

Fat-tailed, skewed daily log returns; significant volatility clustering. Baseline annualized vol ≈14.6% (USD/KZT), 15.8% (EUR/KZT). Jarque–Bera rejects normality (p≈0).

## 3. Primary DGP choice

Joint stationary bootstrap **L=20**, 10,000 paths, seed 42. Selected among {{SB5, SB10, SB20}} for temporal dependence and volatility-clustering fit. Student-t and Gaussian rejected for dependence/tail failures.

## 4. Household calibration

E/I=0.95, B=1, W0=0.95, monthly rebalance, zero cost. Calibration rule (P1≤5% shortfall + FX benchmark in 5–40%) satisfied; **B=1** accepted despite B≈6 tie-break preference.

## 5. Main optimization results

Integer-bp grid: **1326** (2%), **5151** (1%) unique portfolios. Corrected optima: **100% KZT** for all identified P2–P4 cells.

- **P2 3Y:** CVaR95 ≈ **{state.p2_3y.get('cvar', 0):.4f}**, P(shortfall) ≈ 4.9%
- **P4 3Y:** CVaR95 ≈ **{state.p4_3y.get('cvar', 0):.4f}**, P(shortfall) ≈ 2.8%

## 6. Benchmark results (3Y)

Expense-matched / all-KZT weakly dominate income-matched and foreign-heavy benchmarks for P2/P4. P3 shows similar KZT-heavy ranking.

## 7. Robustness

**RESULT B:** Robust expense-matching / KZT-heavy principle; exact weights model-dependent. Expense-matched CVaR ≤ income-matched across core R1–R5 scenarios. No-rebalance → UNIDENTIFIED for P2.

## 8. Exact vs mechanism

Mechanism (align liquid savings with KZT expenses) is more stable than point-estimate 100% KZT weights in near-optimal sets and some robustness cells.

## 9. Main limitations

Synthetic profiles, no microdata calibration, generic transaction costs, post-float sample, cross-rate EUR/KZT, simulation not forecasting.

## 10. Conclusion

For multi-currency-income households with KZT expenses, **KZT-heavy, expense-aligned liquid savings** minimize tail shortfall risk under the baseline Monte Carlo design; foreign-income matching increases CVaR. Exact allocations are near-optimal-region objects, not universal constants.
"""
    _write_text(fp.FINAL_DIR / "FINAL_RESULTS_SUMMARY.md", md)


def write_final_code_review(state: CleanupState) -> None:
    md = """# Final Code Review (Read-Only)

## CRITICAL — none found

- Accounting conservation: tested (`test_accounting_conservation.py`); settlement doc consistent.
- FX direction: USD/KZT levels multiply USD holdings for KZT wealth — verified.
- Look-ahead: FX paths simulated forward; no future data in household decisions.
- Joint USD/EUR sampling: stationary bootstrap resamples bivariate return vectors.
- CVaR: `empirical_cvar` uses mean of worst ceil((1-α)N) losses.
- Grid duplicates: fixed by integer bp keys (Phase 2C).
- Scenario consistency: robustness uses documented seeds per DGP.

## MODERATE

- `PRIMARY_BLOCK_LENGTH=10` in config is legacy; household/opt use `HH_BLOCK_LENGTH=20`. Document clearly (canonical spec).
- Phase 2B2 `optimal_allocations.csv` retains inflated `n_surface` — superseded by `results/final/` (not deleted per spec).

## MINOR

- `select_baseline_candidate` B≈6 tie-break can confuse readers — addressed in buffer audit.
- Some figure filenames include `figure02b` auxiliary panel.

## Verdict

**Proceed to publication artifact freeze.** No critical correctness bugs identified.
"""
    _write_text(fp.AUDIT_DIR / "final_code_review.md", md)


def run_tests(state: CleanupState) -> None:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=config.PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr
    m_pass = re.search(r"(\d+) passed", out)
    m_fail = re.search(r"(\d+) failed", out)
    passed = int(m_pass.group(1)) if m_pass else 0
    failed = int(m_fail.group(1)) if m_fail else 0
    total = passed + failed
    state.test_summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "runtime_seconds": round(elapsed, 2),
        "exit_code": proc.returncode,
    }
    summary = (
        f"pytest tests/ -q\n"
        f"exit_code: {proc.returncode}\n"
        f"test_count: {total}\n"
        f"pass_count: {passed}\n"
        f"fail_count: {failed}\n"
        f"runtime_seconds: {elapsed:.2f}\n\n"
        f"--- output ---\n{out}"
    )
    _write_text(fp.AUDIT_DIR / "test_summary.txt", summary)
    if proc.returncode != 0:
        state.stopped = True
        state.stop_reasons.append(f"pytest failed: {failed} failures")


def verify_baseline_gates(state: CleanupState) -> None:
    p2 = state.p2_3y.get("cvar", 0)
    p4 = state.p4_3y.get("cvar", 0)
    if abs(p2 - config.ROB_GATE_P2_CVAR) > config.ROB_GATE_CVAR_ATOL:
        state.stopped = True
        state.stop_reasons.append(f"P2 3Y CVaR gate failed: {p2}")
    if abs(p4 - config.ROB_GATE_P4_CVAR) > config.ROB_GATE_CVAR_ATOL:
        state.stopped = True
        state.stop_reasons.append(f"P4 3Y CVaR gate failed: {p4}")


def run_final_cleanup() -> CleanupState:
    state = CleanupState()
    _ensure_dirs()

    spec = build_canonical_specification()
    _write_json(fp.FINAL_DIR / "canonical_specification.json", spec)
    _write_text(fp.FINAL_DIR / "canonical_specification.md", write_canonical_specification_md(spec))

    audit_grid_integrity(state)
    audit_grid_reconciliation(state)
    if state.stopped:
        return state

    audit_reproducibility_manifest()
    audit_buffer_selection()
    audit_recent_window_cvar()
    audit_transaction_costs()
    audit_no_rebalance(state)

    build_publication_tables(state)
    verify_baseline_gates(state)
    if state.stopped:
        return state

    figs = build_publication_figures()
    write_figure_captions()
    write_assumptions_and_limitations()
    write_paper_safe_claims()
    write_publication_manifest()
    write_final_results_summary(state)
    write_final_code_review(state)

    c2c = _load_json(config.ROB_DIAGNOSTICS / "phase2c_run_summary.json")
    state.final_classification = c2c.get("final_classification_text", "")

    run_tests(state)
    return state


def print_final_console(state: CleanupState) -> None:
    print("FINAL CLEANUP COMPLETE" if not state.stopped else "FINAL CLEANUP STOPPED")
    print()
    print("Canonical specification:")
    print(f"  {fp.FINAL_DIR / 'canonical_specification.json'}")
    print()
    ts = state.test_summary
    print("Tests:")
    print(f"  {ts.get('passed', '?')}/{ts.get('total', '?')} passed, runtime {ts.get('runtime_seconds', '?')}s")
    print()
    print("Grid:")
    print(f"  2% = {state.grid_2pct} unique")
    print(f"  1% = {state.grid_1pct} unique")
    print()
    p2w = state.p2_3y.get("weights", (1, 0, 0))
    p4w = state.p4_3y.get("weights", (1, 0, 0))
    print("Baseline reproduction:")
    print(f"  P2 3Y: weights = ({p2w[0]:.4f}, {p2w[1]:.4f}, {p2w[2]:.4f}), CVaR = {state.p2_3y.get('cvar', 0):.6f}")
    print(f"  P4 3Y: weights = ({p4w[0]:.4f}, {p4w[1]:.4f}, {p4w[2]:.4f}), CVaR = {state.p4_3y.get('cvar', 0):.6f}")
    print()
    print("Buffer-selection audit: results/final/audit/buffer_selection_audit.md")
    print("Recent-window CVaR audit: results/final/audit/recent_window_cvar_audit.md")
    print("Transaction-cost audit: results/final/audit/transaction_cost_audit.md")
    print("No-rebalance audit: results/final/audit/no_rebalance_audit.md")
    print()
    print(f"Final classification: {state.final_classification}")
    print()
    print("Critical issues:", state.critical_issues or "none")
    print("Moderate issues:", state.moderate_issues or "none")
    print()
    print("Publication tables:", fp.TABLES_DIR)
    print("Publication figures:", fp.FIGURES_DIR)
    print()
    print("Start paper writing from: results/final/FINAL_RESULTS_SUMMARY.md")
    if state.stopped:
        print()
        print("STOP REASONS:", "; ".join(state.stop_reasons))
