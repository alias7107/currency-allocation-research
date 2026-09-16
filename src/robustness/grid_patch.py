"""Part A — Phase 2B2 grid-integrity patch (integer bp; do not overwrite 2B2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.household.profiles import PRIMARY_PROFILES
from src.optimization.evaluator import load_or_generate_fx_levels, monthly_fx_for_horizon
from src.optimization.grid import (
    TOTAL_BP,
    expected_grid_size_bp,
    generate_simplex_grid,
)
from src.robustness.optimize import optimize_cell
from src.utils import write_csv


def _dirs() -> None:
    for p in (
        config.ROB_GRID_PATCH,
        config.ROB_DIAGNOSTICS,
        config.ROB_TABLES,
        config.ROB_FIGURES,
    ):
        p.mkdir(parents=True, exist_ok=True)


def run_grid_patch() -> dict:
    """Rerun corrected grids for all Phase 2B2 cells; compare to original files."""
    _dirs()
    assert expected_grid_size_bp(200) == 1326
    assert expected_grid_size_bp(100) == 5151
    g2 = generate_simplex_grid(200)
    g1 = generate_simplex_grid(100)
    assert len(g2) == 1326 and len(g1) == 5151

    old_opt = pd.read_csv(config.OPT_TABLES / "optimal_allocations.csv")
    daily_levels, s0 = load_or_generate_fx_levels()

    corr_rows = []
    near_rows = []
    cmp_rows = []
    economic_changed = False
    failure_details = []

    for years in config.OPT_HORIZON_YEARS:
        monthly = monthly_fx_for_horizon(daily_levels, years)
        for profile in PRIMARY_PROFILES:
            cell = optimize_cell(
                monthly,
                profile,
                s0,
                expense_income_ratio=config.OPT_EXPENSE_INCOME_RATIO,
                buffer_months=config.OPT_BUFFER_MONTHS,
                rebalance_frequency="monthly",
                conversion_cost=0.0,
            )
            o = cell.opt_row
            corr_rows.append(
                {
                    "profile": profile.name,
                    "horizon_years": years,
                    "expense_income_ratio": config.OPT_EXPENSE_INCOME_RATIO,
                    "buffer_months": config.OPT_BUFFER_MONTHS,
                    "w_kzt": float(o["w_kzt"]),
                    "w_usd": float(o["w_usd"]),
                    "w_eur": float(o["w_eur"]),
                    "cvar95": float(o["cvar95"]),
                    "var95": float(o["var95"]),
                    "shortfall_probability": float(o["shortfall_probability"]),
                    "expected_cumulative_shortfall": float(
                        o["expected_cumulative_shortfall"]
                    ),
                    "wealth_depletion_probability": float(
                        o["wealth_depletion_probability"]
                    ),
                    "identification_class": cell.identification_class,
                    "n_surface": int(o["n_surface"]),
                    "near_optimal_1pct_count": int(o["near_optimal_1pct_count"]),
                    "near_optimal_1pct_frac": float(o["near_optimal_1pct_frac"]),
                    "near_optimal_5pct_count": int(o["near_optimal_5pct_count"]),
                    "near_optimal_5pct_frac": float(o["near_optimal_5pct_frac"]),
                    "near_optimal_1pct_kzt_min": float(o["near_optimal_1pct_kzt_min"]),
                    "near_optimal_1pct_kzt_max": float(o["near_optimal_1pct_kzt_max"]),
                    "near_optimal_1pct_usd_min": float(o["near_optimal_1pct_usd_min"]),
                    "near_optimal_1pct_usd_max": float(o["near_optimal_1pct_usd_max"]),
                    "near_optimal_1pct_eur_min": float(o["near_optimal_1pct_eur_min"]),
                    "near_optimal_1pct_eur_max": float(o["near_optimal_1pct_eur_max"]),
                    "near_optimal_5pct_kzt_min": float(o["near_optimal_5pct_kzt_min"]),
                    "near_optimal_5pct_kzt_max": float(o["near_optimal_5pct_kzt_max"]),
                    "near_optimal_5pct_usd_min": float(o["near_optimal_5pct_usd_min"]),
                    "near_optimal_5pct_usd_max": float(o["near_optimal_5pct_usd_max"]),
                    "near_optimal_5pct_eur_min": float(o["near_optimal_5pct_eur_min"]),
                    "near_optimal_5pct_eur_max": float(o["near_optimal_5pct_eur_max"]),
                }
            )
            near_rows.append(corr_rows[-1])

            old = old_opt[
                (old_opt.profile == profile.name) & (old_opt.horizon_years == years)
            ].iloc[0]
            w_old = np.array([old.w_kzt, old.w_usd, old.w_eur])
            w_new = cell.w_star
            cvar_ok = bool(
                np.isclose(
                    float(o["cvar95"]),
                    float(old.cvar95),
                    rtol=config.ROB_GATE_CVAR_RTOL,
                    atol=config.ROB_GATE_CVAR_ATOL,
                )
            )
            w_ok = bool(np.allclose(w_old, w_new, atol=config.ROB_GATE_WEIGHT_ATOL))
            # Identification may change solely due to corrected fractions
            ident_changed = old.identification_class != cell.identification_class
            counts_changed = int(old.n_surface) != int(o["n_surface"])
            econ = not (cvar_ok and w_ok)
            if econ:
                economic_changed = True
                failure_details.append(
                    {
                        "profile": profile.name,
                        "horizon_years": years,
                        "old_w": w_old.tolist(),
                        "new_w": w_new.tolist(),
                        "old_cvar": float(old.cvar95),
                        "new_cvar": float(o["cvar95"]),
                    }
                )
            # Zero-CVaR full-surface sanity
            if float(o["cvar95"]) <= config.OPT_CVAR_ZERO_TOL and int(o["n_surface"]) >= 5000:
                if int(o["n_surface"]) != 5151:
                    economic_changed = True
                    failure_details.append(
                        {
                            "profile": profile.name,
                            "horizon_years": years,
                            "issue": f"zero-CVaR n_surface={o['n_surface']} != 5151",
                        }
                    )

            cmp_rows.append(
                {
                    "profile": profile.name,
                    "horizon_years": years,
                    "old_surface_count": int(old.n_surface),
                    "corrected_surface_count": int(o["n_surface"]),
                    "old_near_1pct_count": int(old.near_optimal_1pct_count),
                    "corrected_near_1pct_count": int(o["near_optimal_1pct_count"]),
                    "old_near_1pct_frac": float(old.near_optimal_1pct_frac),
                    "corrected_near_1pct_frac": float(o["near_optimal_1pct_frac"]),
                    "old_opt_weights": f"({old.w_kzt:.4f},{old.w_usd:.4f},{old.w_eur:.4f})",
                    "corrected_opt_weights": (
                        f"({w_new[0]:.4f},{w_new[1]:.4f},{w_new[2]:.4f})"
                    ),
                    "old_cvar": float(old.cvar95),
                    "corrected_cvar": float(o["cvar95"]),
                    "old_identification_class": old.identification_class,
                    "corrected_identification_class": cell.identification_class,
                    "identification_changed": ident_changed,
                    "counts_changed": counts_changed,
                    "economic_result_changed": econ,
                }
            )

    corr = pd.DataFrame(corr_rows)
    near = pd.DataFrame(near_rows)
    cmp = pd.DataFrame(cmp_rows)
    write_csv(cmp, config.ROB_GRID_PATCH / "grid_patch_comparison.csv")
    write_csv(corr, config.ROB_GRID_PATCH / "corrected_optimal_allocations.csv")
    write_csv(near, config.ROB_GRID_PATCH / "corrected_near_optimal_regions.csv")

    # Gate: P2/P4 3Y must remain 100% KZT with expected CVaR
    for pname, expect_cvar in (
        ("P2_foreign_income_mismatch", config.ROB_GATE_P2_CVAR),
        ("P4_full_multi_currency", config.ROB_GATE_P4_CVAR),
    ):
        r = corr[(corr.profile == pname) & (corr.horizon_years == 3)].iloc[0]
        if not np.isclose(r.w_kzt, 1.0, atol=1e-8):
            economic_changed = True
            failure_details.append({"gate": pname, "issue": "opt not 100% KZT"})
        if not np.isclose(
            r.cvar95,
            expect_cvar,
            rtol=config.ROB_GATE_CVAR_RTOL,
            atol=config.ROB_GATE_CVAR_ATOL,
        ):
            economic_changed = True
            failure_details.append(
                {
                    "gate": pname,
                    "issue": "CVaR mismatch",
                    "got": float(r.cvar95),
                    "expected": expect_cvar,
                }
            )

    report_lines = [
        "# Phase 2B2 Grid Integrity Patch Report",
        "",
        "## Original issue",
        "",
        "Phase 2B2 reported `n_surface = 5280` for some cells even though the complete "
        "1% simplex has exactly `C(102,2) = 5151` unique allocations. "
        "Float-keyed deduplication of coarse (2%) ∪ refined (1%) grids treated "
        "near-identical decimals (e.g. 0.14 vs 0.14000000000000001) as distinct.",
        "",
        "## Technical cause",
        "",
        "Grid construction mixed integer loops with float multiplication and "
        "`drop_duplicates` on decimal weight columns; `np.round` / `np.isclose` "
        "were used as uniqueness mechanisms in local refinement.",
        "",
        "## Integer basis-point solution",
        "",
        f"Internal weights use `TOTAL_BP = {TOTAL_BP}` with "
        "`w_kzt_bp + w_usd_bp + w_eur_bp = 10000`. "
        "2% grid: step_bp=200 → 1326; 1% grid: step_bp=100 → 5151. "
        "Merges deduplicate on integer tuples only.",
        "",
        f"- Exact 2% count: **{expected_grid_size_bp(200)}**",
        f"- Exact 1% count: **{expected_grid_size_bp(100)}**",
        "",
        "## Comparison outcomes",
        "",
        cmp.to_string(index=False),
        "",
        f"- Exact optima changed (beyond tolerance): **{'YES' if economic_changed else 'NO'}**",
        f"- Near-optimal counts changed: **{'YES' if bool(cmp['counts_changed'].any()) else 'NO'}**",
        f"- Identification classes changed: **{'YES' if bool(cmp['identification_changed'].any()) else 'NO'}**",
        "",
        "## Phase 2B2 economic conclusions",
        "",
        (
            "**INVALIDATED — STOP BEFORE ROBUSTNESS**"
            if economic_changed
            else (
                "Economic conclusions remain valid: optima and CVaR metrics match "
                "within numerical tolerance. Near-optimal counts/fractions are corrected "
                "(duplicate inflation removed). Identification labels may shift only where "
                "fractions were previously inflated by duplicates."
            )
        ),
        "",
    ]
    (config.ROB_DIAGNOSTICS / "grid_patch_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    if economic_changed:
        fail = [
            "# Grid Integrity Failure Report",
            "",
            "Economic results changed under the integer-bp grid patch. "
            "Phase 2C robustness was **not** started.",
            "",
            "```json",
            json.dumps(failure_details, indent=2),
            "```",
            "",
        ]
        (config.ROB_DIAGNOSTICS / "grid_integrity_failure_report.md").write_text(
            "\n".join(fail), encoding="utf-8"
        )

    return {
        "economic_changed": economic_changed,
        "failure_details": failure_details,
        "comparison": cmp,
        "corrected": corr,
        "grid_2pct": 1326,
        "grid_1pct": 5151,
        "counts_changed": bool(cmp["counts_changed"].any()),
        "ident_changed": bool(cmp["identification_changed"].any()),
    }
