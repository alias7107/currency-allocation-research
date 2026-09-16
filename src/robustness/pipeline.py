"""Phase 2C main pipeline: grid patch + staged robustness R1–R5."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Avoid matplotlib font-cache aborts in restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-currency-research")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.household.profiles import PRIMARY_PROFILES
from src.optimization.evaluator import monthly_fx_for_horizon
from src.optimization.grid import expected_grid_size_bp, generate_simplex_grid
from src.robustness.classify import (
    allocation_l1_distance,
    allocation_max_distance,
    classify_robustness,
    expense_matched_weights,
    final_paper_classification,
    income_matched_weights,
)
from src.robustness.grid_patch import run_grid_patch
from src.robustness.optimize import baseline_weights, make_cashflows, optimize_cell
from src.robustness.scenarios import (
    baseline_sb20_levels,
    generate_fx_levels,
    recent_window_block_diagnostic,
)
from src.utils import write_csv


plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _dirs() -> None:
    for p in (
        config.ROB_DIR,
        config.ROB_FIGURES,
        config.ROB_TABLES,
        config.ROB_DIAGNOSTICS,
        config.ROB_GRID_PATCH,
    ):
        p.mkdir(parents=True, exist_ok=True)


def _profile_by_name(name: str):
    return next(p for p in PRIMARY_PROFILES if p.name == name)


def _save_fig(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(config.ROB_FIGURES / f"{stem}.png")
    fig.savefig(config.ROB_FIGURES / f"{stem}.pdf")
    plt.close(fig)


def _record(
    *,
    dimension: str,
    specification: str,
    profile: str,
    horizon: int,
    dgp: str,
    block_length,
    estimation_window: str,
    ei: float,
    B: float,
    rebalance: str,
    cost: float,
    cell,
    w_base: np.ndarray,
) -> dict:
    w = cell.w_star
    rob_class = classify_robustness(
        w_base=w_base,
        w_rob=w,
        cvar_expense=cell.expense_cvar,
        cvar_income=cell.income_cvar,
        cvar_opt=float(cell.opt_row["cvar95"]),
        ident_class=cell.identification_class,
        near_1pct_frac=float(cell.opt_row["near_optimal_1pct_frac"]),
    )
    ranking = (
        "expense_better"
        if cell.expense_cvar < cell.income_cvar - 1e-12
        else (
            "income_better"
            if cell.income_cvar < cell.expense_cvar - 1e-12
            else "tie"
        )
    )
    abs_diff = cell.expense_cvar - cell.income_cvar
    if cell.income_cvar > config.OPT_CVAR_ZERO_TOL:
        rel_pen = 100.0 * (cell.income_cvar - cell.expense_cvar) / cell.income_cvar
    else:
        rel_pen = np.nan
    mean_cost = getattr(cell.ev_star, "mean_cumulative_conversion_cost", 0.0)
    mean_vol = getattr(cell.ev_star, "mean_cumulative_conversion_volume", 0.0)
    annual_E = ei * config.OPT_I0 * 12.0
    return {
        "robustness_dimension": dimension,
        "specification": specification,
        "profile": profile,
        "horizon": horizon,
        "dgp": dgp,
        "block_length": block_length if block_length is not None else "",
        "estimation_window": estimation_window,
        "expense_income_ratio": ei,
        "buffer_months": B,
        "rebalance_frequency": rebalance,
        "conversion_cost": cost,
        "w_kzt": float(w[0]),
        "w_usd": float(w[1]),
        "w_eur": float(w[2]),
        "cvar95": float(cell.opt_row["cvar95"]),
        "shortfall_probability": float(cell.opt_row["shortfall_probability"]),
        "expected_shortfall": float(cell.opt_row["expected_cumulative_shortfall"]),
        "identification_class": cell.identification_class,
        "n_surface": int(cell.opt_row["n_surface"]),
        "expense_matched_cvar": cell.expense_cvar,
        "income_matched_cvar": cell.income_cvar,
        "expense_shortfall_probability": cell.expense_sf,
        "income_shortfall_probability": cell.income_sf,
        "expense_vs_income_abs_diff": abs_diff,
        "relative_income_penalty_pct": rel_pen,
        "expense_vs_income_ranking": ranking,
        "near_1pct_count": int(cell.opt_row["near_optimal_1pct_count"]),
        "near_1pct_frac": float(cell.opt_row["near_optimal_1pct_frac"]),
        "near_1pct_kzt_min": float(cell.opt_row["near_optimal_1pct_kzt_min"]),
        "near_1pct_kzt_max": float(cell.opt_row["near_optimal_1pct_kzt_max"]),
        "near_1pct_usd_min": float(cell.opt_row["near_optimal_1pct_usd_min"]),
        "near_1pct_usd_max": float(cell.opt_row["near_optimal_1pct_usd_max"]),
        "near_1pct_eur_min": float(cell.opt_row["near_optimal_1pct_eur_min"]),
        "near_1pct_eur_max": float(cell.opt_row["near_optimal_1pct_eur_max"]),
        "near_5pct_kzt_min": float(cell.opt_row["near_optimal_5pct_kzt_min"]),
        "near_5pct_kzt_max": float(cell.opt_row["near_optimal_5pct_kzt_max"]),
        "allocation_distance_from_baseline": allocation_l1_distance(w, w_base),
        "allocation_max_distance_from_baseline": allocation_max_distance(w, w_base),
        "mean_cumulative_conversion_cost": float(mean_cost) if mean_cost is not None else 0.0,
        "mean_cumulative_conversion_volume": float(mean_vol) if mean_vol is not None else 0.0,
        "cost_over_annual_expenses": (
            float(mean_cost) / annual_E if annual_E > 0 else np.nan
        ),
        "robustness_class": rob_class,
    }


def _run_opt(
    levels, s0, profile_name, years, ei, B, rebalance, cost, **opt_kwargs
):
    monthly = monthly_fx_for_horizon(levels, years)
    profile = _profile_by_name(profile_name)
    return optimize_cell(
        monthly,
        profile,
        s0,
        expense_income_ratio=ei,
        buffer_months=B,
        rebalance_frequency=rebalance,
        conversion_cost=cost,
        **opt_kwargs,
    )


def verify_baseline_gate(levels, s0) -> dict:
    rows = []
    ok = True
    for pname, expect in (
        ("P2_foreign_income_mismatch", config.ROB_GATE_P2_CVAR),
        ("P4_full_multi_currency", config.ROB_GATE_P4_CVAR),
    ):
        cell = _run_opt(
            levels, s0, pname, 3,
            config.ROB_BASELINE_EI, config.ROB_BASELINE_B,
            "monthly", 0.0,
        )
        w_ok = np.isclose(cell.w_star[0], 1.0, atol=1e-8)
        c_ok = np.isclose(
            float(cell.opt_row["cvar95"]),
            expect,
            rtol=config.ROB_GATE_CVAR_RTOL,
            atol=config.ROB_GATE_CVAR_ATOL,
        )
        rows.append(
            {
                "profile": pname,
                "w_kzt": float(cell.w_star[0]),
                "cvar95": float(cell.opt_row["cvar95"]),
                "expected_cvar": expect,
                "weights_ok": bool(w_ok),
                "cvar_ok": bool(c_ok),
            }
        )
        ok = ok and w_ok and c_ok
    return {"ok": ok, "rows": rows}


def run_robustness_suite(levels_base, s0_base) -> dict:
    w_base = baseline_weights()
    years = config.ROB_HORIZON_YEARS
    results = []
    surfaces = {}  # for figures
    reversals = []

    core = list(config.ROB_CORE_PROFILES)
    secondary = list(config.ROB_SECONDARY_PROFILES)
    win_base = f"{config.ROB_BASELINE_WINDOW_START}–{config.ROB_BASELINE_WINDOW_END}"

    # ---- Baseline reference cells ----
    for pname in core + secondary:
        cell = _run_opt(
            levels_base, s0_base, pname, years,
            config.ROB_BASELINE_EI, config.ROB_BASELINE_B, "monthly", 0.0,
        )
        rec = _record(
            dimension="R0_baseline",
            specification="baseline_SB20_EI95_B1_monthly_c0",
            profile=pname,
            horizon=years,
            dgp="SB20",
            block_length=20,
            estimation_window=win_base,
            ei=config.ROB_BASELINE_EI,
            B=config.ROB_BASELINE_B,
            rebalance="monthly",
            cost=0.0,
            cell=cell,
            w_base=w_base,
        )
        results.append(rec)
        if pname in core:
            surfaces[("baseline", pname)] = cell.coarse

    # ---- R1: alternative household ----
    for pname in core + secondary:
        cell = _run_opt(
            levels_base, s0_base, pname, years,
            config.ROB_ALT_EI, config.ROB_ALT_B, "monthly", 0.0,
        )
        # P1 structural deficit control
        rec = _record(
            dimension="R1_household",
            specification="EI105_B3",
            profile=pname,
            horizon=years,
            dgp="SB20",
            block_length=20,
            estimation_window=win_base,
            ei=config.ROB_ALT_EI,
            B=config.ROB_ALT_B,
            rebalance="monthly",
            cost=0.0,
            cell=cell,
            w_base=w_base,
        )
        results.append(rec)
        if pname in core:
            surfaces[("EI105_B3", pname)] = cell.coarse
        if rec["robustness_class"] == "REVERSED":
            reversals.append(rec)

    # P1 control under stress
    cell_p1 = _run_opt(
        levels_base, s0_base, "P1_local_control", years,
        config.ROB_ALT_EI, config.ROB_ALT_B, "monthly", 0.0,
    )
    results.append(
        _record(
            dimension="R1_household",
            specification="EI105_B3_P1_control",
            profile="P1_local_control",
            horizon=years,
            dgp="SB20",
            block_length=20,
            estimation_window=win_base,
            ei=config.ROB_ALT_EI,
            B=config.ROB_ALT_B,
            rebalance="monthly",
            cost=0.0,
            cell=cell_p1,
            w_base=w_base,
        )
    )

    # ---- R2: alternative DGPs ----
    cache_dir = config.ROB_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for spec in config.ROB_DGP_SPECS:
        if spec["name"] == "SB20":
            # already have baseline; still record explicit DGP row from baseline results
            continue
        levels, s0, meta = generate_fx_levels(
            kind=spec["kind"],
            block=spec["block"],
            seed=spec["seed"],
            n_paths=config.ROB_N_PATHS,
            window_start=config.ROB_BASELINE_WINDOW_START,
            window_end=config.ROB_BASELINE_WINDOW_END,
            cache_path=cache_dir / f"{spec['name']}_daily_levels.npy",
        )
        for pname in core:
            cell = _run_opt(
                levels, s0, pname, years,
                config.ROB_BASELINE_EI, config.ROB_BASELINE_B, "monthly", 0.0,
            )
            rec = _record(
                dimension="R2_dgp",
                specification=spec["name"],
                profile=pname,
                horizon=years,
                dgp=spec["name"],
                block_length=spec["block"],
                estimation_window=win_base,
                ei=config.ROB_BASELINE_EI,
                B=config.ROB_BASELINE_B,
                rebalance="monthly",
                cost=0.0,
                cell=cell,
                w_base=w_base,
            )
            results.append(rec)
            if rec["robustness_class"] == "REVERSED":
                reversals.append(rec)

    # Duplicate baseline into DGP table as SB20
    for r in list(results):
        if r["specification"] == "baseline_SB20_EI95_B1_monthly_c0" and r["profile"] in core:
            rr = dict(r)
            rr["robustness_dimension"] = "R2_dgp"
            rr["specification"] = "SB20"
            results.append(rr)

    # ---- R3: recent window ----
    levels_r, s0_r, meta_r = generate_fx_levels(
        kind="stationary_bootstrap",
        block=20,
        seed=config.ROB_RECENT_SEED,
        n_paths=config.ROB_N_PATHS,
        window_start=config.ROB_RECENT_START,
        window_end=config.ROB_RECENT_END,
        cache_path=cache_dir / "SB20_recent_2020_2025_daily_levels.npy",
    )
    win_recent = f"{config.ROB_RECENT_START}–{config.ROB_RECENT_END}"
    for pname in core:
        cell = _run_opt(
            levels_r, s0_r, pname, years,
            config.ROB_BASELINE_EI, config.ROB_BASELINE_B, "monthly", 0.0,
        )
        rec = _record(
            dimension="R3_window",
            specification="recent_2020_2025_SB20",
            profile=pname,
            horizon=years,
            dgp="SB20",
            block_length=20,
            estimation_window=win_recent,
            ei=config.ROB_BASELINE_EI,
            B=config.ROB_BASELINE_B,
            rebalance="monthly",
            cost=0.0,
            cell=cell,
            w_base=w_base,
        )
        results.append(rec)
        if rec["robustness_class"] == "REVERSED":
            reversals.append(rec)
    # baseline window rows for window table
    for r in list(results):
        if r["specification"] == "baseline_SB20_EI95_B1_monthly_c0" and r["profile"] in core:
            rr = dict(r)
            rr["robustness_dimension"] = "R3_window"
            rr["specification"] = "full_2015_2025_SB20"
            results.append(rr)

    block_diag = recent_window_block_diagnostic()

    # ---- R4: rebalancing (zero cost) ----
    for freq in config.ROB_REBALANCE_FREQS:
        if freq == "monthly":
            continue  # baseline already
        for pname in core:
            cell = _run_opt(
                levels_base, s0_base, pname, years,
                config.ROB_BASELINE_EI, config.ROB_BASELINE_B, freq, 0.0,
            )
            rec = _record(
                dimension="R4_rebalance",
                specification=f"rebalance_{freq}_c0",
                profile=pname,
                horizon=years,
                dgp="SB20",
                block_length=20,
                estimation_window=win_base,
                ei=config.ROB_BASELINE_EI,
                B=config.ROB_BASELINE_B,
                rebalance=freq,
                cost=0.0,
                cell=cell,
                w_base=w_base,
            )
            results.append(rec)
            if rec["robustness_class"] == "REVERSED":
                reversals.append(rec)
    for r in list(results):
        if r["specification"] == "baseline_SB20_EI95_B1_monthly_c0" and r["profile"] in core:
            rr = dict(r)
            rr["robustness_dimension"] = "R4_rebalance"
            rr["specification"] = "rebalance_monthly_c0"
            results.append(rr)

    # ---- R5: conversion costs (monthly) ----
    for c in config.ROB_CONVERSION_COSTS:
        if c == 0.0:
            continue
        for pname in core:
            cell = _run_opt(
                levels_base, s0_base, pname, years,
                config.ROB_BASELINE_EI, config.ROB_BASELINE_B, "monthly", c,
            )
            rec = _record(
                dimension="R5_costs",
                specification=f"cost_{c:.4f}_monthly",
                profile=pname,
                horizon=years,
                dgp="SB20",
                block_length=20,
                estimation_window=win_base,
                ei=config.ROB_BASELINE_EI,
                B=config.ROB_BASELINE_B,
                rebalance="monthly",
                cost=c,
                cell=cell,
                w_base=w_base,
            )
            results.append(rec)
            if rec["robustness_class"] == "REVERSED":
                reversals.append(rec)
    for r in list(results):
        if r["specification"] == "baseline_SB20_EI95_B1_monthly_c0" and r["profile"] in core:
            rr = dict(r)
            rr["robustness_dimension"] = "R5_costs"
            rr["specification"] = "cost_0.0000_monthly"
            results.append(rr)

    # Cross: informative cost × rebalance freqs
    c_star = config.ROB_COST_REBALANCE_CROSS
    for freq in config.ROB_REBALANCE_FREQS:
        for pname in core:
            if freq == "monthly":
                # may already exist
                existing = [
                    x
                    for x in results
                    if x["profile"] == pname
                    and x["conversion_cost"] == c_star
                    and x["rebalance_frequency"] == "monthly"
                    and x["robustness_dimension"] == "R5_costs"
                ]
                if existing:
                    continue
            cell = _run_opt(
                levels_base, s0_base, pname, years,
                config.ROB_BASELINE_EI, config.ROB_BASELINE_B, freq, c_star,
            )
            rec = _record(
                dimension="R5_costs",
                specification=f"cost_{c_star:.4f}_{freq}",
                profile=pname,
                horizon=years,
                dgp="SB20",
                block_length=20,
                estimation_window=win_base,
                ei=config.ROB_BASELINE_EI,
                B=config.ROB_BASELINE_B,
                rebalance=freq,
                cost=c_star,
                cell=cell,
                w_base=w_base,
            )
            results.append(rec)
            if rec["robustness_class"] == "REVERSED":
                reversals.append(rec)

    # ---- 50k confirmation for reversals / material shifts ----
    confirm_rows = []
    material = [
        r
        for r in results
        if r["robustness_class"] == "REVERSED"
        or r["allocation_max_distance_from_baseline"] > 0.15
    ]
    # Always confirm R1 P2/P4 if weights moved
    for r in results:
        if r["robustness_dimension"] == "R1_household" and r["profile"] in core:
            if r not in material:
                material.append(r)

    if material:
        # Confirm under the same DGP as the flagged case when possible; default SB20 base
        levels50, s0_50, _ = generate_fx_levels(
            kind="stationary_bootstrap",
            block=20,
            seed=config.ROB_CONFIRM_SEED,
            n_paths=config.ROB_CONFIRM_N_PATHS,
            window_start=config.ROB_BASELINE_WINDOW_START,
            window_end=config.ROB_BASELINE_WINDOW_END,
            cache_path=cache_dir / "SB20_confirm50k_daily_levels.npy",
        )
        monthly50 = monthly_fx_for_horizon(levels50, years)
        seen = set()
        for r in material:
            key = (r["profile"], r["specification"], r["rebalance_frequency"], r["conversion_cost"],
                   r["expense_income_ratio"], r["buffer_months"])
            if key in seen:
                continue
            seen.add(key)
            # Only confirm on baseline FX when DGP/window match baseline; else skip note
            if r["dgp"] != "SB20" or "2020" in str(r["estimation_window"]):
                confirm_rows.append(
                    {
                        "profile": r["profile"],
                        "specification": r["specification"],
                        "status": "skipped_non_baseline_fx",
                        "note": "50k confirmation limited to SB20 baseline-window paths",
                    }
                )
                continue
            profile = _profile_by_name(r["profile"])
            cf, E0, W0 = make_cashflows(
                profile, s0_base, r["expense_income_ratio"], r["buffer_months"]
            )
            from src.optimization.evaluator import evaluate_weights
            from src.household.profiles import strategy_catalog

            candidates = {
                "opt_10k": np.array([r["w_kzt"], r["w_usd"], r["w_eur"]]),
                "all_kzt": np.array([1.0, 0.0, 0.0]),
                "expense": expense_matched_weights(profile),
                "income": income_matched_weights(profile),
            }
            for name, w in strategy_catalog(profile):
                candidates[name] = w
            evals = {}
            for lab, w in candidates.items():
                ev = evaluate_weights(
                    monthly50,
                    cf,
                    W0,
                    w,
                    rebalance_frequency=r["rebalance_frequency"],
                    conversion_cost=r["conversion_cost"],
                )
                evals[lab] = ev
            best_lab = min(evals, key=lambda k: (evals[k].cvar95, -evals[k].weights[0]))
            confirm_rows.append(
                {
                    "profile": r["profile"],
                    "specification": r["specification"],
                    "status": "ok",
                    "best_label_50k": best_lab,
                    "best_cvar_50k": evals[best_lab].cvar95,
                    "opt10k_cvar_50k": evals["opt_10k"].cvar95,
                    "expense_cvar_50k": evals["expense"].cvar95,
                    "income_cvar_50k": evals["income"].cvar95,
                    "ranking_stable": (
                        evals["expense"].cvar95 <= evals["income"].cvar95 + 1e-12
                    ),
                }
            )

    df = pd.DataFrame(results)
    write_csv(df, config.ROB_TABLES / "robustness_results.csv")

    # Summary one row per specification × profile (core)
    summary_rows = []
    base_map = {
        p: df[
            (df.specification == "baseline_SB20_EI95_B1_monthly_c0") & (df.profile == p)
        ].iloc[0]
        for p in core + secondary
        if len(df[(df.specification == "baseline_SB20_EI95_B1_monthly_c0") & (df.profile == p)])
    }
    for _, r in df.iterrows():
        if r.profile not in base_map:
            continue
        b = base_map[r.profile]
        summary_rows.append(
            {
                "dimension": r.robustness_dimension,
                "specification": r.specification,
                "profile": r.profile,
                "baseline_w_kzt": b.w_kzt,
                "robustness_w_kzt": r.w_kzt,
                "robustness_w_usd": r.w_usd,
                "robustness_w_eur": r.w_eur,
                "allocation_shift_l1": r.allocation_distance_from_baseline,
                "baseline_cvar": b.cvar95,
                "robustness_cvar": r.cvar95,
                "cvar_shift": r.cvar95 - b.cvar95,
                "expense_vs_income_ranking": r.expense_vs_income_ranking,
                "near_1pct_kzt_range": f"[{r.near_1pct_kzt_min:.2f},{r.near_1pct_kzt_max:.2f}]",
                "robustness_class": r.robustness_class,
            }
        )
    summary = pd.DataFrame(summary_rows).drop_duplicates(
        subset=["dimension", "specification", "profile"], keep="last"
    )
    write_csv(summary, config.ROB_TABLES / "robustness_summary.csv")

    # Expense vs income
    ei_rows = []
    for _, r in df.iterrows():
        if r.profile not in core + secondary:
            continue
        profile = _profile_by_name(r.profile)
        we = expense_matched_weights(profile)
        wi = income_matched_weights(profile)
        ei_rows.append(
            {
                "profile": r.profile,
                "specification": r.specification,
                "dimension": r.robustness_dimension,
                "expense_weights": f"({we[0]:.2f},{we[1]:.2f},{we[2]:.2f})",
                "income_weights": f"({wi[0]:.2f},{wi[1]:.2f},{wi[2]:.2f})",
                "expense_cvar": r.expense_matched_cvar,
                "income_cvar": r.income_matched_cvar,
                "absolute_difference": r.expense_vs_income_abs_diff,
                "relative_income_penalty_pct": r.relative_income_penalty_pct,
                "expense_shortfall_probability": r.expense_shortfall_probability,
                "income_shortfall_probability": r.income_shortfall_probability,
                "ranking": r.expense_vs_income_ranking,
            }
        )
    write_csv(pd.DataFrame(ei_rows), config.ROB_TABLES / "expense_vs_income_matching.csv")

    # Cost / rebalance / dgp / window extracts
    write_csv(
        df[df.robustness_dimension == "R5_costs"].copy(),
        config.ROB_TABLES / "transaction_cost_sensitivity.csv",
    )
    write_csv(
        df[df.robustness_dimension == "R4_rebalance"].copy(),
        config.ROB_TABLES / "rebalancing_sensitivity.csv",
    )
    write_csv(
        df[df.robustness_dimension == "R2_dgp"].copy(),
        config.ROB_TABLES / "dgp_sensitivity.csv",
    )
    write_csv(
        df[df.robustness_dimension == "R3_window"].copy(),
        config.ROB_TABLES / "window_sensitivity.csv",
    )
    if confirm_rows:
        write_csv(pd.DataFrame(confirm_rows), config.ROB_TABLES / "confirmation_50k_robustness.csv")

    (config.ROB_DIAGNOSTICS / "recent_window_block_diagnostic.json").write_text(
        json.dumps(block_diag, indent=2)
    )

    return {
        "results": df,
        "summary": summary,
        "surfaces": surfaces,
        "reversals": reversals,
        "confirm": confirm_rows,
        "block_diag": block_diag,
    }


def make_figures(df: pd.DataFrame, surfaces: dict) -> None:
    core = list(config.ROB_CORE_PROFILES)

    # R1: KZT share across specs
    for pname, stem, title in (
        (core[0], "figR1_P2_kzt_share_across_specs", "Figure R1. P2 3Y optimal/near-optimal KZT share"),
        (core[1], "figR2_P4_kzt_share_across_specs", "Figure R2. P4 3Y optimal/near-optimal KZT share"),
    ):
        g = df[df.profile == pname].drop_duplicates(
            subset=["robustness_dimension", "specification"], keep="last"
        )
        # Prefer one row per dimension's main specs
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = [f"{r.robustness_dimension}\n{r.specification}" for _, r in g.iterrows()]
        ax.bar(range(len(g)), g.w_kzt, color="steelblue", label="exact w_KZT")
        ax.errorbar(
            range(len(g)),
            (g.near_1pct_kzt_min + g.near_1pct_kzt_max) / 2,
            yerr=[
                (g.near_1pct_kzt_min + g.near_1pct_kzt_max) / 2 - g.near_1pct_kzt_min,
                g.near_1pct_kzt_max - (g.near_1pct_kzt_min + g.near_1pct_kzt_max) / 2,
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
            label="1% near-opt KZT range",
        )
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
        ax.set_ylabel("KZT weight")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
        _save_fig(fig, stem)

    # R3 expense vs income
    g = df[df.profile.isin(core)].drop_duplicates(
        subset=["profile", "robustness_dimension", "specification"], keep="last"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.expense_matched_cvar, 0.4, label="expense-matched")
    ax.bar(x + 0.2, g.income_matched_cvar, 0.4, label="income-matched")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.profile.split('_')[0]}|{r.specification}" for _, r in g.iterrows()],
        fontsize=5,
        rotation=65,
        ha="right",
    )
    ax.set_ylabel("CVaR95")
    ax.set_title("Figure R3. Expense-matched vs income-matched CVaR")
    ax.legend()
    _save_fig(fig, "figR3_expense_vs_income_cvar")

    # R4 surfaces baseline vs EI105
    for key, stem, title in (
        (("baseline", core[0]), "figR4a_surface_P2_baseline", "Figure R4a. P2 CVaR surface — baseline"),
        (("EI105_B3", core[0]), "figR4b_surface_P2_EI105_B3", "Figure R4b. P2 CVaR surface — E/I=1.05, B=3"),
        (("baseline", core[1]), "figR4c_surface_P4_baseline", "Figure R4c. P4 CVaR surface — baseline"),
        (("EI105_B3", core[1]), "figR4d_surface_P4_EI105_B3", "Figure R4d. P4 CVaR surface — E/I=1.05, B=3"),
    ):
        if key not in surfaces:
            continue
        surf = surfaces[key]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sc = ax.scatter(surf.w_usd, surf.w_eur, c=surf.cvar95, s=16, cmap="viridis")
        fig.colorbar(sc, ax=ax, label="CVaR95")
        ax.set_xlabel("w_USD")
        ax.set_ylabel("w_EUR")
        ax.set_title(title)
        _save_fig(fig, stem)

    # R5 costs
    cost_df = df[
        (df.robustness_dimension == "R5_costs")
        & (df.rebalance_frequency == "monthly")
        & (df.profile.isin(core))
    ].drop_duplicates(subset=["profile", "conversion_cost"], keep="last")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pname, ylab in zip(
        axes, core, ("Optimal w_KZT", "CVaR95")
    ):
        pass
    for ax, col, ylab, title in zip(
        axes,
        ("w_kzt", "cvar95"),
        ("Optimal w_KZT", "CVaR95"),
        ("KZT share vs cost", "CVaR vs cost"),
    ):
        for pname in core:
            g = cost_df[cost_df.profile == pname].sort_values("conversion_cost")
            ax.plot(100 * g.conversion_cost, g[col], marker="o", label=pname.split("_")[0])
        ax.set_xlabel("Conversion cost (%)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Figure R5. Transaction-cost sensitivity (monthly rebalance)")
    _save_fig(fig, "figR5_transaction_costs")

    # R6 rebalancing
    reb = df[
        (df.robustness_dimension == "R4_rebalance") & (df.profile.isin(core))
    ].drop_duplicates(subset=["profile", "rebalance_frequency"], keep="last")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for ax, col, title in zip(
        axes, ("w_kzt", "cvar95"), ("Optimal w_KZT", "CVaR95")
    ):
        for pname in core:
            g = reb[reb.profile == pname]
            order = ["monthly", "quarterly", "none"]
            g = g.set_index("rebalance_frequency").reindex(order).reset_index()
            ax.plot(order, g[col], marker="o", label=pname.split("_")[0])
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Figure R6. Rebalancing frequency (c=0)")
    _save_fig(fig, "figR6_rebalancing")


def write_robustness_report(
    patch: dict,
    gate: dict,
    pack: dict,
    letter: str,
    letter_text: str,
) -> None:
    df = pack["results"]
    lines = [
        "# Phase 2C Robustness Report",
        "",
        "## 1–2. Grid integrity patch",
        "",
        f"- 2% unique grid count: {patch['grid_2pct']}",
        f"- 1% unique grid count: {patch['grid_1pct']}",
        f"- Economic optima/CVaR changed: **{'YES' if patch['economic_changed'] else 'NO'}**",
        f"- Near-optimal counts changed: **{'YES' if patch['counts_changed'] else 'NO'}**",
        f"- Identification classes changed: **{'YES' if patch['ident_changed'] else 'NO'}**",
        "",
        "See `grid_patch_report.md`. Original Phase 2B2 files were **not** overwritten.",
        "",
        "## Baseline reproduction gate",
        "",
        json.dumps(gate, indent=2, default=str),
        "",
        "## 3–4. Household environment (R1)",
        "",
    ]
    r1 = df[df.robustness_dimension == "R1_household"]
    lines.append(r1[
        [
            "specification", "profile", "w_kzt", "w_usd", "w_eur", "cvar95",
            "expense_matched_cvar", "income_matched_cvar", "robustness_class",
            "near_1pct_kzt_min", "near_1pct_kzt_max",
        ]
    ].to_string(index=False))
    lines.extend(
        [
            "",
            "P1 under E/I=1.05, B=3 is the structural-deficit control: shortfalls can arise "
            "from cumulative E>I even with all-KZT (no FX mismatch).",
            "",
            "## 5–7. DGP robustness (R2)",
            "",
        ]
    )
    r2 = df[df.robustness_dimension == "R2_dgp"]
    lines.append(r2[
        ["specification", "profile", "w_kzt", "cvar95", "expense_vs_income_ranking", "robustness_class"]
    ].to_string(index=False))
    lines.extend(["", "## 8. Estimation window (R3)", ""])
    r3 = df[df.robustness_dimension == "R3_window"]
    lines.append(r3[
        ["specification", "profile", "w_kzt", "cvar95", "robustness_class"]
    ].to_string(index=False))
    lines.append("")
    lines.append("Recent-window block diagnostic:")
    lines.append(json.dumps(pack["block_diag"], indent=2))
    lines.extend(["", "## 9–10. Rebalancing (R4)", ""])
    r4 = df[df.robustness_dimension == "R4_rebalance"]
    lines.append(r4[
        ["specification", "profile", "w_kzt", "cvar95", "mean_cumulative_conversion_volume", "robustness_class"]
    ].to_string(index=False))
    lines.extend(["", "## 11–12. Conversion costs (R5)", ""])
    r5 = df[df.robustness_dimension == "R5_costs"]
    lines.append(r5[
        [
            "specification", "profile", "conversion_cost", "rebalance_frequency",
            "w_kzt", "cvar95", "mean_cumulative_conversion_cost",
            "expense_matched_cvar", "income_matched_cvar", "robustness_class",
        ]
    ].to_string(index=False))
    lines.extend(
        [
            "",
            "## 13–15. Expense vs income matching",
            "",
            "See `expense_vs_income_matching.csv`. Summary ranking counts:",
            "",
        ]
    )
    for prof in list(config.ROB_CORE_PROFILES) + list(config.ROB_SECONDARY_PROFILES):
        sub = df[df.profile == prof]
        if sub.empty:
            continue
        vc = sub.drop_duplicates(
            subset=["robustness_dimension", "specification"]
        ).expense_vs_income_ranking.value_counts()
        lines.append(f"- {prof}: {vc.to_dict()}")
    lines.extend(
        [
            "",
            "## 16–19. What is robust / paper framing",
            "",
            f"**Final classification: {letter_text}**",
            "",
            "Strong enough for the paper (conditional on the modeled frictions):",
            "- Expense-aligned / KZT-heavy liquid savings reduce CVaR relative to income-matched "
            "foreign-heavy savings across the staged robustness design, unless a specification "
            "is classified REVERSED.",
            "- Exact 100% KZT is the baseline argmin but need not be claimed as uniquely "
            "structural when near-optimal regions widen or exact weights move under alternatives.",
            "",
            "Conditional findings:",
            "- Thin buffer (B=1) and monthly zero-cost rebalancing favor KZT-heavy solutions.",
            "- Positive conversion costs and no-rebalance change wealth dynamics; interpret "
            "exact weights with near-optimal ranges.",
            "",
            "## 17. Reversals",
            "",
        ]
    )
    if pack["reversals"]:
        lines.append(pd.DataFrame(pack["reversals"])[
            ["specification", "profile", "w_kzt", "cvar95", "robustness_class"]
        ].to_string(index=False))
    else:
        lines.append("None under the staged design at N=10,000.")
    lines.extend(
        [
            "",
            "## 20. Limitations",
            "",
            "- No GARCH/DCC, interest, inflation, asset returns, or taxes.",
            "- Conversion costs are scenario parameters, not calibrated KZ spreads.",
            "- Cross-DGP path indices are not economically paired.",
            "- 50k confirmation used selectively for material shifts on SB20 baseline window.",
            "",
            "## RQ5",
            "",
            "Optimal currency allocations and downside-risk reductions are **partially sensitive** "
            "to household design (buffer/deficit), DGP block length, sample window, rebalancing, "
            "and conversion costs. The **expense-matching / KZT-heavy principle** is the more "
            "stable object than any single grid-point weight vector; exact all-KZT dominance "
            f"is best described under classification **{letter}**.",
            "",
        ]
    )
    (config.ROB_DIAGNOSTICS / "robustness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_phase2c() -> dict:
    _dirs()
    # Sanity grid counts
    assert expected_grid_size_bp(200) == 1326
    assert expected_grid_size_bp(100) == 5151
    assert len(generate_simplex_grid(200)) == 1326

    print("PART A — grid integrity patch...")
    patch = run_grid_patch()
    if patch["economic_changed"]:
        print("STOP: grid integrity economic failure — see grid_integrity_failure_report.md")
        return {"stopped": True, "patch": patch}

    print("Baseline FX + reproduction gate...")
    levels, s0, meta = baseline_sb20_levels()
    gate = verify_baseline_gate(levels, s0)
    write_csv(pd.DataFrame(gate["rows"]), config.ROB_DIAGNOSTICS / "baseline_reproduction_gate.csv")
    if not gate["ok"]:
        (config.ROB_DIAGNOSTICS / "grid_integrity_failure_report.md").write_text(
            "# Baseline reproduction gate failed\n\n" + json.dumps(gate, indent=2),
            encoding="utf-8",
        )
        print("STOP: baseline reproduction gate failed")
        return {"stopped": True, "patch": patch, "gate": gate}

    print("PART B — staged robustness R1–R5...")
    pack = run_robustness_suite(levels, s0)
    make_figures(pack["results"], pack["surfaces"])
    classes = (
        pack["summary"]["robustness_class"].tolist()
        if len(pack["summary"])
        else []
    )
    letter, letter_text = final_paper_classification(classes)
    write_robustness_report(patch, gate, pack, letter, letter_text)

    summary = {
        "stopped": False,
        "patch": {
            "economic_changed": patch["economic_changed"],
            "counts_changed": patch["counts_changed"],
            "ident_changed": patch["ident_changed"],
            "grid_2pct": patch["grid_2pct"],
            "grid_1pct": patch["grid_1pct"],
        },
        "gate": gate,
        "final_classification": letter,
        "final_classification_text": letter_text,
        "n_result_rows": int(len(pack["results"])),
        "n_reversals": len(pack["reversals"]),
    }
    (config.ROB_DIAGNOSTICS / "phase2c_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
