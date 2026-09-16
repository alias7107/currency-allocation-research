"""Phase 2B2 CVaR allocation optimization pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.household.profiles import PRIMARY_PROFILES, strategy_catalog
from src.household.risk_metrics import empirical_cvar
from src.optimization.evaluator import (
    evaluate_grid,
    evaluate_weights,
    load_or_generate_fx_levels,
    monthly_fx_for_horizon,
    primary_cashflows,
    verify_phase2b1_benchmarks,
)
from src.optimization.grid import (
    TOTAL_BP,
    expected_grid_size,
    expected_grid_size_bp,
    generate_simplex_grid,
    local_refined_grid_bp,
    merge_result_frames,
)
from src.optimization.near_optimal import (
    classify_identification,
    near_optimal_summary,
    pick_argmin,
)
from src.optimization.uncertainty import paired_bootstrap_cvar, paired_path_comparison
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
    for p in (config.OPT_FIGURES, config.OPT_TABLES, config.OPT_DIAGNOSTICS):
        p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(config.OPT_FIGURES / f"{stem}.png")
    fig.savefig(config.OPT_FIGURES / f"{stem}.pdf")
    plt.close(fig)


def _row_metrics(ev, profile: str, years: int, stage: str) -> dict:
    return {
        "profile": profile,
        "horizon_years": years,
        "expense_income_ratio": config.OPT_EXPENSE_INCOME_RATIO,
        "buffer_months": config.OPT_BUFFER_MONTHS,
        "stage": stage,
        "w_kzt": float(ev.weights[0]),
        "w_usd": float(ev.weights[1]),
        "w_eur": float(ev.weights[2]),
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
        "n_paths": config.OPT_N_PATHS,
    }


def optimize_all_cells(daily_levels: np.ndarray, s0: np.ndarray) -> dict:
    coarse_step_bp = int(round(config.OPT_COARSE_STEP * TOTAL_BP))
    refined_step_bp = int(round(config.OPT_REFINED_STEP * TOTAL_BP))
    radius_bp = int(round(config.OPT_REFINE_NEIGHBORHOOD * TOTAL_BP))
    coarse_bp = generate_simplex_grid(coarse_step_bp)
    assert len(coarse_bp) == expected_grid_size_bp(coarse_step_bp)

    coarse_rows = []
    refined_rows = []
    optimal_rows = []
    bench_rows = []
    horizon_rows = []
    loss_store: dict[tuple, dict[str, np.ndarray]] = {}
    cell_detail = {}

    annual_E = config.OPT_EXPENSE_INCOME_RATIO * config.OPT_I0 * 12.0

    for years in config.OPT_HORIZON_YEARS:
        monthly = monthly_fx_for_horizon(daily_levels, years)
        for profile in PRIMARY_PROFILES:
            cf, E0, W0 = primary_cashflows(profile, s0)
            # Coarse
            cdf = evaluate_grid(monthly, cf, W0, coarse_bp, grid_is_bp=True)
            cdf.insert(0, "horizon_years", years)
            cdf.insert(0, "profile", profile.name)
            cdf["expense_income_ratio"] = config.OPT_EXPENSE_INCOME_RATIO
            cdf["buffer_months"] = config.OPT_BUFFER_MONTHS
            cdf["n_paths"] = config.OPT_N_PATHS
            coarse_rows.append(cdf)

            fine_bp = local_refined_grid_bp(
                coarse_bp,
                cdf["cvar95"].to_numpy(),
                refined_step_bp,
                config.OPT_REFINE_CVAR_BAND,
                radius_bp,
            )
            rdf = evaluate_grid(monthly, cf, W0, fine_bp, grid_is_bp=True)
            rdf.insert(0, "horizon_years", years)
            rdf.insert(0, "profile", profile.name)
            rdf["expense_income_ratio"] = config.OPT_EXPENSE_INCOME_RATIO
            rdf["buffer_months"] = config.OPT_BUFFER_MONTHS
            rdf["n_paths"] = config.OPT_N_PATHS
            refined_rows.append(rdf)

            # Combined search surface: integer-bp dedup only
            surf = merge_result_frames([cdf, rdf])
            star = pick_argmin(surf)
            cstar = float(star["cvar95"])
            nos = near_optimal_summary(surf, cstar, annual_E)
            ident = classify_identification(nos, cstar)

            w_star = np.array([star["w_kzt"], star["w_usd"], star["w_eur"]], dtype=float)
            ev_star = evaluate_weights(monthly, cf, W0, w_star)

            opt_rec = {
                "profile": profile.name,
                "horizon_years": years,
                "expense_income_ratio": config.OPT_EXPENSE_INCOME_RATIO,
                "buffer_months": config.OPT_BUFFER_MONTHS,
                "w_kzt": float(w_star[0]),
                "w_usd": float(w_star[1]),
                "w_eur": float(w_star[2]),
                "cvar95": ev_star.cvar95,
                "var95": ev_star.var95,
                "shortfall_probability": ev_star.shortfall_probability,
                "expected_cumulative_shortfall": ev_star.expected_cumulative_shortfall,
                "wealth_depletion_probability": ev_star.wealth_depletion_probability,
                "identification_class": ident,
                "boundary_solution": bool(
                    min(w_star) < 1e-8 or max(w_star) > 1 - 1e-8
                ),
                "n_coarse": int(len(cdf)),
                "n_refined": int(len(rdf)),
                "n_surface": int(len(surf)),
                **nos,
            }
            optimal_rows.append(opt_rec)
            horizon_rows.append(opt_rec)

            # Benchmarks + store losses
            losses = {"opt": ev_star.path_losses}
            for bname, bw in strategy_catalog(profile):
                bev = evaluate_weights(monthly, cf, W0, bw)
                losses[bname] = bev.path_losses
                b_cvar = bev.cvar95
                abs_red = b_cvar - ev_star.cvar95
                if b_cvar > config.OPT_CVAR_ZERO_TOL:
                    rel = 100.0 * abs_red / b_cvar
                else:
                    rel = np.nan
                pair = paired_path_comparison(ev_star.path_losses, bev.path_losses)
                bench_rows.append(
                    {
                        "profile": profile.name,
                        "horizon_years": years,
                        "optimized_w_kzt": float(w_star[0]),
                        "optimized_w_usd": float(w_star[1]),
                        "optimized_w_eur": float(w_star[2]),
                        "benchmark_name": bname,
                        "benchmark_w_kzt": float(bw[0]),
                        "benchmark_w_usd": float(bw[1]),
                        "benchmark_w_eur": float(bw[2]),
                        "optimized_cvar95": ev_star.cvar95,
                        "benchmark_cvar95": b_cvar,
                        "absolute_cvar_reduction": abs_red,
                        "relative_cvar_reduction_pct": rel,
                        "optimized_shortfall_probability": ev_star.shortfall_probability,
                        "benchmark_shortfall_probability": bev.shortfall_probability,
                        "optimized_expected_shortfall": ev_star.expected_cumulative_shortfall,
                        "benchmark_expected_shortfall": bev.expected_cumulative_shortfall,
                        "optimized_depletion_probability": ev_star.wealth_depletion_probability,
                        "benchmark_depletion_probability": bev.wealth_depletion_probability,
                        **{f"paired_{k}": v for k, v in pair.items()},
                    }
                )
            # Boundary near-optimal samples for uncertainty
            m1 = surf[
                surf["cvar95"]
                <= (
                    cstar * (1 + config.OPT_NEAR_OPTIMAL_REL_TOL_1PCT)
                    if cstar > config.OPT_CVAR_ZERO_TOL
                    else max(
                        config.OPT_NEAR_OPTIMAL_ABS_TOL,
                        config.OPT_NEAR_OPTIMAL_ABS_FRAC_ANNUAL * annual_E,
                    )
                )
                + 1e-15
            ]
            for label, col in (("near_max_usd", "w_usd"), ("near_max_eur", "w_eur"), ("near_max_kzt", "w_kzt")):
                if m1.empty:
                    break
                row = m1.sort_values(col, ascending=False).iloc[0]
                ww = np.array([row.w_kzt, row.w_usd, row.w_eur])
                losses[label] = evaluate_weights(monthly, cf, W0, ww).path_losses

            loss_store[(profile.name, years)] = losses
            cell_detail[(profile.name, years)] = {
                "coarse": cdf,
                "refined": rdf,
                "surface": surf,
                "opt": opt_rec,
                "w_star": w_star,
            }

    coarse_all = pd.concat(coarse_rows, ignore_index=True)
    refined_all = pd.concat(refined_rows, ignore_index=True)
    optimal = pd.DataFrame(optimal_rows)
    benches = pd.DataFrame(bench_rows)
    horizons = pd.DataFrame(horizon_rows)

    write_csv(coarse_all, config.OPT_TABLES / "coarse_grid_results.csv")
    write_csv(refined_all, config.OPT_TABLES / "refined_grid_results.csv")
    write_csv(optimal, config.OPT_TABLES / "optimal_allocations.csv")
    write_csv(benches, config.OPT_TABLES / "benchmark_comparison.csv")
    write_csv(horizons, config.OPT_TABLES / "horizon_allocation_comparison.csv")

    return {
        "coarse": coarse_all,
        "refined": refined_all,
        "optimal": optimal,
        "benchmarks": benches,
        "loss_store": loss_store,
        "cell_detail": cell_detail,
        "s0": s0,
    }


def run_uncertainty(loss_store: dict) -> pd.DataFrame:
    rows = []
    for (profile, years), losses in loss_store.items():
        boot = paired_bootstrap_cvar(losses)
        boot.insert(0, "horizon_years", years)
        boot.insert(0, "profile", profile)
        rows.append(boot)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    write_csv(out, config.OPT_TABLES / "optimization_uncertainty.csv")
    return out


def run_50k_confirmation(optimal: pd.DataFrame, s0: np.ndarray) -> pd.DataFrame:
    levels50, s0_50 = load_or_generate_fx_levels(
        n_paths=config.OPT_CONFIRM_N_PATHS,
        seed=config.OPT_CONFIRM_SEED,
        use_cache=False,
    )
    # s0 must match baseline last levels for cashflow normalization
    s0_use = s0
    rows = []
    years = config.OPT_CONFIRM_HORIZON_YEARS
    monthly = monthly_fx_for_horizon(levels50, years)
    for profile in PRIMARY_PROFILES:
        if profile.name not in config.OPT_CONFIRM_PROFILES:
            continue
        cf, E0, W0 = primary_cashflows(profile, s0_use)
        opt_row = optimal[
            (optimal.profile == profile.name) & (optimal.horizon_years == years)
        ].iloc[0]
        candidates = {
            "opt_10k": np.array([opt_row.w_kzt, opt_row.w_usd, opt_row.w_eur]),
        }
        for name, w in strategy_catalog(profile):
            candidates[name] = w
        # neighbors on 2% grid around opt
        for dk in (-0.02, 0.0, 0.02):
            for du in (-0.02, 0.0, 0.02):
                wk = float(np.clip(opt_row.w_kzt + dk, 0, 1))
                wu = float(np.clip(opt_row.w_usd + du, 0, 1))
                we = 1.0 - wk - wu
                if we < -1e-10:
                    continue
                we = max(we, 0.0)
                s = wk + wu + we
                candidates[f"nb_{wk:.2f}_{wu:.2f}_{we:.2f}"] = np.array([wk, wu, we]) / s

        evals = {}
        for lab, w in candidates.items():
            ev = evaluate_weights(monthly, cf, W0, w)
            evals[lab] = ev
            rows.append(
                {
                    "profile": profile.name,
                    "horizon_years": years,
                    "n_paths": config.OPT_CONFIRM_N_PATHS,
                    "seed": config.OPT_CONFIRM_SEED,
                    "label": lab,
                    "w_kzt": float(w[0]),
                    "w_usd": float(w[1]),
                    "w_eur": float(w[2]),
                    "cvar95": ev.cvar95,
                    "shortfall_probability": ev.shortfall_probability,
                    "expected_cumulative_shortfall": ev.expected_cumulative_shortfall,
                }
            )
        # best among checked
        best_lab = min(evals, key=lambda k: (evals[k].cvar95, -evals[k].weights[0]))
        best = evals[best_lab]
        opt10 = evals["opt_10k"]
        rows.append(
            {
                "profile": profile.name,
                "horizon_years": years,
                "n_paths": config.OPT_CONFIRM_N_PATHS,
                "seed": config.OPT_CONFIRM_SEED,
                "label": "CONFIRMATION_SUMMARY",
                "w_kzt": float(best.weights[0]),
                "w_usd": float(best.weights[1]),
                "w_eur": float(best.weights[2]),
                "cvar95": best.cvar95,
                "shortfall_probability": best.shortfall_probability,
                "expected_cumulative_shortfall": best.expected_cumulative_shortfall,
                "best_label": best_lab,
                "opt10k_cvar95": opt10.cvar95,
                "cvar_diff_best_minus_opt10k": best.cvar95 - opt10.cvar95,
                "weights_changed": not np.allclose(best.weights, opt10.weights, atol=1e-8),
                "stable_flag": bool(
                    abs(best.cvar95 - opt10.cvar95)
                    <= max(1e-6, 0.05 * max(opt10.cvar95, 1e-12))
                    or best_lab == "opt_10k"
                ),
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, config.OPT_TABLES / "confirmation_50k.csv")
    return out


def make_figures(cell_detail: dict, optimal: pd.DataFrame, benches: pd.DataFrame) -> None:
    # Fig 1-2: risk surfaces P2/P4 3Y
    for key, stem, title in (
        (("P2_foreign_income_mismatch", 3), "fig01_cvar_surface_P2_3Y", "Figure 1. CVaR95 surface — P2, 3Y"),
        (("P4_full_multi_currency", 3), "fig02_cvar_surface_P4_3Y", "Figure 2. CVaR95 surface — P4, 3Y"),
    ):
        if key not in cell_detail:
            continue
        surf = cell_detail[key]["coarse"]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sc = ax.scatter(
            surf["w_usd"],
            surf["w_eur"],
            c=surf["cvar95"],
            s=18,
            cmap="viridis",
        )
        fig.colorbar(sc, ax=ax, label="CVaR95")
        ax.set_xlabel("w_USD")
        ax.set_ylabel("w_EUR")
        ax.set_title(title + "\n(w_KZT = 1 − w_USD − w_EUR)")
        # mark special points
        marks = {
            "opt": cell_detail[key]["w_star"],
            "KZT": np.array([1.0, 0.0, 0.0]),
            "USD": np.array([0.0, 1.0, 0.0]),
            "EUR": np.array([0.0, 0.0, 1.0]),
            "eq": np.array([1 / 3, 1 / 3, 1 / 3]),
        }
        for lab, w in marks.items():
            ax.scatter([w[1]], [w[2]], marker="x", s=80, label=lab)
        ax.legend(fontsize=8, loc="upper right")
        _save_fig(fig, stem)

    # Fig 3: optimal weights by profile 3Y
    o3 = optimal[optimal.horizon_years == 3]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(o3))
    ax.bar(x - 0.2, o3.w_kzt, 0.2, label="KZT")
    ax.bar(x, o3.w_usd, 0.2, label="USD")
    ax.bar(x + 0.2, o3.w_eur, 0.2, label="EUR")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in o3.profile], fontsize=8)
    ax.set_ylabel("Weight")
    ax.set_title("Figure 3. Exact argmin weights by profile (3Y)")
    ax.legend()
    _save_fig(fig, "fig03_optimal_weights_by_profile_3Y")

    # Fig 4: horizon weights
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, cur, col in zip(axes, ("KZT", "USD", "EUR"), ("w_kzt", "w_usd", "w_eur")):
        for prof, g in optimal.groupby("profile"):
            g = g.sort_values("horizon_years")
            ax.plot(g.horizon_years, g[col], marker="o", label=prof.split("_")[0])
        ax.set_title(cur)
        ax.set_xlabel("Horizon (years)")
        ax.set_xticks([1, 3, 5])
    axes[0].set_ylabel("Weight (exact argmin)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 4. Exact argmin weights by horizon")
    _save_fig(fig, "fig04_weights_by_horizon")

    # Fig 5-6 CVaR benchmarks P2/P4
    for prof, stem, title in (
        ("P2_foreign_income_mismatch", "fig05_cvar_benchmarks_P2", "Figure 5. CVaR benchmarks — P2"),
        ("P4_full_multi_currency", "fig06_cvar_benchmarks_P4", "Figure 6. CVaR benchmarks — P4"),
    ):
        g = benches[(benches.profile == prof) & (benches.horizon_years == 3)]
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = ["opt"] + list(g.benchmark_name)
        vals = [float(g.optimized_cvar95.iloc[0])] + list(g.benchmark_cvar95)
        ax.bar(labels, vals)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_ylabel("CVaR95")
        ax.set_title(title + " (3Y)")
        _save_fig(fig, stem)

    # Fig 7 shortfall probability
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, prof in zip(
        axes, ("P2_foreign_income_mismatch", "P4_full_multi_currency")
    ):
        g = benches[(benches.profile == prof) & (benches.horizon_years == 3)]
        if g.empty:
            continue
        labels = ["opt"] + list(g.benchmark_name)
        vals = [float(g.optimized_shortfall_probability.iloc[0])] + list(
            g.benchmark_shortfall_probability
        )
        ax.bar(labels, vals)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_title(prof.split("_")[0])
        ax.set_ylabel("P(shortfall>0)")
    fig.suptitle("Figure 7. Shortfall probability — P2 & P4 (3Y)")
    _save_fig(fig, "fig07_shortfall_prob_benchmarks")


def write_report(
    optimal: pd.DataFrame,
    benches: pd.DataFrame,
    uncertainty: pd.DataFrame,
    confirm: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 2B2 CVaR Currency Allocation Optimization Report",
        "",
        f"Environment: E/I={config.OPT_EXPENSE_INCOME_RATIO}, B={config.OPT_BUFFER_MONTHS}, "
        f"I0={config.OPT_I0}, W0={config.OPT_BUFFER_MONTHS * config.OPT_EXPENSE_INCOME_RATIO * config.OPT_I0}.",
        f"DGP: joint stationary bootstrap L={config.OPT_BLOCK_LENGTH}, "
        f"N={config.OPT_N_PATHS}, seed={config.OPT_RANDOM_SEED}.",
        f"Coarse Δw={config.OPT_COARSE_STEP} ({expected_grid_size(config.OPT_COARSE_STEP)} portfolios); "
        f"local refinement Δw={config.OPT_REFINED_STEP}.",
        f"CVaR95 = mean of largest k=ceil(0.05N) path losses (k=500 at N=10,000).",
        "",
        "Interpretation rule: report exact argmin **and** identification class / near-optimal region.",
        "",
        "## 1–3. Optima and identification",
        "",
        optimal[
            [
                "profile",
                "horizon_years",
                "w_kzt",
                "w_usd",
                "w_eur",
                "cvar95",
                "shortfall_probability",
                "identification_class",
                "boundary_solution",
                "near_optimal_1pct_count",
                "near_optimal_1pct_frac",
            ]
        ].to_string(index=False),
        "",
        "## 4–6. Composition and horizon",
        "",
        "See `horizon_allocation_comparison.csv` and Figure 3–4. "
        "If a cell is weakly/not identified, interpret weight ranges rather than a single point.",
        "",
        "## 7–8. Expense vs income matching and benchmark improvements",
        "",
    ]
    for prof in (
        "P2_foreign_income_mismatch",
        "P3_mixed_income",
        "P4_full_multi_currency",
    ):
        g = benches[(benches.profile == prof) & (benches.horizon_years == 3)]
        lines.append(f"### {prof} (3Y)")
        if g.empty:
            lines.append("No rows.")
            continue
        sub = g[g.benchmark_name.isin(["B5_expense_matched", "B6_income_matched", "B1_all_kzt", "B2_all_usd"])]
        # B5/B6 may be deduped into B1/B2
        lines.append(g[["benchmark_name", "benchmark_cvar95", "optimized_cvar95", "absolute_cvar_reduction", "relative_cvar_reduction_pct"]].to_string(index=False))
        lines.append("")

    lines.extend(
        [
            "## 9–10. Materiality and profile contrast",
            "",
            "Improvements are material when relative CVaR reduction is large **and** "
            "identification is not class C. P1 is a control: FX risk should be limited "
            "under all-KZT; optimization may be weakly identified if shortfall risk is tiny.",
            "",
            "## 11–12. Monte Carlo uncertainty and 50k confirmation",
            "",
            f"Bootstrap replicates: {config.OPT_BOOTSTRAP_REPLICATES} (paired path indices).",
            "",
        ]
    )
    if len(confirm):
        summ = confirm[confirm.label == "CONFIRMATION_SUMMARY"]
        lines.append(summ.to_string(index=False))
        lines.append("")

    # Pairwise clear rankings
    if len(uncertainty):
        diffs = uncertainty[uncertainty.get("kind", pd.Series(dtype=str)) == "paired_difference"] if "kind" in uncertainty.columns else uncertainty[uncertainty.comparison.str.contains("minus", na=False)]
        lines.append("### Paired bootstrap differences (opt − benchmark)")
        if len(diffs):
            lines.append(
                diffs[
                    [c for c in diffs.columns if c in (
                        "profile", "horizon_years", "comparison",
                        "point_estimate_cvar_difference", "bootstrap_ci_lower",
                        "bootstrap_ci_upper", "ranking_statistically_clear",
                    )]
                ].head(40).to_string(index=False)
            )
        lines.append("")

    lines.extend(
        [
            "## 13–14. Unidentified / indistinguishable cells",
            "",
            "See `identification_class` in optimal_allocations.csv. "
            "When class is C_NOT_IDENTIFIED or near-optimal fraction is large, "
            "do not claim a unique economic optimum.",
            "",
            "## 15. Strongest findings for Phase 2C robustness",
            "",
            "- Whether expense-matching vs income-matching ranking survives transaction costs / rebalancing frequency.",
            "- Stability of boundary vs interior solutions under alternative DGPs and calibration windows.",
            "- Whether thin buffer (B=1) conclusions hold under the Phase 2B1 alternative (E/I=1.05, B=3).",
            "",
            "## Research-question mapping",
            "",
            "**RQ1:** Report exact argmin with identification class and near-optimal ranges.",
            "**RQ2:** Compare P1–P4 optima/regions at fixed horizon.",
            "**RQ3:** Compare 1Y/3Y/5Y for each profile (ranges if weakly identified).",
            "**RQ4:** Use benchmark_comparison.csv absolute/relative CVaR reductions and paired bootstrap CIs.",
            "**RQ5:** Deferred to Phase 2C.",
            "",
        ]
    )
    (config.OPT_DIAGNOSTICS / "optimization_report.md").write_text("\n".join(lines), encoding="utf-8")


def optional_full_1pct_check(
    daily_levels: np.ndarray,
    s0: np.ndarray,
    cell_detail: dict,
) -> pd.DataFrame:
    """Optional complete 1% simplex for P2/P4 3Y as global check."""
    fine_bp = generate_simplex_grid(100)
    rows = []
    years = 3
    monthly = monthly_fx_for_horizon(daily_levels, years)
    for pname in config.OPT_CONFIRM_PROFILES:
        profile = next(p for p in PRIMARY_PROFILES if p.name == pname)
        cf, E0, W0 = primary_cashflows(profile, s0)
        df = evaluate_grid(monthly, cf, W0, fine_bp, grid_is_bp=True)
        star = pick_argmin(df)
        prev = cell_detail[(pname, years)]["opt"]
        rows.append(
            {
                "profile": pname,
                "horizon_years": years,
                "n_full_1pct": len(df),
                "full_1pct_w_kzt": float(star.w_kzt),
                "full_1pct_w_usd": float(star.w_usd),
                "full_1pct_w_eur": float(star.w_eur),
                "full_1pct_cvar95": float(star.cvar95),
                "refined_w_kzt": prev["w_kzt"],
                "refined_w_usd": prev["w_usd"],
                "refined_w_eur": prev["w_eur"],
                "refined_cvar95": prev["cvar95"],
                "weights_match": bool(
                    np.allclose(
                        [star.w_kzt, star.w_usd, star.w_eur],
                        [prev["w_kzt"], prev["w_usd"], prev["w_eur"]],
                        atol=0.011,
                    )
                ),
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, config.OPT_TABLES / "global_1pct_check_P2_P4_3Y.csv")
    return out


def run_phase2b2(run_full_1pct_check: bool = True) -> dict:
    _dirs()
    daily_levels, s0 = load_or_generate_fx_levels()
    # Verify Phase 2B1 benchmarks
    p2b1 = pd.read_csv(config.HH_TABLES / "stress_calibration.csv")
    repro = verify_phase2b1_benchmarks(daily_levels, s0, p2b1)
    write_csv(repro, config.OPT_DIAGNOSTICS / "phase2b1_reproduction_check.csv")

    pack = optimize_all_cells(daily_levels, s0)
    unc = run_uncertainty(pack["loss_store"])
    confirm = run_50k_confirmation(pack["optimal"], s0)
    if run_full_1pct_check:
        gcheck = optional_full_1pct_check(daily_levels, s0, pack["cell_detail"])
    else:
        gcheck = pd.DataFrame()
    make_figures(pack["cell_detail"], pack["optimal"], pack["benchmarks"])
    write_report(pack["optimal"], pack["benchmarks"], unc, confirm)

    summary = {
        "n_coarse_per_cell": expected_grid_size(config.OPT_COARSE_STEP),
        "n_optimal_rows": int(len(pack["optimal"])),
        "reproduction_ok": bool(repro["ok"].all()),
        "optimal": pack["optimal"].to_dict(orient="records"),
        "confirm_stable": (
            confirm[confirm.label == "CONFIRMATION_SUMMARY"]["stable_flag"].tolist()
            if len(confirm) and "stable_flag" in confirm.columns
            else []
        ),
        "global_1pct_check": gcheck.to_dict(orient="records") if len(gcheck) else [],
    }
    (config.OPT_DIAGNOSTICS / "phase2b2_run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
