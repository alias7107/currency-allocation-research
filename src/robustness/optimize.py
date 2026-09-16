"""Single-cell CVaR optimization using integer basis-point grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.household.cashflows import NominalCashFlows, nominal_cashflows
from src.household.profiles import HouseholdProfile
from src.optimization.evaluator import evaluate_grid, evaluate_weights
from src.optimization.grid import (
    TOTAL_BP,
    generate_simplex_grid,
    local_refined_grid_bp,
    merge_result_frames,
)
from src.optimization.near_optimal import (
    classify_identification,
    near_optimal_summary,
    pick_argmin,
)
from src.robustness.classify import (
    allocation_l1_distance,
    expense_matched_weights,
    income_matched_weights,
)


@dataclass
class CellOptResult:
    surface: pd.DataFrame
    coarse: pd.DataFrame
    refined: pd.DataFrame
    opt_row: pd.Series
    identification_class: str
    w_star: np.ndarray
    ev_star: object
    expense_cvar: float
    income_cvar: float
    expense_sf: float
    income_sf: float
    expense_w: np.ndarray
    income_w: np.ndarray


def make_cashflows(
    profile: HouseholdProfile,
    s0: np.ndarray,
    expense_income_ratio: float,
    buffer_months: float,
    I0: float = 1.0,
) -> tuple[NominalCashFlows, float, float]:
    E0 = expense_income_ratio * I0
    W0 = buffer_months * E0
    cf = nominal_cashflows(profile, I0, E0, float(s0[0]), float(s0[1]))
    return cf, E0, W0


def optimize_cell(
    monthly_fx: np.ndarray,
    profile: HouseholdProfile,
    s0: np.ndarray,
    *,
    expense_income_ratio: float,
    buffer_months: float,
    rebalance_frequency: str = "monthly",
    conversion_cost: float = 0.0,
    coarse_step_bp: int | None = None,
    refined_step_bp: int | None = None,
    radius_bp: int | None = None,
    full_1pct: bool = False,
) -> CellOptResult:
    coarse_step_bp = coarse_step_bp or config.ROB_COARSE_STEP_BP
    refined_step_bp = refined_step_bp or config.ROB_REFINED_STEP_BP
    radius_bp = radius_bp or config.ROB_REFINE_NEIGHBORHOOD_BP
    cf, E0, W0 = make_cashflows(profile, s0, expense_income_ratio, buffer_months)
    annual_E = E0 * 12.0

    kwargs = dict(
        rebalance_frequency=rebalance_frequency,
        conversion_cost=conversion_cost,
        grid_is_bp=True,
    )

    if full_1pct:
        fine_bp = generate_simplex_grid(refined_step_bp)
        surf = evaluate_grid(monthly_fx, cf, W0, fine_bp, **kwargs)
        coarse = surf
        refined = surf
    else:
        coarse_bp = generate_simplex_grid(coarse_step_bp)
        coarse = evaluate_grid(monthly_fx, cf, W0, coarse_bp, **kwargs)
        fine_bp = local_refined_grid_bp(
            coarse_bp,
            coarse["cvar95"].to_numpy(),
            refined_step_bp,
            config.ROB_REFINE_CVAR_BAND,
            radius_bp,
        )
        refined = evaluate_grid(monthly_fx, cf, W0, fine_bp, **kwargs)
        surf = merge_result_frames([coarse, refined])

    star = pick_argmin(surf)
    cstar = float(star["cvar95"])
    nos = near_optimal_summary(surf, cstar, annual_E)
    ident = classify_identification(nos, cstar)
    w_star = np.array([star["w_kzt"], star["w_usd"], star["w_eur"]], dtype=float)
    ev_star = evaluate_weights(
        monthly_fx,
        cf,
        W0,
        w_star,
        rebalance_frequency=rebalance_frequency,
        conversion_cost=conversion_cost,
    )

    w_exp = expense_matched_weights(profile)
    w_inc = income_matched_weights(profile)
    ev_exp = evaluate_weights(
        monthly_fx, cf, W0, w_exp,
        rebalance_frequency=rebalance_frequency,
        conversion_cost=conversion_cost,
    )
    ev_inc = evaluate_weights(
        monthly_fx, cf, W0, w_inc,
        rebalance_frequency=rebalance_frequency,
        conversion_cost=conversion_cost,
    )

    opt_row = pd.Series(
        {
            **star.to_dict(),
            "identification_class": ident,
            "n_surface": int(len(surf)),
            "n_coarse": int(len(coarse)),
            "n_refined": int(len(refined)),
            **nos,
            "expense_income_ratio": expense_income_ratio,
            "buffer_months": buffer_months,
            "rebalance_frequency": rebalance_frequency,
            "conversion_cost": conversion_cost,
        }
    )
    return CellOptResult(
        surface=surf,
        coarse=coarse,
        refined=refined,
        opt_row=opt_row,
        identification_class=ident,
        w_star=w_star,
        ev_star=ev_star,
        expense_cvar=ev_exp.cvar95,
        income_cvar=ev_inc.cvar95,
        expense_sf=ev_exp.shortfall_probability,
        income_sf=ev_inc.shortfall_probability,
        expense_w=w_exp,
        income_w=w_inc,
    )


def baseline_weights() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0])
