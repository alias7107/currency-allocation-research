"""Robustness classification labels and expense/income matching helpers."""

from __future__ import annotations

import numpy as np

from src import config
from src.household.profiles import HouseholdProfile


def expense_matched_weights(profile: HouseholdProfile) -> np.ndarray:
    return np.array([profile.qE_kzt, profile.qE_usd, profile.qE_eur], dtype=float)


def income_matched_weights(profile: HouseholdProfile) -> np.ndarray:
    return np.array([profile.qI_kzt, profile.qI_usd, profile.qI_eur], dtype=float)


def allocation_l1_distance(w_a: np.ndarray, w_b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(w_a) - np.asarray(w_b))))


def allocation_max_distance(w_a: np.ndarray, w_b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(w_a) - np.asarray(w_b))))


def classify_robustness(
    *,
    w_base: np.ndarray,
    w_rob: np.ndarray,
    cvar_expense: float,
    cvar_income: float,
    cvar_opt: float,
    ident_class: str,
    near_1pct_frac: float,
) -> str:
    """Return ROBUST_EXACT / ROBUST_MECHANISM / WEAKENED / REVERSED / UNIDENTIFIED."""
    if ident_class == "C_NOT_IDENTIFIED" or (
        cvar_opt <= config.OPT_CVAR_ZERO_TOL and near_1pct_frac >= config.OPT_NOT_IDENTIFIED_MIN_FRAC
    ):
        return "UNIDENTIFIED"

    exp_beats_inc = cvar_expense <= cvar_income + config.ROB_MECHANISM_CVAR_EDGE
    # Foreign-heavy / income-matched clearly better than expense-matched
    if cvar_income + config.ROB_MECHANISM_CVAR_EDGE < cvar_expense:
        # and income region preferred over baseline KZT-heavy opt distance
        if w_rob[0] < 0.5 and cvar_income < cvar_opt * 0.98:
            return "REVERSED"
        if not exp_beats_inc:
            return "REVERSED"

    max_d = allocation_max_distance(w_base, w_rob)
    if max_d <= config.ROB_EXACT_WEIGHT_TOL and exp_beats_inc:
        return "ROBUST_EXACT"

    # Mechanism: KZT/expense-aligned still near-optimal & expense beats income
    kzt_heavy = w_rob[0] >= 0.70
    if exp_beats_inc and (kzt_heavy or cvar_expense <= cvar_opt * 1.05 + 1e-12):
        # Weakened if edge is tiny relative to levels
        edge = cvar_income - cvar_expense
        scale = max(cvar_income, cvar_expense, 1e-12)
        if edge / scale < 0.02 and scale > config.OPT_CVAR_ZERO_TOL:
            return "WEAKENED"
        return "ROBUST_MECHANISM"

    if exp_beats_inc:
        return "WEAKENED"
    return "REVERSED"


def final_paper_classification(classes: list[str]) -> tuple[str, str]:
    """Map per-spec labels to RESULT A/B/C/D."""
    s = set(classes)
    if "REVERSED" in s:
        return (
            "D",
            "RESULT D: BASELINE RESULT REVERSED UNDER REASONABLE ROBUSTNESS TESTS",
        )
    non_uid = [c for c in classes if c != "UNIDENTIFIED"]
    # Unidentified cores weaken a claim of sharp all-KZT dominance
    if "UNIDENTIFIED" in s and non_uid and all(c == "ROBUST_EXACT" for c in non_uid):
        return (
            "B",
            "RESULT B: ROBUST EXPENSE-MATCHING / KZT-HEAVY PRINCIPLE, "
            "BUT EXACT WEIGHTS ARE MODEL-DEPENDENT",
        )
    if non_uid and all(c == "ROBUST_EXACT" for c in non_uid) and "UNIDENTIFIED" not in s:
        return (
            "A",
            "RESULT A: ROBUST ALL-KZT DOMINANCE",
        )
    if "WEAKENED" in s and "ROBUST_MECHANISM" not in s and "ROBUST_EXACT" not in s:
        return (
            "C",
            "RESULT C: HIGH MODEL/REGIME DEPENDENCE; NO STABLE ALLOCATION PRINCIPLE",
        )
    if any(c in s for c in ("ROBUST_MECHANISM", "ROBUST_EXACT", "WEAKENED")):
        return (
            "B",
            "RESULT B: ROBUST EXPENSE-MATCHING / KZT-HEAVY PRINCIPLE, "
            "BUT EXACT WEIGHTS ARE MODEL-DEPENDENT",
        )
    return (
        "C",
        "RESULT C: HIGH MODEL/REGIME DEPENDENCE; NO STABLE ALLOCATION PRINCIPLE",
    )
