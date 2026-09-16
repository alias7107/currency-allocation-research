"""Near-optimal sets and identification classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def near_optimal_mask(
    cvar: np.ndarray,
    cvar_star: float,
    rel_tol: float,
    annual_expenses: float,
) -> np.ndarray:
    """Mask of allocations within relative (or absolute if CVaR≈0) of minimum."""
    c = np.asarray(cvar, dtype=float)
    if not np.isfinite(cvar_star):
        return np.zeros(len(c), dtype=bool)
    if cvar_star <= config.OPT_CVAR_ZERO_TOL:
        abs_tol = max(
            config.OPT_NEAR_OPTIMAL_ABS_TOL,
            config.OPT_NEAR_OPTIMAL_ABS_FRAC_ANNUAL * annual_expenses,
        )
        return c <= abs_tol + 1e-15
    return c <= cvar_star * (1.0 + rel_tol) + 1e-15


def near_optimal_summary(
    df: pd.DataFrame,
    cvar_star: float,
    annual_expenses: float,
) -> dict:
    m1 = near_optimal_mask(
        df["cvar95"].to_numpy(),
        cvar_star,
        config.OPT_NEAR_OPTIMAL_REL_TOL_1PCT,
        annual_expenses,
    )
    m5 = near_optimal_mask(
        df["cvar95"].to_numpy(),
        cvar_star,
        config.OPT_NEAR_OPTIMAL_REL_TOL_5PCT,
        annual_expenses,
    )
    s1 = df.loc[m1]
    s5 = df.loc[m5]

    def rng(sub: pd.DataFrame, col: str) -> tuple[float, float]:
        if sub.empty:
            return (np.nan, np.nan)
        return float(sub[col].min()), float(sub[col].max())

    k1 = rng(s1, "w_kzt")
    u1 = rng(s1, "w_usd")
    e1 = rng(s1, "w_eur")
    k5 = rng(s5, "w_kzt")
    u5 = rng(s5, "w_usd")
    e5 = rng(s5, "w_eur")
    return {
        "near_optimal_1pct_count": int(m1.sum()),
        "near_optimal_1pct_frac": float(m1.mean()),
        "near_optimal_1pct_kzt_min": k1[0],
        "near_optimal_1pct_kzt_max": k1[1],
        "near_optimal_1pct_usd_min": u1[0],
        "near_optimal_1pct_usd_max": u1[1],
        "near_optimal_1pct_eur_min": e1[0],
        "near_optimal_1pct_eur_max": e1[1],
        "near_optimal_5pct_count": int(m5.sum()),
        "near_optimal_5pct_frac": float(m5.mean()),
        "near_optimal_5pct_kzt_min": k5[0],
        "near_optimal_5pct_kzt_max": k5[1],
        "near_optimal_5pct_usd_min": u5[0],
        "near_optimal_5pct_usd_max": u5[1],
        "near_optimal_5pct_eur_min": e5[0],
        "near_optimal_5pct_eur_max": e5[1],
        "cvar_zero_flag": bool(cvar_star <= config.OPT_CVAR_ZERO_TOL),
    }


def classify_identification(summary: dict, cvar_star: float) -> str:
    """A WELL / B WEAKLY / C NOT identified."""
    frac1 = summary["near_optimal_1pct_frac"]
    frac5 = summary["near_optimal_5pct_frac"]
    if cvar_star <= config.OPT_CVAR_ZERO_TOL and frac5 >= config.OPT_NOT_IDENTIFIED_MIN_FRAC:
        return "C_NOT_IDENTIFIED"
    if frac1 >= config.OPT_NOT_IDENTIFIED_MIN_FRAC:
        return "C_NOT_IDENTIFIED"
    if frac1 <= config.OPT_WELL_IDENTIFIED_MAX_FRAC and cvar_star > config.OPT_CVAR_ZERO_TOL:
        return "A_WELL_IDENTIFIED"
    return "B_WEAKLY_IDENTIFIED"


def pick_argmin(df: pd.DataFrame) -> pd.Series:
    """Deterministic argmin: lowest CVaR, then max KZT, then max USD (tie-break only)."""
    work = df.sort_values(
        ["cvar95", "w_kzt", "w_usd", "w_eur"],
        ascending=[True, False, False, False],
    )
    return work.iloc[0]
