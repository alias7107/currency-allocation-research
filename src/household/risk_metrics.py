"""Path-level household loss and empirical CVaR (no optimization)."""

from __future__ import annotations

import numpy as np

from src import config


def cumulative_shortfall(monthly_shortfall: np.ndarray) -> np.ndarray:
    return np.sum(monthly_shortfall, axis=1)


def max_monthly_shortfall(monthly_shortfall: np.ndarray) -> np.ndarray:
    return np.max(monthly_shortfall, axis=1)


def any_shortfall(monthly_shortfall: np.ndarray) -> np.ndarray:
    return (np.max(monthly_shortfall, axis=1) > 0).astype(float)


def _tail_k(n: int, alpha: float) -> int:
    raw = (1.0 - alpha) * n
    k = int(np.ceil(raw - 1e-12))
    return max(k, 1)


def empirical_cvar(losses: np.ndarray, alpha: float | None = None) -> float:
    """Mean of the largest k = ceil((1-alpha) N) path losses (descending sort).

    For N=10,000 and alpha=0.95, k=500. Point mass at zero is handled by this
    order-statistic definition; no kernel smoothing.
    """
    alpha = config.HH_CVAR_ALPHA if alpha is None else alpha
    x = np.asarray(losses, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan")
    tail = np.sort(x)[::-1][: _tail_k(n, alpha)]
    return float(np.mean(tail))


def empirical_var(losses: np.ndarray, alpha: float | None = None) -> float:
    """k-th largest loss, k = ceil((1-alpha) N) (same k as CVaR)."""
    alpha = config.HH_CVAR_ALPHA if alpha is None else alpha
    x = np.asarray(losses, dtype=float)
    n = len(x)
    k = _tail_k(n, alpha)
    return float(np.sort(x)[::-1][k - 1])


def summarize_run(
    monthly_shortfall: np.ndarray,
    terminal_wealth: np.ndarray,
    E0: float,
    n_months: int,
) -> dict:
    L = cumulative_shortfall(monthly_shortfall)
    mx = max_monthly_shortfall(monthly_shortfall)
    annual_exp = E0 * 12.0
    depleted = terminal_wealth <= config.HH_WEALTH_DEPLETE_TOL
    return {
        "n_paths": int(len(L)),
        "shortfall_probability": float(np.mean(L > 0)),
        "expected_cumulative_shortfall": float(np.mean(L)),
        "var95_cumulative_shortfall": empirical_var(L),
        "cvar95_cumulative_shortfall": empirical_cvar(L),
        "mean_max_monthly_shortfall": float(np.mean(mx)),
        "wealth_depletion_probability": float(np.mean(depleted)),
        "mean_terminal_wealth": float(np.mean(terminal_wealth)),
        "median_terminal_wealth": float(np.median(terminal_wealth)),
        "p05_terminal_wealth": float(np.quantile(terminal_wealth, 0.05)),
        "p95_terminal_wealth": float(np.quantile(terminal_wealth, 0.95)),
        "min_terminal_wealth": float(np.min(terminal_wealth)),
        "max_terminal_wealth": float(np.max(terminal_wealth)),
        "expected_shortfall_over_annual_expenses": float(np.mean(L) / annual_exp) if annual_exp else np.nan,
        "cvar95_over_annual_expenses": empirical_cvar(L) / annual_exp if annual_exp else np.nan,
        "horizon_months": int(n_months),
    }
