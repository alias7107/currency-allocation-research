"""Map joint log-return paths to positive FX level paths."""

from __future__ import annotations

import numpy as np


def returns_to_levels(
    log_returns: np.ndarray,
    s0: np.ndarray,
) -> np.ndarray:
    """S_t = S_0 * exp(cumsum r).

    Parameters
    ----------
    log_returns : (n_paths, horizon, 2)
    s0 : (2,) initial USD/KZT and EUR/KZT levels

    Returns
    -------
    levels : (n_paths, horizon + 1, 2) including t=0
    """
    r = np.asarray(log_returns, dtype=float)
    s0 = np.asarray(s0, dtype=float).reshape(2)
    if r.ndim != 3 or r.shape[2] != 2:
        raise ValueError("log_returns must have shape (n_paths, horizon, 2)")
    n_paths, horizon, _ = r.shape
    levels = np.empty((n_paths, horizon + 1, 2), dtype=float)
    levels[:, 0, :] = s0
    levels[:, 1:, :] = s0 * np.exp(np.cumsum(r, axis=1))
    return levels


def implied_log_returns(levels: np.ndarray) -> np.ndarray:
    """Recover log(S_t / S_{t-1}) from a level path that includes t=0."""
    return np.log(levels[:, 1:, :] / levels[:, :-1, :])


def sanity_check_paths(levels: np.ndarray, log_returns: np.ndarray, atol: float = 1e-10) -> dict:
    implied = implied_log_returns(levels)
    return {
        "has_nan": bool(np.isnan(levels).any() or np.isnan(log_returns).any()),
        "has_inf": bool(np.isinf(levels).any() or np.isinf(log_returns).any()),
        "all_positive": bool(np.all(levels > 0)),
        "return_identity_ok": bool(np.allclose(implied, log_returns, atol=atol, equal_nan=False)),
        "max_abs_identity_error": float(np.max(np.abs(implied - log_returns))),
        "shape_levels": tuple(levels.shape),
        "shape_returns": tuple(log_returns.shape),
        "n_paths": int(levels.shape[0]),
        "horizon_days": int(log_returns.shape[1]),
    }
