"""Paired bootstrap over simulation path indices (MC uncertainty)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.household.risk_metrics import empirical_cvar


def paired_bootstrap_cvar(
    loss_matrix: dict[str, np.ndarray],
    n_boot: int | None = None,
    seed: int | None = None,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Bootstrap CVaR and paired differences using common path indices.

    ``loss_matrix`` maps label -> path losses of shape (N,).
    """
    n_boot = n_boot or config.OPT_BOOTSTRAP_REPLICATES
    seed = seed if seed is not None else config.OPT_BOOTSTRAP_SEED
    alpha = alpha or config.OPT_CVAR_ALPHA
    labels = list(loss_matrix.keys())
    if not labels:
        return pd.DataFrame()
    n = len(next(iter(loss_matrix.values())))
    for v in loss_matrix.values():
        if len(v) != n:
            raise ValueError("all loss vectors must share length N")
    rng = np.random.default_rng(seed)
    boot_cvar = {lab: np.empty(n_boot) for lab in labels}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for lab in labels:
            boot_cvar[lab][b] = empirical_cvar(loss_matrix[lab][idx], alpha)
    rows = []
    for lab in labels:
        x = boot_cvar[lab]
        pe = empirical_cvar(loss_matrix[lab], alpha)
        rows.append(
            {
                "comparison": lab,
                "point_estimate_cvar": pe,
                "bootstrap_mean_cvar": float(np.mean(x)),
                "bootstrap_ci_lower": float(np.quantile(x, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(x, 0.975)),
                "bootstrap_replicates": n_boot,
                "kind": "level",
            }
        )
    # Paired differences vs first label treated as reference if named 'opt'
    ref = "opt" if "opt" in loss_matrix else labels[0]
    for lab in labels:
        if lab == ref:
            continue
        d = boot_cvar[ref] - boot_cvar[lab]
        pe = empirical_cvar(loss_matrix[ref], alpha) - empirical_cvar(loss_matrix[lab], alpha)
        lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
        # ranking clear if CI for (opt - bench) does not contain 0 when we care about opt better (negative)
        clear = (hi < 0) or (lo > 0)
        rows.append(
            {
                "comparison": f"{ref}_minus_{lab}",
                "point_estimate_cvar": pe,
                "point_estimate_cvar_difference": pe,
                "bootstrap_mean_cvar": float(np.mean(d)),
                "bootstrap_ci_lower": lo,
                "bootstrap_ci_upper": hi,
                "bootstrap_replicates": n_boot,
                "ranking_statistically_clear": bool(clear),
                "kind": "paired_difference",
            }
        )
    return pd.DataFrame(rows)


def paired_path_comparison(opt_losses: np.ndarray, bench_losses: np.ndarray) -> dict:
    d = np.asarray(opt_losses, dtype=float) - np.asarray(bench_losses, dtype=float)
    return {
        "mean_D": float(np.mean(d)),
        "median_D": float(np.median(d)),
        "P_D_lt_0": float(np.mean(d < 0)),
        "P_D_gt_0": float(np.mean(d > 0)),
        "P_D_eq_0": float(np.mean(np.isclose(d, 0.0))),
    }
