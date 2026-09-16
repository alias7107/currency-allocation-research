"""FX scenario generation for Phase 2C robustness dimensions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src import config
from src.simulation.iid_bootstrap import IIDBootstrapSimulator
from src.simulation.path_builder import returns_to_levels
from src.simulation.stationary_bootstrap import StationaryBootstrapSimulator
from src.simulation.validation import load_baseline_bundle


def generate_fx_levels(
    *,
    kind: str,
    block: int | None,
    seed: int,
    n_paths: int,
    window_start: str,
    window_end: str,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (daily_levels, s0, meta). Uses cache when path exists and matches."""
    bundle = load_baseline_bundle(window_start, window_end)
    meta = {
        "kind": kind,
        "block": block,
        "seed": seed,
        "n_paths": n_paths,
        "window_start": window_start,
        "window_end": window_end,
        "n_returns": int(len(bundle.returns)),
        "date_min": str(np.min(bundle.dates)) if bundle.dates is not None else None,
        "date_max": str(np.max(bundle.dates)) if bundle.dates is not None else None,
        "s0_usd": float(bundle.s0[0]),
        "s0_eur": float(bundle.s0[1]),
    }
    if cache_path is not None and cache_path.exists():
        levels = np.load(cache_path)
        if levels.shape[0] == n_paths:
            meta["from_cache"] = True
            return levels, bundle.s0, meta

    if kind == "stationary_bootstrap":
        if block is None:
            raise ValueError("block required for stationary_bootstrap")
        sim = StationaryBootstrapSimulator(expected_block_length=int(block))
        sim.fit(bundle.returns)
        r = sim.simulate(n_paths, config.HORIZON_5Y_DAYS, random_seed=seed)
    elif kind == "iid_bootstrap":
        sim = IIDBootstrapSimulator()
        sim.fit(bundle.returns)
        r = sim.simulate(n_paths, config.HORIZON_5Y_DAYS, random_seed=seed)
    else:
        raise ValueError(f"unknown DGP kind: {kind}")

    levels = returns_to_levels(r, bundle.s0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, levels)
    meta["from_cache"] = False
    return levels, bundle.s0, meta


def baseline_sb20_levels(n_paths: int | None = None, seed: int | None = None):
    """Prefer Phase 2B1 cache for exact baseline reproduction."""
    n_paths = n_paths or config.ROB_N_PATHS
    seed = seed if seed is not None else config.ROB_RANDOM_SEED
    cache = config.HH_DIR / "cache" / "sb20_daily_levels.npy"
    if (
        n_paths == config.HH_N_PATHS
        and seed == config.HH_RANDOM_SEED
        and cache.exists()
    ):
        levels = np.load(cache)
        bundle = load_baseline_bundle()
        # Prefer s0 from levels for cashflow consistency with Phase 2B1/2B2
        s0 = levels[0, 0, :].copy()
        meta = {
            "kind": "stationary_bootstrap",
            "block": 20,
            "seed": seed,
            "n_paths": n_paths,
            "window_start": config.BASELINE_START,
            "window_end": config.BASELINE_END,
            "from_cache": True,
            "n_returns": int(len(bundle.returns)),
        }
        return levels, s0, meta
    return generate_fx_levels(
        kind="stationary_bootstrap",
        block=20,
        seed=seed,
        n_paths=n_paths,
        window_start=config.BASELINE_START,
        window_end=config.BASELINE_END,
    )


def recent_window_block_diagnostic() -> dict:
    """Optional Politis–White-style note for 2020–2025 |r_USD| dependence."""
    try:
        from arch.utility.timeseries import add_trend  # noqa: F401
    except Exception:
        pass
    bundle = load_baseline_bundle(config.ROB_RECENT_START, config.ROB_RECENT_END)
    # Simple lag-1 autocorrelation heuristic on |r_usd| as a lightweight note
    x = np.abs(bundle.returns[:, 0])
    x = x - x.mean()
    if len(x) < 30:
        return {"n": len(x), "note": "sample too short"}
    ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    # Rough geometric block suggestion L ≈ (1+ρ)/(1-ρ) for AR(1)-like |r|
    if abs(ac1) < 0.999:
        L_hat = (1.0 + ac1) / (1.0 - ac1)
    else:
        L_hat = np.nan
    return {
        "window_start": config.ROB_RECENT_START,
        "window_end": config.ROB_RECENT_END,
        "n_returns": int(len(bundle.returns)),
        "abs_usd_ac1": ac1,
        "rough_L_hat_ar1": float(L_hat) if np.isfinite(L_hat) else None,
        "note": (
            "Diagnostic only; baseline robustness uses L=20 for the recent window. "
            "Not a Politis–White automatic bandwidth."
        ),
    }
