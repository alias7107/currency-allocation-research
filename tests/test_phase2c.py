"""Tests for Phase 2C grid integrity, settlement modes, and robustness helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.household.cashflows import nominal_cashflows
from src.household.profiles import P2_FOREIGN_INCOME, P4_MULTI_CURRENCY
from src.household.settlement import simulate_household
from src.optimization.grid import (
    TOTAL_BP,
    assert_unique_bp_keys,
    expected_grid_size_bp,
    generate_simplex_grid,
    local_refined_grid_bp,
    merge_bp_grids,
    merge_result_frames,
    weights_to_bp,
)
from src.robustness.classify import classify_robustness, final_paper_classification
from src.robustness.scenarios import generate_fx_levels
from src.simulation.validation import load_baseline_bundle


def test_grid_2pct_count_1326():
    g = generate_simplex_grid(200)
    assert len(g) == 1326
    assert expected_grid_size_bp(200) == 1326
    assert np.all(g.sum(axis=1) == TOTAL_BP)
    assert np.all(g >= 0)
    assert_unique_bp_keys(g)


def test_grid_1pct_count_5151():
    g = generate_simplex_grid(100)
    assert len(g) == 5151
    assert expected_grid_size_bp(100) == 5151
    assert np.all(g.sum(axis=1) == TOTAL_BP)
    assert_unique_bp_keys(g)


def test_grid_boundaries_and_no_float_keys():
    g = generate_simplex_grid(200)
    keys = {(int(a), int(b), int(c)) for a, b, c in g}
    assert (10000, 0, 0) in keys
    assert (0, 10000, 0) in keys
    assert (0, 0, 10000) in keys
    assert (5000, 5000, 0) in keys


def test_coarse_refined_dedup_integer():
    coarse = generate_simplex_grid(200)
    # Fake cvar: all-KZT best
    cvar = np.ones(len(coarse))
    idx = np.where((coarse[:, 0] == 10000) & (coarse[:, 1] == 0))[0][0]
    cvar[idx] = 0.0
    fine = local_refined_grid_bp(coarse, cvar, 100, band=0.02, radius_bp=500)
    merged = merge_bp_grids(coarse, fine)
    assert_unique_bp_keys(merged)
    assert (10000, 0, 0) in {(int(a), int(b), int(c)) for a, b, c in merged}


def test_merge_result_frames_rejects_float_dup_inflation():
    bp = generate_simplex_grid(200)[:5]
    rows = []
    for r in bp:
        rows.append(
            {
                "w_kzt_bp": int(r[0]),
                "w_usd_bp": int(r[1]),
                "w_eur_bp": int(r[2]),
                "w_kzt": r[0] / TOTAL_BP,
                "w_usd": r[1] / TOTAL_BP,
                "w_eur": r[2] / TOTAL_BP,
                "cvar95": 1.0,
            }
        )
    df1 = pd.DataFrame(rows)
    df2 = df1.copy()
    # Intentionally perturb float columns but keep bp keys identical
    df2["w_kzt"] = df2["w_kzt"] + 1e-16
    merged = merge_result_frames([df1, df2])
    assert len(merged) == 5


def test_zero_cost_monthly_matches_default_api():
    rng = np.random.default_rng(1)
    fx = np.stack(
        [
            400 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(4, 12)), axis=1)),
            500 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(4, 12)), axis=1)),
        ],
        axis=-1,
    )
    cf = nominal_cashflows(P2_FOREIGN_INCOME, 1.0, 0.95, 400.0, 500.0)
    w = np.array([1.0, 0.0, 0.0])
    a = simulate_household(fx, cf, w, 0.95)
    b = simulate_household(
        fx, cf, w, 0.95, rebalance_frequency="monthly", conversion_cost=0.0
    )
    assert np.allclose(a.monthly_shortfall, b.monthly_shortfall)
    assert np.allclose(a.terminal_wealth, b.terminal_wealth)
    assert a.max_conservation_error < 1e-8


def test_quarterly_rebalance_schedule_and_none():
    rng = np.random.default_rng(2)
    fx = np.stack(
        [
            400 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(3, 12)), axis=1)),
            500 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(3, 12)), axis=1)),
        ],
        axis=-1,
    )
    cf = nominal_cashflows(P4_MULTI_CURRENCY, 1.0, 0.95, 400.0, 500.0)
    w = np.array([0.5, 0.3, 0.2])
    m = simulate_household(fx, cf, w, 2.0, rebalance_frequency="monthly", conversion_cost=0.0)
    q = simulate_household(fx, cf, w, 2.0, rebalance_frequency="quarterly", conversion_cost=0.0)
    n = simulate_household(fx, cf, w, 2.0, rebalance_frequency="none", conversion_cost=0.0)
    assert not m.negative_holdings and not q.negative_holdings and not n.negative_holdings
    assert m.max_conservation_error < 1e-7
    assert q.max_conservation_error < 1e-7
    assert n.max_conservation_error < 1e-7
    # No-rebalance terminal holdings need not match target weights
    hk, hu, he = n.holdings_end
    S_u, S_e = fx[:, -1, 0], fx[:, -1, 1]
    W = hk + hu * S_u + he * S_e
    # At least one path should deviate from exact target if FX moved (typical)
    target_k = 0.5 * W
    assert np.any(np.abs(hk - target_k) > 1e-6) or np.allclose(fx[:, 0, :], fx[:, -1, :])


def test_conversion_cost_accounting():
    rng = np.random.default_rng(3)
    fx = np.stack(
        [
            400 * np.exp(np.cumsum(rng.normal(0, 0.015, size=(5, 24)), axis=1)),
            500 * np.exp(np.cumsum(rng.normal(0, 0.015, size=(5, 24)), axis=1)),
        ],
        axis=-1,
    )
    cf = nominal_cashflows(P2_FOREIGN_INCOME, 1.0, 0.95, 400.0, 500.0)
    w = np.array([0.2, 0.5, 0.3])
    z = simulate_household(fx, cf, w, 0.95, conversion_cost=0.0)
    p = simulate_household(fx, cf, w, 0.95, conversion_cost=0.01)
    assert np.all(z.cumulative_conversion_cost == 0)
    assert np.all(p.cumulative_conversion_cost >= -1e-12)
    assert float(np.mean(p.cumulative_conversion_cost)) > 0
    assert p.max_conservation_error < 1e-7
    assert not p.negative_holdings
    assert not p.has_nan


def test_recent_window_uses_only_2020_2025():
    bundle = load_baseline_bundle(config.ROB_RECENT_START, config.ROB_RECENT_END)
    assert np.min(bundle.dates) >= np.datetime64(config.ROB_RECENT_START)
    assert np.max(bundle.dates) <= np.datetime64(config.ROB_RECENT_END)
    levels, s0, meta = generate_fx_levels(
        kind="stationary_bootstrap",
        block=20,
        seed=7,
        n_paths=2,
        window_start=config.ROB_RECENT_START,
        window_end=config.ROB_RECENT_END,
    )
    assert meta["n_returns"] == len(bundle.returns)
    assert levels.shape[0] == 2


def test_sb56_block_length():
    from src.simulation.stationary_bootstrap import StationaryBootstrapSimulator

    sim = StationaryBootstrapSimulator(expected_block_length=56)
    assert sim.expected_block_length == 56


def test_robustness_classification_logic():
    w0 = np.array([1.0, 0.0, 0.0])
    assert (
        classify_robustness(
            w_base=w0,
            w_rob=w0,
            cvar_expense=1.0,
            cvar_income=1.2,
            cvar_opt=1.0,
            ident_class="A_WELL_IDENTIFIED",
            near_1pct_frac=0.02,
        )
        == "ROBUST_EXACT"
    )
    assert (
        classify_robustness(
            w_base=w0,
            w_rob=np.array([0.0, 1.0, 0.0]),
            cvar_expense=2.0,
            cvar_income=1.0,
            cvar_opt=1.0,
            ident_class="A_WELL_IDENTIFIED",
            near_1pct_frac=0.02,
        )
        == "REVERSED"
    )
    letter, _ = final_paper_classification(["ROBUST_EXACT", "ROBUST_EXACT"])
    assert letter == "A"
    letter_b, _ = final_paper_classification(["ROBUST_EXACT", "UNIDENTIFIED"])
    assert letter_b == "B"


def test_weights_to_bp_no_float_identity_keys():
    w = np.array([[0.14, 0.14, 0.72], [0.14000000000000001, 0.14, 0.72]])
    bp = weights_to_bp(w)
    assert tuple(bp[0]) == tuple(bp[1])
