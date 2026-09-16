"""Tests for Phase 2B2 optimization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.household.risk_metrics import empirical_cvar
from src.optimization.grid import (
    expected_grid_size,
    expected_grid_size_bp,
    generate_simplex_grid,
    local_refined_grid,
    simplex_grid,
)
from src.optimization.near_optimal import (
    classify_identification,
    near_optimal_mask,
    near_optimal_summary,
    pick_argmin,
)
from src.optimization.uncertainty import paired_bootstrap_cvar, paired_path_comparison


def test_coarse_grid_size_1326():
    g = simplex_grid(0.02)
    assert len(g) == 1326
    assert expected_grid_size(0.02) == 1326
    assert expected_grid_size_bp(200) == 1326
    assert len(generate_simplex_grid(200)) == 1326
    assert np.allclose(g.sum(axis=1), 1.0)
    assert np.all(g >= -1e-12)


def test_refined_grid_size_5151():
    assert expected_grid_size(0.01) == 5151
    assert expected_grid_size_bp(100) == 5151
    g = simplex_grid(0.01)
    assert len(g) == 5151
    assert len(generate_simplex_grid(100)) == 5151


def test_boundary_included():
    g = simplex_grid(0.02)
    corners = {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}
    found = {tuple(np.round(r, 10)) for r in g}
    for c in corners:
        assert c in found


def test_no_negative_weights():
    assert np.all(simplex_grid(0.05) >= -1e-12)


def test_empirical_cvar_k500_definition():
    x = np.arange(10_000.0)
    # largest 500 mean = mean(9999..9500)
    expected = np.mean(np.arange(9500, 10000))
    assert np.isclose(empirical_cvar(x, 0.95), expected)


def test_optimizer_finds_synthetic_minimum():
    # Synthetic: CVaR = (w_usd - 0.2)^2 + (w_eur - 0.1)^2
    g = simplex_grid(0.02)
    cvar = (g[:, 1] - 0.2) ** 2 + (g[:, 2] - 0.1) ** 2
    df = pd.DataFrame({"w_kzt": g[:, 0], "w_usd": g[:, 1], "w_eur": g[:, 2], "cvar95": cvar})
    star = pick_argmin(df)
    assert abs(star.w_usd - 0.2) < 0.021
    assert abs(star.w_eur - 0.1) < 0.021


def test_local_refinement_includes_boundary_center():
    coarse = simplex_grid(0.02)
    cvar = np.ones(len(coarse))
    # Make all-KZT uniquely best
    idx = np.where(np.isclose(coarse[:, 0], 1.0))[0][0]
    cvar[idx] = 0.0
    fine = local_refined_grid(coarse, cvar, 0.01, band=0.02, radius=0.05)
    assert np.any(np.isclose(fine[:, 0], 1.0) & np.isclose(fine[:, 1], 0.0))


def test_near_optimal_and_zero_cvar_handling():
    g = simplex_grid(0.1)
    df = pd.DataFrame(
        {"w_kzt": g[:, 0], "w_usd": g[:, 1], "w_eur": g[:, 2], "cvar95": np.zeros(len(g))}
    )
    s = near_optimal_summary(df, 0.0, annual_expenses=0.95 * 12)
    assert s["cvar_zero_flag"] is True
    assert s["near_optimal_1pct_count"] == len(g)
    assert classify_identification(s, 0.0) == "C_NOT_IDENTIFIED"


def test_near_optimal_relative():
    c = np.array([1.0, 1.005, 1.02, 2.0])
    m = near_optimal_mask(c, 1.0, 0.01, 10.0)
    assert m.tolist() == [True, True, False, False]


def test_paired_bootstrap_reproducible():
    rng = np.random.default_rng(0)
    a = rng.random(200)
    b = a + 0.1
    x1 = paired_bootstrap_cvar({"opt": a, "bench": b}, n_boot=50, seed=1)
    x2 = paired_bootstrap_cvar({"opt": a, "bench": b}, n_boot=50, seed=1)
    assert np.allclose(x1["bootstrap_ci_lower"], x2["bootstrap_ci_lower"])


def test_paired_path_comparison():
    opt = np.array([0.0, 0.0, 1.0])
    bench = np.array([0.0, 2.0, 0.5])
    d = paired_path_comparison(opt, bench)
    assert np.isclose(d["P_D_eq_0"], 1 / 3)
    assert np.isclose(d["P_D_lt_0"], 1 / 3)
    assert np.isclose(d["P_D_gt_0"], 1 / 3)


def test_benchmark_dedup_p1():
    from src.household.profiles import P1_LOCAL_CONTROL, strategy_catalog

    names = [n for n, _ in strategy_catalog(P1_LOCAL_CONTROL)]
    assert "B5_expense_matched" not in names
    assert "B6_income_matched" not in names
