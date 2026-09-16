"""Fit, simulate, validate, and score competing joint FX scenario generators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.simulation.iid_bootstrap import IIDBootstrapSimulator
from src.simulation.path_builder import returns_to_levels, sanity_check_paths
from src.simulation.stationary_bootstrap import (
    MovingBlockBootstrapSimulator,
    StationaryBootstrapSimulator,
    politis_white_mean_block_length,
)
from src.simulation.student_t import GaussianSimulator, StudentTSimulator
from src.simulation.validation import (
    CURRENCIES,
    HistoricalBundle,
    acf_values,
    dependence_stats,
    flatten_returns,
    historical_rolling_vol,
    historical_threshold_coexceed,
    load_baseline_bundle,
    mean_acf_from_paths,
    moments,
    overlapping_compound,
    rolling_vol_panel,
    tail_quantiles,
    vol_distribution_stats,
)
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


def _ensure_dirs() -> None:
    for p in (config.SIM_FIGURES, config.SIM_TABLES, config.SIM_DIAGNOSTICS):
        p.mkdir(parents=True, exist_ok=True)


def build_models(block_lengths=None) -> dict:
    block_lengths = block_lengths or config.STATIONARY_BLOCK_LENGTHS
    models = {
        "iid_bootstrap": IIDBootstrapSimulator(),
        "student_t": StudentTSimulator(),
        "gaussian": GaussianSimulator(),
    }
    for l in block_lengths:
        models[f"stationary_bootstrap_L{l}"] = StationaryBootstrapSimulator(expected_block_length=l)
    models["moving_block_L10"] = MovingBlockBootstrapSimulator(block_length=10)
    return models


def fit_all(models: dict, hist: np.ndarray) -> dict:
    for m in models.values():
        m.fit(hist)
    return models


def simulate_all(models: dict, n_paths: int, horizon: int, seed: int) -> dict:
    out = {}
    for name, m in models.items():
        out[name] = m.simulate(n_paths, horizon, random_seed=seed)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def convergence_table(hist: np.ndarray, seed: int) -> pd.DataFrame:
    """One-day draws; bootstrap models share the empirical one-day law."""
    iid = IIDBootstrapSimulator().fit(hist)
    tmod = StudentTSimulator().fit(hist)
    gmod = GaussianSimulator().fit(hist)
    rows = []
    truth = {
        "usd_mean": hist[:, 0].mean(),
        "usd_std": hist[:, 0].std(ddof=1),
        "usd_p1": np.quantile(hist[:, 0], 0.01),
        "usd_p5": np.quantile(hist[:, 0], 0.05),
        "eur_mean": hist[:, 1].mean(),
        "eur_std": hist[:, 1].std(ddof=1),
        "pearson": float(pd.Series(hist[:, 0]).corr(pd.Series(hist[:, 1]))),
        "usd_q005": np.quantile(hist[:, 0], 0.005),
    }
    for name, sim in (("iid_bootstrap", iid), ("student_t", tmod), ("gaussian", gmod)):
        for n in config.N_PATHS_CONVERGENCE:
            draws = sim.simulate(n, 1, random_seed=seed)[:, 0, :]
            rec = {"model": name, "n_paths": n}
            rec["usd_mean"] = draws[:, 0].mean()
            rec["usd_std"] = draws[:, 0].std(ddof=1)
            rec["usd_p1"] = np.quantile(draws[:, 0], 0.01)
            rec["usd_p5"] = np.quantile(draws[:, 0], 0.05)
            rec["eur_mean"] = draws[:, 1].mean()
            rec["eur_std"] = draws[:, 1].std(ddof=1)
            rec["pearson"] = float(pd.Series(draws[:, 0]).corr(pd.Series(draws[:, 1])))
            rec["usd_q005"] = np.quantile(draws[:, 0], 0.005)
            for k, v in truth.items():
                rec[f"{k}_hist"] = v
                rec[f"{k}_abs_err"] = abs(rec[k] - v)
            rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "convergence_analysis.csv")
    return table


def marginal_validation(hist: np.ndarray, sims: dict) -> pd.DataFrame:
    rows = []
    for col, cname in enumerate(CURRENCIES):
        h = moments(hist[:, col])
        for model, paths in sims.items():
            m = moments(flatten_returns(paths)[:, col])
            rec = {"model": model, "currency": cname}
            for k, hv in h.items():
                rec[f"hist_{k}"] = hv
                rec[f"sim_{k}"] = m[k]
                rec[f"abs_err_{k}"] = abs(m[k] - hv) if np.isfinite(m[k]) and np.isfinite(hv) else np.nan
                rec[f"rel_err_{k}"] = (
                    (m[k] - hv) / hv if np.isfinite(hv) and hv != 0 else np.nan
                )
            rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "marginal_validation.csv")
    return table


def tail_validation(hist: np.ndarray, sims: dict) -> pd.DataFrame:
    rows = []
    qs = config.TAIL_QUANTILES
    for col, cname in enumerate(CURRENCIES):
        hx = hist[:, col]
        h_q = {q: float(np.quantile(hx, q)) for q in qs}
        # historically relevant extreme thresholds: sample min / 2015-scale
        thr_low = float(np.min(hx))
        thr_p005 = h_q[0.005]
        for model, paths in sims.items():
            sx = flatten_returns(paths)[:, col]
            rec = {"model": model, "currency": cname}
            for q in qs:
                hq = h_q[q]
                sq = float(np.quantile(sx, q))
                rec[f"hist_q{q}"] = hq
                rec[f"sim_q{q}"] = sq
                rec[f"abs_err_q{q}"] = abs(sq - hq)
            rec["hist_min"] = float(np.min(hx))
            rec["sim_min"] = float(np.min(sx))
            rec["hist_max"] = float(np.max(hx))
            rec["sim_max"] = float(np.max(sx))
            rec["p_le_hist_min"] = float(np.mean(sx <= thr_low))
            rec["p_le_hist_q005"] = float(np.mean(sx <= thr_p005))
            rec["p_ge_hist_max"] = float(np.mean(sx >= np.max(hx)))
            rec["p_ge_hist_q995"] = float(np.mean(sx >= h_q[0.995]))
            rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "tail_validation.csv")
    return table


def dependence_validation(hist: np.ndarray, sims: dict) -> pd.DataFrame:
    hdep = dependence_stats(hist[:, 0], hist[:, 1])
    rows = []
    for model, paths in sims.items():
        flat = flatten_returns(paths)
        sdep = dependence_stats(flat[:, 0], flat[:, 1])
        # co-exceedance at historical thresholds
        thr = historical_threshold_coexceed(
            flat[:, 0],
            flat[:, 1],
            hdep["usd_q_low"],
            hdep["eur_q_low"],
            hdep["usd_q_high"],
            hdep["eur_q_high"],
        )
        rec = {"model": model}
        for k in ("pearson", "spearman", "lower_coexceed", "upper_coexceed"):
            rec[f"hist_{k}"] = hdep[k]
            rec[f"sim_{k}"] = sdep[k]
            rec[f"abs_err_{k}"] = abs(sdep[k] - hdep[k])
        rec["sim_lower_coexceed_hist_thr"] = thr["lower_coexceed_hist_thr"]
        rec["sim_upper_coexceed_hist_thr"] = thr["upper_coexceed_hist_thr"]
        rec["abs_err_lower_hist_thr"] = abs(thr["lower_coexceed_hist_thr"] - hdep["lower_coexceed"])
        rec["abs_err_upper_hist_thr"] = abs(thr["upper_coexceed_hist_thr"] - hdep["upper_coexceed"])
        rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "dependence_validation.csv")
    return table


def temporal_validation(hist: np.ndarray, sims: dict, nlags: int = 30) -> pd.DataFrame:
    rows = []
    for col, cname in enumerate(CURRENCIES):
        for transform, hx in (
            ("raw", hist[:, col]),
            ("abs", np.abs(hist[:, col])),
            ("sq", hist[:, col] ** 2),
        ):
            hacf = acf_values(hx, nlags=nlags)
            for model, paths in sims.items():
                sacf = mean_acf_from_paths(paths, col, transform if transform != "sq" else "sq", nlags)
                rec = {
                    "model": model,
                    "currency": cname,
                    "transform": transform,
                    "acf_rmse": float(np.sqrt(np.nanmean((sacf[1:] - hacf[1:]) ** 2))),
                }
                for lag in config.ACF_COMPARE_LAGS:
                    rec[f"hist_lag{lag}"] = float(hacf[lag])
                    rec[f"sim_lag{lag}"] = float(sacf[lag])
                    rec[f"abs_err_lag{lag}"] = abs(float(sacf[lag] - hacf[lag]))
                rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "temporal_dependence_validation.csv")
    return table


def vol_clustering_validation(hist: np.ndarray, sims: dict) -> pd.DataFrame:
    rows = []
    for col, cname in enumerate(CURRENCIES):
        for window in (30, 90):
            hvol = historical_rolling_vol(hist[:, col], window)
            hm = vol_distribution_stats(hvol)
            for model, paths in sims.items():
                svol = rolling_vol_panel(paths, window, col)
                sm = vol_distribution_stats(svol)
                rec = {"model": model, "currency": cname, "window": window}
                for k, hv in hm.items():
                    rec[f"hist_{k}"] = hv
                    rec[f"sim_{k}"] = sm[k]
                    rec[f"abs_err_{k}"] = abs(sm[k] - hv) if np.isfinite(sm[k]) and np.isfinite(hv) else np.nan
                rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "volatility_clustering_validation.csv")
    return table


def multihorizon_validation(hist: np.ndarray, sims: dict) -> pd.DataFrame:
    rows = []
    for h in config.MULTIHORIZON_WINDOWS:
        for col, cname in enumerate(CURRENCIES):
            hc = overlapping_compound(hist[:, col], h)
            hstats = {
                "mean": float(np.mean(hc)),
                "std": float(np.std(hc, ddof=1)),
                "q5": float(np.quantile(hc, 0.05)),
                "q1": float(np.quantile(hc, 0.01)),
                "q95": float(np.quantile(hc, 0.95)),
                "q99": float(np.quantile(hc, 0.99)),
            }
            for model, paths in sims.items():
                if paths.shape[1] < h:
                    continue
                # non-overlapping first-h days of each path (independent paths)
                sc = paths[:, :h, col].sum(axis=1)
                rec = {
                    "model": model,
                    "currency": cname,
                    "horizon_days": h,
                    "note": "historical stats use overlapping windows (dependent); simulated use independent paths",
                }
                sstats = {
                    "mean": float(np.mean(sc)),
                    "std": float(np.std(sc, ddof=1)),
                    "q5": float(np.quantile(sc, 0.05)),
                    "q1": float(np.quantile(sc, 0.01)),
                    "q95": float(np.quantile(sc, 0.95)),
                    "q99": float(np.quantile(sc, 0.99)),
                }
                for k in hstats:
                    rec[f"hist_{k}"] = hstats[k]
                    rec[f"sim_{k}"] = sstats[k]
                    rec[f"abs_err_{k}"] = abs(sstats[k] - hstats[k])
                rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "multihorizon_validation.csv")
    return table


def path_sanity_table(bundle: HistoricalBundle, sims: dict) -> pd.DataFrame:
    rows = []
    for model, r in sims.items():
        levels = returns_to_levels(r[: min(200, len(r))], bundle.s0)
        chk = sanity_check_paths(levels, r[: min(200, len(r))])
        chk["model"] = model
        chk["s0_usd"] = float(bundle.s0[0])
        chk["s0_eur"] = float(bundle.s0[1])
        rows.append(chk)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "path_sanity_checks.csv")
    return table


def student_t_fit_table(models: dict) -> pd.DataFrame:
    m = models["student_t"]
    p = m.params
    corr = p.covariance[0, 1] / np.sqrt(p.covariance[0, 0] * p.covariance[1, 1])
    scale_corr = p.scale[0, 1] / np.sqrt(p.scale[0, 0] * p.scale[1, 1])
    table = pd.DataFrame(
        [
            {
                "mu_usd": p.mu[0],
                "mu_eur": p.mu[1],
                "df": p.df,
                "scale_usd": p.scale[0, 0],
                "scale_eur": p.scale[1, 1],
                "scale_offdiag": p.scale[0, 1],
                "scale_correlation": scale_corr,
                "cov_usd": p.covariance[0, 0],
                "cov_eur": p.covariance[1, 1],
                "cov_offdiag": p.covariance[0, 1],
                "implied_pearson": corr,
                "nll": p.nll,
                "mle_success": p.success,
                "mle_message": p.message,
                "note": "Σ is the scale matrix; Cov = ν/(ν-2) Σ for ν>2",
            }
        ]
    )
    write_csv(table, config.SIM_TABLES / "student_t_parameters.csv")
    return table


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rmse(series) -> float:
    x = np.asarray(series, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def score_models(
    marginal: pd.DataFrame,
    tail: pd.DataFrame,
    dep: pd.DataFrame,
    temporal: pd.DataFrame,
    vol: pd.DataFrame,
    multi: pd.DataFrame,
    oos: pd.DataFrame | None,
) -> pd.DataFrame:
    """Equal-weight mean of category discrepancy scores (lower is better)."""
    models = sorted(marginal["model"].unique())
    rows = []
    for model in models:
        cat = {}
        m = marginal[marginal["model"] == model]
        cat["marginal"] = _rmse(m["rel_err_std"])
        t = tail[tail["model"] == model]
        qcols = [c for c in t.columns if c.startswith("abs_err_q")]
        cat["tail"] = _rmse(t[qcols].to_numpy().ravel())
        d = dep[dep["model"] == model]
        cat["dependence"] = _rmse(d[["abs_err_pearson", "abs_err_spearman", "abs_err_lower_hist_thr", "abs_err_upper_hist_thr"]].to_numpy().ravel())
        tm = temporal[temporal["model"] == model]
        cat["temporal"] = float(tm["acf_rmse"].mean())
        v = vol[vol["model"] == model]
        cat["vol_clustering"] = _rmse(v["abs_err_mean"])
        mh = multi[multi["model"] == model]
        cat["multihorizon"] = _rmse(mh["abs_err_std"])
        if oos is not None and len(oos) and "model" in oos.columns:
            o = oos[oos["model"] == model]
            if len(o) and "abs_err_std" in o.columns:
                cat["pseudo_oos"] = _rmse(o["abs_err_std"])
            else:
                cat["pseudo_oos"] = np.nan
        else:
            cat["pseudo_oos"] = np.nan
        rec = {"model": model, **cat}
        vals = [cat[k] for k in config.COMPARISON_CATEGORIES if np.isfinite(cat.get(k, np.nan))]
        rec["overall_equal_weight"] = float(np.mean(vals)) if vals else np.nan
        rec["weighting"] = "unweighted mean of available category RMSE/abs-error scores; lower is better"
        rows.append(rec)
    table = pd.DataFrame(rows).sort_values("overall_equal_weight")
    write_csv(table, config.SIM_TABLES / "model_scores.csv")
    return table


# ---------------------------------------------------------------------------
# Pseudo-OOS and sensitivity
# ---------------------------------------------------------------------------

def pseudo_oos_table(seed: int) -> pd.DataFrame:
    train = load_baseline_bundle(config.OOS_TRAIN_START, config.OOS_TRAIN_END)
    evalb = load_baseline_bundle(config.OOS_EVAL_START, config.OOS_EVAL_END)
    models = build_models()
    fit_all(models, train.returns)
    sims = simulate_all(models, n_paths=config.N_PATHS_PRIMARY, horizon=config.HORIZON_1Y_DAYS, seed=seed)
    # Compare simulated one-year / one-day properties to EVALUATION sample
    rows = []
    for col, cname in enumerate(CURRENCIES):
        hm = moments(evalb.returns[:, col])
        for name, paths in sims.items():
            sm = moments(flatten_returns(paths)[:, col])
            rec = {
                "exercise": "pseudo_oos_distributional",
                "train": f"{config.OOS_TRAIN_START}:{config.OOS_TRAIN_END}",
                "eval": f"{config.OOS_EVAL_START}:{config.OOS_EVAL_END}",
                "model": name,
                "currency": cname,
            }
            for k in ("std", "p5", "p1", "p95", "p99", "mean"):
                rec[f"eval_{k}"] = hm[k]
                rec[f"sim_{k}"] = sm[k]
                rec[f"abs_err_{k}"] = abs(sm[k] - hm[k])
            rows.append(rec)
    # dependence vs eval
    hdep = dependence_stats(evalb.returns[:, 0], evalb.returns[:, 1])
    for name, paths in sims.items():
        flat = flatten_returns(paths)
        sdep = dependence_stats(flat[:, 0], flat[:, 1])
        rec = {
            "exercise": "pseudo_oos_dependence",
            "train": f"{config.OOS_TRAIN_START}:{config.OOS_TRAIN_END}",
            "eval": f"{config.OOS_EVAL_START}:{config.OOS_EVAL_END}",
            "model": name,
            "currency": "joint",
            "eval_std": hdep["pearson"],
            "sim_std": sdep["pearson"],
            "abs_err_std": abs(sdep["pearson"] - hdep["pearson"]),
            "eval_p5": hdep["spearman"],
            "sim_p5": sdep["spearman"],
            "abs_err_p5": abs(sdep["spearman"] - hdep["spearman"]),
        }
        rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "pseudo_oos_validation.csv")
    return table


def shock_sensitivity_table(seed: int) -> pd.DataFrame:
    full = load_baseline_bundle(exclude_shock=False)
    noshock = load_baseline_bundle(exclude_shock=True)
    rows = []
    for label, bundle in (("baseline", full), ("exclude_float_shock_2015_08_20_to_31", noshock)):
        t = StudentTSimulator().fit(bundle.returns)
        iid = IIDBootstrapSimulator().fit(bundle.returns)
        sb = StationaryBootstrapSimulator(10).fit(bundle.returns)
        rec = {
            "sample": label,
            "n": int(len(bundle.returns)),
            "usd_std": float(bundle.returns[:, 0].std(ddof=1)),
            "eur_std": float(bundle.returns[:, 1].std(ddof=1)),
            "usd_skew": float(pd.Series(bundle.returns[:, 0]).skew()),
            "usd_exkurt": float(pd.Series(bundle.returns[:, 0]).kurtosis()),
            "usd_p1": float(np.quantile(bundle.returns[:, 0], 0.01)),
            "usd_min": float(bundle.returns[:, 0].min()),
            "student_t_df": t.params.df,
            "student_t_scale_corr": float(
                t.params.scale[0, 1] / np.sqrt(t.params.scale[0, 0] * t.params.scale[1, 1])
            ),
        }
        # small sim for tail
        for name, sim in (("iid_bootstrap", iid), ("stationary_bootstrap_L10", sb), ("student_t", t)):
            draws = sim.simulate(config.N_PATHS_PRIMARY, 1, random_seed=seed)[:, 0, 0]
            rec[f"{name}_usd_p1"] = float(np.quantile(draws, 0.01))
            rec[f"{name}_usd_q005"] = float(np.quantile(draws, 0.005))
        rows.append(rec)
    table = pd.DataFrame(rows)
    write_csv(table, config.SIM_TABLES / "float_shock_sensitivity.csv")
    return table


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(config.SIM_FIGURES / f"{stem}.png")
    fig.savefig(config.SIM_FIGURES / f"{stem}.pdf")
    plt.close(fig)


def make_figures(hist: np.ndarray, sims: dict, bundle: HistoricalBundle) -> None:
    # A, B histograms
    for col, cname, stem, title in (
        (0, "USD/KZT", "figA_usd_return_distributions", "Figure A. USD/KZT daily log returns"),
        (1, "EUR/KZT", "figB_eur_return_distributions", "Figure B. EUR/KZT daily log returns"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        hx = hist[:, col]
        ax.hist(hx, bins=60, density=True, alpha=0.45, label="historical")
        for model in ("iid_bootstrap", "stationary_bootstrap_L10", "student_t", "gaussian"):
            if model not in sims:
                continue
            sx = flatten_returns(sims[model])[:, col]
            ax.hist(sx, bins=60, density=True, histtype="step", label=model)
        ax.set_title(title)
        ax.legend(fontsize=8)
        _save(fig, stem)

    # C, D QQ vs historical empirical (not normal)
    for col, stem, title in (
        (0, "figC_usd_qq", "Figure C. USD/KZT simulated vs historical quantiles"),
        (1, "figD_eur_qq", "Figure D. EUR/KZT simulated vs historical quantiles"),
    ):
        hx = np.sort(hist[:, col])
        p = (np.arange(1, len(hx) + 1) - 0.5) / len(hx)
        fig, ax = plt.subplots(figsize=(6, 6))
        for model in ("iid_bootstrap", "stationary_bootstrap_L10", "student_t", "gaussian"):
            if model not in sims:
                continue
            sq = np.quantile(flatten_returns(sims[model])[:, col], p)
            ax.plot(hx, sq, label=model, linewidth=1)
        ax.plot(hx, hx, linestyle="--", linewidth=1, label="y = x")
        ax.set_xlabel("Historical quantile")
        ax.set_ylabel("Simulated quantile")
        ax.set_title(title)
        ax.legend(fontsize=8)
        _save(fig, stem)

    # E, F squared ACF
    lags = np.arange(0, 31)
    for col, stem, title in (
        (0, "figE_usd_sq_acf", "Figure E. USD/KZT ACF of squared returns"),
        (1, "figF_eur_sq_acf", "Figure F. EUR/KZT ACF of squared returns"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(lags, acf_values(hist[:, col] ** 2, 30), label="historical")
        for model in ("iid_bootstrap", "stationary_bootstrap_L5", "stationary_bootstrap_L10",
                      "stationary_bootstrap_L20", "student_t", "gaussian"):
            if model not in sims:
                continue
        ax.plot(lags, mean_acf_from_paths(sims[model][: config.N_PATHS_TEMPORAL], col, "sq", 30), label=model, linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Lag")
        ax.legend(fontsize=8)
        _save(fig, stem)

    # G rolling vol distributions
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, window in zip(axes, (30, 90)):
        data = [historical_rolling_vol(hist[:, 0], window)]
        labels = ["historical"]
        for model in ("iid_bootstrap", "stationary_bootstrap_L10", "student_t", "gaussian"):
            if model not in sims:
                continue
            data.append(rolling_vol_panel(sims[model][: config.N_PATHS_TEMPORAL], window, 0))
            labels.append(model)
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(f"{window}-day ann. USD vol")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle("Figure G. Rolling-volatility distributions")
    _save(fig, "figG_rolling_vol_distributions")

    # H joint scatter (subsample)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
    rng = np.random.default_rng(config.SIM_EXAMPLE_PATH_SEED)
    panels = [("historical", hist)]
    for model in ("stationary_bootstrap_L10", "student_t"):
        if model in sims:
            flat = flatten_returns(sims[model])
            take = rng.choice(len(flat), size=min(len(hist), len(flat)), replace=False)
            panels.append((model, flat[take]))
    for ax, (label, arr) in zip(axes, panels):
        ax.scatter(arr[:, 0], arr[:, 1], s=4, alpha=0.25)
        ax.set_title(label)
        ax.set_xlabel("USD log return")
    axes[0].set_ylabel("EUR log return")
    fig.suptitle("Figure H. Joint USD/EUR daily log returns")
    _save(fig, "figH_joint_scatter")

    # I example 1y paths (deterministic first N_PATHS_EXAMPLE of seed)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    axes = axes.ravel()
    plotted = 0
    for model in ("iid_bootstrap", "stationary_bootstrap_L10", "student_t", "gaussian"):
        if model not in sims:
            continue
        r = sims[model][: config.N_PATHS_EXAMPLE, : config.HORIZON_1Y_DAYS, :]
        lvl = returns_to_levels(r, bundle.s0)
        ax = axes[plotted]
        for i in range(lvl.shape[0]):
            ax.plot(lvl[i, :, 0], linewidth=0.8)
        ax.set_title(model)
        ax.set_ylabel("USD/KZT")
        plotted += 1
        if plotted == 4:
            break
    fig.suptitle("Figure I. Example 1-year USD/KZT paths (fixed seed, first 8 paths)")
    _save(fig, "figI_example_1y_paths")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    scores: pd.DataFrame,
    tparams: pd.DataFrame,
    temporal: pd.DataFrame,
    vol: pd.DataFrame,
    shock: pd.DataFrame,
    oos: pd.DataFrame,
    pw_l: float,
    recommended_l: int,
) -> None:
    def best(col):
        s = scores.dropna(subset=[col])
        if s.empty:
            return "n/a"
        row = s.loc[s[col].idxmin()]
        return f"{row['model']} (score={row[col]:.4g})"

    t_acf = temporal.groupby("model")["acf_rmse"].mean().sort_values()
    v_err = vol.groupby("model")["abs_err_mean"].mean().sort_values()

    df = float(tparams.loc[0, "df"])
    text = f"""# Phase 2A Model Comparison Report

Simulation seed: `{config.SIM_RANDOM_SEED}`. Primary `n_paths={config.N_PATHS_PRIMARY}`,
horizon for path-based diagnostics = {config.HORIZON_1Y_DAYS} weekdays.
Category scores are **unweighted RMSEs of the listed discrepancy metrics; lower is better**.
An overall mean is shown but **must not hide a category failure**.

Student-t MLE: location μ, **scale matrix Σ**, degrees of freedom ν.
Covariance = ν/(ν-2) Σ. Fitted ν = **{df:.3f}**. MLE success={tparams.loc[0, 'mle_success']}.

Politis–White heuristic L on |r_USD|: **{pw_l:.2f}**.

## 1. Marginal distributions

Best category score: {best("marginal")}.

IID and stationary bootstrap resample historical rows, so one-day marginals
(and Pearson/Spearman in large samples) match the empirical law by construction.
Student-t and Gaussian are parametric approximations.

## 2. Tails

Best category score: {best("tail")}.

Do not judge solely on kurtosis: historical excess kurtosis is dominated by
the August 2015 float. Compare quantile errors and P(r ≤ historical min).
Bootstrap methods can reproduce the exact historical extremes by resampling them.
Student-t is symmetric around μ, so it cannot match the large **positive**
skewness of USD/KZT. Gaussian thins both tails.

## 3. Joint dependence

Best category score: {best("dependence")}.

EUR/KZT is an NBK cross rate. Bootstrap preserves the empirical copula of
sampled days. Student-t has a single correlation and symmetric tail dependence.
Rolling historical correlation was unstable in Phase 1; no i.i.d. model
reproduces time-varying dependence.

## 4. Temporal dependence

Best category score: {best("temporal")}.

Mean ACF RMSE by model (raw, |r|, r², both currencies):

{t_acf.to_string()}

IID / Gaussian / Student-t should collapse serial correlation. Stationary and
moving-block bootstrap should retain more short-run dependence. Verify the
table, not this paragraph.

## 5. Volatility clustering

Best category score: {best("vol_clustering")}.

Mean absolute error of rolling-vol means:

{v_err.to_string()}

## 6. Sensitivity to L (5 / 10 / 20)

Compare `stationary_bootstrap_L5/L10/L20` rows in temporal, vol, and multi-horizon
tables. Short L approaches IID; very long L repeats long historical stretches.
Recommended L for the primary DGP: **{recommended_l}**
(see also Politis–White heuristic {pw_l:.1f}).

## 7. August 2015 float-shock sensitivity

Exclusion window (sensitivity only; **not** the baseline):
{config.FLOAT_SHOCK_SENSITIVITY_START} through {config.FLOAT_SHOCK_SENSITIVITY_END}.

{shock.to_string(index=False)}

Student-t ν and historical kurtosis/quantiles move when the shock is dropped.
Bootstrap still includes other large moves. **Do not rewrite the baseline.**

## 8. Pseudo-out-of-sample (distributional)

Train {config.OOS_TRAIN_START}–{config.OOS_TRAIN_END}; evaluate against
{config.OOS_EVAL_START}–{config.OOS_EVAL_END} **moments**, not path forecasts.

This checks whether an earlier-regime calibration remains plausible later.
Phase 1 already showed Window C is calmer than the full post-float sample,
so models that lock in 2015–2019 tails may look too heavy vs 2020–2025.

## 9. Student-t vs Gaussian

Gaussian is a straw-man. Student-t should dominate Gaussian on tail RMSE if
ν is moderate. Neither has volatility clustering. If Student-t still misses
skewness, that is a structural limit of a symmetric elliptical copula.

## 10. Is GARCH/DCC needed later?

Add GARCH-type models in Phase 2B **robustness** if stationary bootstrap:
- still under-replicates ACF of |r| and r² at the chosen L, **or**
- rolling-vol dispersion is far from historical, **or**
- multi-year drawdowns look too much like shuffled history.

If SB with L around 10 already matches clustering and multi-horizon tails
adequately for household scenario use, GARCH is **optional**, not required,
especially given Phase 1 IGARCH-boundary diagnostics.

## 11. Primary scenario generator (for later household work)

**{scores.iloc[0]["model"]}** is the lowest overall equal-weight score, but the
primary recommendation is the model that best balances **tails + clustering +
joint empirical copula**, not the overall number alone.

Recommended primary DGP: **stationary bootstrap with L={recommended_l}**, even if a moving-block variant has a slightly lower overall score.
One-day margins of all bootstrap models are the same empirical law; ranking on overall RMSE can flip with Monte Carlo noise.
Choose SB over MBB for the geometric/circular construction (strict stationarity).
IID remains the no-clustering benchmark. Student-t and Gaussian are not primary.

## 12. Robustness DGPs

Keep:
- joint IID historical bootstrap (naive / no clustering)
- bivariate Student-t (parametric, no clustering)
- Gaussian straw-man (to show tail loss)
- alternative SB block lengths L=5 and L=20
- optional moving-block L=10

Do not use Gaussian as a household DGP.

## Weighting disclosure

Overall score = unweighted mean of available category scores
(marginal rel-SD RMSE, tail quantile abs RMSE, dependence abs errors,
ACF RMSE, rolling-vol mean abs error, multi-horizon SD abs error,
pseudo-OOS SD abs error). No extra tuning weights were applied.
"""
    (config.SIM_DIAGNOSTICS / "model_comparison_report.md").write_text(text, encoding="utf-8")


def recommend_block_length(temporal: pd.DataFrame, vol: pd.DataFrame) -> int:
    """Choose L among {{5,10,20}} by clustering/temporal fit, not by matching the mean."""
    cands = [f"stationary_bootstrap_L{l}" for l in config.STATIONARY_BLOCK_LENGTHS]
    scores = {}
    for m in cands:
        tm = temporal[temporal["model"] == m]
        vm = vol[vol["model"] == m]
        scores[m] = float(tm["acf_rmse"].mean()) + float(vm["abs_err_mean"].mean())
    best = min(scores, key=scores.get)
    return int(best.rsplit("L", 1)[1])


def run_phase2a() -> dict:
    _ensure_dirs()
    seed = config.SIM_RANDOM_SEED
    bundle = load_baseline_bundle()
    hist = bundle.returns
    models = build_models()
    fit_all(models, hist)
    tparams = student_t_fit_table(models)
    pw_l = politis_white_mean_block_length(np.abs(hist[:, 0]))
    Path(config.SIM_DIAGNOSTICS / "politis_white_block_length.json").write_text(
        json.dumps({"L_abs_usd": pw_l, "n": int(len(hist))}, indent=2)
    )

    conv = convergence_table(hist, seed)
    sims = simulate_all(
        models,
        n_paths=config.N_PATHS_PRIMARY,
        horizon=config.HORIZON_1Y_DAYS,
        seed=seed,
    )
    # extra paths for temporal ACF stability at shorter n but same horizon
    sims_temporal = {k: v[: config.N_PATHS_TEMPORAL] for k, v in sims.items()}
    marg = marginal_validation(hist, sims)
    tail = tail_validation(hist, sims)
    dep = dependence_validation(hist, sims)
    temporal = temporal_validation(hist, sims_temporal)
    vol = vol_clustering_validation(hist, sims_temporal)
    multi = multihorizon_validation(hist, sims)
    sanity = path_sanity_table(bundle, sims)
    oos = pseudo_oos_table(seed)
    shock = shock_sensitivity_table(seed)
    scores = score_models(marg, tail, dep, temporal, vol, multi, oos)
    rec_l = recommend_block_length(temporal, vol)
    make_figures(hist, sims, bundle)
    write_report(scores, tparams, temporal, vol, shock, oos, pw_l, rec_l)

    rec = {
        "n_hist": int(len(hist)),
        "n_paths": config.N_PATHS_PRIMARY,
        "seed": seed,
        "student_t_df": float(tparams.loc[0, "df"]),
        "politis_white_L": pw_l,
        "recommended_L": rec_l,
        "lowest_overall_score": scores.iloc[0]["model"],
        "path_sanity_all_ok": bool(sanity["return_identity_ok"].all() and sanity["all_positive"].all()),
        "convergence_rows": int(len(conv)),
    }
    Path(config.SIM_DIAGNOSTICS / "phase2a_run_summary.json").write_text(
        json.dumps(rec, indent=2)
    )
    return rec
