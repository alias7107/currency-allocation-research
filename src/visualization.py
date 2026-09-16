"""Publication-style figures for Phase 1."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

from . import config
from .utils import slice_window

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _save(fig: plt.Figure, stem: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"{stem}.png")
    fig.savefig(config.FIGURES / f"{stem}.pdf")
    plt.close(fig)


def _mark_float(ax) -> None:
    ax.axvline(pd.Timestamp(config.FLOAT_TRANSITION_DATE), linestyle="--", linewidth=1)
    ax.set_xlabel("Date")


def figure_levels(daily: pd.DataFrame) -> None:
    work = daily.copy()
    work["date"] = pd.to_datetime(work["date"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(work["date"], work["usd_kzt"])
    ax.set_title("Figure 1. Official USD/KZT, 2010–2025")
    ax.set_ylabel("KZT per 1 USD")
    _mark_float(ax)
    _save(fig, "fig01_usd_kzt_level")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(work["date"], work["eur_kzt"])
    ax.set_title("Figure 2. Official EUR/KZT, 2010–2025")
    ax.set_ylabel("KZT per 1 EUR")
    _mark_float(ax)
    _save(fig, "fig02_eur_kzt_level")


def figure_normalized(daily: pd.DataFrame) -> None:
    base = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    if base.empty:
        return
    start_usd = base.iloc[0]["usd_kzt"]
    start_eur = base.iloc[0]["eur_kzt"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pd.to_datetime(base["date"]), 100 * base["usd_kzt"] / start_usd, label="USD/KZT")
    ax.plot(pd.to_datetime(base["date"]), 100 * base["eur_kzt"] / start_eur, label="EUR/KZT")
    ax.set_title("Figure 3. Levels normalized to 100 at baseline start")
    ax.set_ylabel("Index (baseline start = 100)")
    ax.legend()
    ax.set_xlabel("Date")
    _save(fig, "fig03_normalized_baseline")


def figure_returns(daily: pd.DataFrame) -> None:
    work = daily.dropna(subset=["usd_kzt_log_return"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pd.to_datetime(work["date"]), work["usd_kzt_log_return"])
    ax.set_title("Figure 4. USD/KZT daily log returns")
    ax.set_ylabel("Log return")
    _mark_float(ax)
    _save(fig, "fig04_usd_log_returns")

    work = daily.dropna(subset=["eur_kzt_log_return"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pd.to_datetime(work["date"]), work["eur_kzt_log_return"])
    ax.set_title("Figure 5. EUR/KZT daily log returns")
    ax.set_ylabel("Log return")
    _mark_float(ax)
    _save(fig, "fig05_eur_log_returns")


def figure_rolling_vol(daily: pd.DataFrame) -> None:
    work = daily.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    factor = np.sqrt(config.ANNUALIZATION_FACTOR)
    for window, fignum, stem in (
        (30, 6, "fig06_rolling_vol_30"),
        (90, 7, "fig07_rolling_vol_90"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(
            pd.to_datetime(work["date"]),
            work["usd_kzt_log_return"].rolling(window).std() * factor,
            label="USD/KZT",
        )
        ax.plot(
            pd.to_datetime(work["date"]),
            work["eur_kzt_log_return"].rolling(window).std() * factor,
            label="EUR/KZT",
        )
        ax.set_title(f"Figure {fignum}. Rolling {window}-day annualized volatility")
        ax.set_ylabel("Annualized volatility")
        ax.legend()
        _mark_float(ax)
        _save(fig, stem)


def figure_distributions(daily: pd.DataFrame) -> None:
    sample = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    for label, col, stem in (
        ("USD/KZT", "usd_kzt_log_return", "fig08_usd_distribution"),
        ("EUR/KZT", "eur_kzt_log_return", "fig09_eur_distribution"),
    ):
        x = sample[col].dropna().astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(x, bins=50, density=True, alpha=0.75)
        grid = np.linspace(x.min(), x.max(), 200)
        kde = stats.gaussian_kde(x)
        axes[0].plot(grid, kde(grid))
        axes[0].set_title(f"{label} histogram and KDE (baseline)")
        stats.probplot(x, dist="norm", plot=axes[1])
        axes[1].set_title(f"{label} normal Q-Q plot")
        _save(fig, stem)


def figure_rolling_correlation(corr_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dates = pd.to_datetime(corr_df["date"])
    for w in config.ROLLING_CORR_WINDOWS:
        ax.plot(dates, corr_df[f"corr_{w}d"], label=f"{w}-day")
    ax.set_title("Rolling correlation of USD/KZT and EUR/KZT log returns")
    ax.set_ylabel("Correlation")
    ax.legend()
    _mark_float(ax)
    _save(fig, "fig10_rolling_correlation")


def figure_acf(daily: pd.DataFrame) -> None:
    sample = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    for label, col, prefix in (
        ("USD/KZT", "usd_kzt_log_return", "fig11_usd"),
        ("EUR/KZT", "eur_kzt_log_return", "fig12_eur"),
    ):
        x = sample[col].dropna().astype(float)
        for data, suffix, title in (
            (x, "acf_raw", "raw returns"),
            (x.abs(), "acf_abs", "absolute returns"),
            (x ** 2, "acf_sq", "squared returns"),
        ):
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_acf(data, lags=config.ACF_MAX_LAG, ax=ax)
            ax.set_title(f"{label} ACF, {title} (baseline)")
            _save(fig, f"{prefix}_{suffix}")


def save_all_figures(daily: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    figure_levels(daily)
    figure_normalized(daily)
    figure_returns(daily)
    figure_rolling_vol(daily)
    figure_distributions(daily)
    figure_rolling_correlation(corr_df)
    figure_acf(daily)
