"""Historical vs simulated validation statistics for joint FX returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from src import config


CURRENCIES = ("usd", "eur")


def moments(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    s = pd.Series(x)
    return {
        "n": int(len(x)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(x) > 1 else np.nan,
        "skewness": float(s.skew()) if len(x) > 2 else np.nan,
        "excess_kurtosis": float(s.kurtosis()) if len(x) > 3 else np.nan,
        "min": float(s.min()) if len(x) else np.nan,
        "max": float(s.max()) if len(x) else np.nan,
        "p1": float(s.quantile(0.01)) if len(x) else np.nan,
        "p5": float(s.quantile(0.05)) if len(x) else np.nan,
        "p95": float(s.quantile(0.95)) if len(x) else np.nan,
        "p99": float(s.quantile(0.99)) if len(x) else np.nan,
    }


def flatten_returns(paths: np.ndarray) -> np.ndarray:
    return paths.reshape(-1, 2)


def overlapping_compound(series: np.ndarray, horizon: int) -> np.ndarray:
    """Sums of `horizon` consecutive log returns (overlapping)."""
    x = np.asarray(series, dtype=float)
    if len(x) < horizon:
        return np.array([])
    c = np.concatenate([[0.0], np.cumsum(x)])
    return c[horizon:] - c[:-horizon]


def acf_values(series: np.ndarray, nlags: int = 30) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < nlags + 5:
        return np.full(nlags + 1, np.nan)
    return acf(x, nlags=nlags, fft=True)


def mean_acf_from_paths(paths: np.ndarray, col: int, transform: str, nlags: int = 30) -> np.ndarray:
    """Average ACF across paths for one currency column."""
    acc = []
    for i in range(paths.shape[0]):
        x = paths[i, :, col]
        if transform == "abs":
            x = np.abs(x)
        elif transform == "sq":
            x = x ** 2
        acc.append(acf_values(x, nlags=nlags))
    return np.nanmean(np.vstack(acc), axis=0)


def rolling_vol_panel(paths: np.ndarray, window: int, col: int, ann: float | None = None) -> np.ndarray:
    ann = np.sqrt(config.ANNUALIZATION_FACTOR) if ann is None else ann
    n_paths, h, _ = paths.shape
    if h < window:
        return np.array([])
    out = []
    for i in range(n_paths):
        r = paths[i, :, col]
        rv = pd.Series(r).rolling(window).std(ddof=1).dropna().to_numpy() * ann
        out.append(rv[np.isfinite(rv)])
    return np.concatenate(out) if out else np.array([])


def historical_rolling_vol(series: np.ndarray, window: int) -> np.ndarray:
    ann = np.sqrt(config.ANNUALIZATION_FACTOR)
    return pd.Series(series).rolling(window).std(ddof=1).dropna().to_numpy() * ann


def dependence_stats(usd: np.ndarray, eur: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> dict:
    usd = np.asarray(usd, dtype=float)
    eur = np.asarray(eur, dtype=float)
    pearson = float(pd.Series(usd).corr(pd.Series(eur), method="pearson"))
    spearman = float(pd.Series(usd).corr(pd.Series(eur), method="spearman"))
    ul = np.quantile(usd, q_low)
    el = np.quantile(eur, q_low)
    uh = np.quantile(usd, q_high)
    eh = np.quantile(eur, q_high)
    lower = float(np.mean((usd <= ul) & (eur <= el)))
    upper = float(np.mean((usd >= uh) & (eur >= eh)))
    return {
        "pearson": pearson,
        "spearman": spearman,
        "lower_coexceed": lower,
        "upper_coexceed": upper,
        "usd_q_low": float(ul),
        "eur_q_low": float(el),
        "usd_q_high": float(uh),
        "eur_q_high": float(eh),
    }


def historical_threshold_coexceed(
    usd: np.ndarray,
    eur: np.ndarray,
    usd_q_low: float,
    eur_q_low: float,
    usd_q_high: float,
    eur_q_high: float,
) -> dict:
    """Co-exceedance using *historical* quantile thresholds (apples-to-apples)."""
    return {
        "lower_coexceed_hist_thr": float(np.mean((usd <= usd_q_low) & (eur <= eur_q_low))),
        "upper_coexceed_hist_thr": float(np.mean((usd >= usd_q_high) & (eur >= eur_q_high))),
    }


def tail_quantiles(x: np.ndarray, qs=None) -> dict:
    qs = qs or config.TAIL_QUANTILES
    s = pd.Series(np.asarray(x, dtype=float))
    return {f"q{int(q*1000)/10:.1f}".replace(".0", ""): float(s.quantile(q)) for q in qs}


def exceedance_probs(x: np.ndarray, thresholds: dict) -> dict:
    x = np.asarray(x, dtype=float)
    return {name: float(np.mean(x <= thr) if "low" in name or thr < 0 else np.mean(x >= thr))
            for name, thr in thresholds.items()}


def vol_distribution_stats(vol: np.ndarray) -> dict:
    s = pd.Series(np.asarray(vol, dtype=float))
    if s.empty:
        return {k: np.nan for k in ("mean", "median", "std", "p5", "p95", "max")}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "p5": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


@dataclass
class HistoricalBundle:
    returns: np.ndarray
    dates: np.ndarray | None
    s0: np.ndarray

    @property
    def usd(self) -> np.ndarray:
        return self.returns[:, 0]

    @property
    def eur(self) -> np.ndarray:
        return self.returns[:, 1]


def load_baseline_bundle(
    start: str | None = None,
    end: str | None = None,
    exclude_shock: bool = False,
) -> HistoricalBundle:
    path = config.DATA_PROCESSED / "fx_daily_2010_2025.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    start = start or config.BASELINE_START
    end = end or config.BASELINE_END
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    work = df.loc[mask].dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    if exclude_shock:
        shock0 = pd.Timestamp(config.FLOAT_SHOCK_SENSITIVITY_START)
        shock1 = pd.Timestamp(config.FLOAT_SHOCK_SENSITIVITY_END)
        work = work[(work["date"] < shock0) | (work["date"] > shock1)]
    r = work[["usd_kzt_log_return", "eur_kzt_log_return"]].to_numpy(dtype=float)
    s0 = work[["usd_kzt", "eur_kzt"]].iloc[-1].to_numpy(dtype=float)
    return HistoricalBundle(returns=r, dates=work["date"].to_numpy(), s0=s0)
