"""Pre/post-float comparison and change-point diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm

from . import config
from .utils import write_csv


def _moments(x: pd.Series) -> dict:
    x = x.dropna().astype(float)
    return {
        "N": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "variance": float(x.var(ddof=1)),
        "skewness": float(x.skew()),
        "excess_kurtosis": float(x.kurtosis()),
    }


def pre_post_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    pre = work[work["post_float"] == 0]
    post = work[work["post_float"] == 1]
    rows = []
    for series, col in (
        ("usd_kzt", "usd_kzt_log_return"),
        ("eur_kzt", "eur_kzt_log_return"),
    ):
        pre_m = _moments(pre[col])
        post_m = _moments(post[col])
        lev_stat, lev_p = stats.levene(pre[col].dropna(), post[col].dropna(), center="median")
        t_stat, t_p = stats.ttest_ind(
            pre[col].dropna(), post[col].dropna(), equal_var=False
        )
        rows.append(
            {
                "series": series,
                "comparison": "pre_float_vs_post_float",
                "break_date_assumed": config.FLOAT_TRANSITION_DATE,
                **{f"pre_{k}": v for k, v in pre_m.items()},
                **{f"post_{k}": v for k, v in post_m.items()},
                "levene_stat": float(lev_stat),
                "levene_pvalue": float(lev_p),
                "welch_t_stat": float(t_stat),
                "welch_t_pvalue": float(t_p),
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "structural_break_prepost.csv")
    return table


def chow_known_break(daily: pd.DataFrame) -> pd.DataFrame:
    """Chow test on a mean-only model of log returns at the documented float date."""
    rows = []
    work = daily.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    work = work.sort_values("date")
    break_ts = pd.Timestamp(config.FLOAT_TRANSITION_DATE)
    for series, col in (
        ("usd_kzt", "usd_kzt_log_return"),
        ("eur_kzt", "eur_kzt_log_return"),
    ):
        y = work[col].astype(float)
        x = np.ones(len(y))
        model = sm.OLS(y, x).fit()
        # split by known date
        mask = work["date"] < break_ts
        n1, n2 = int(mask.sum()), int((~mask).sum())
        if n1 < 10 or n2 < 10:
            rows.append({"series": series, "status": "insufficient_split"})
            continue
        rss_p = float(np.sum(model.resid ** 2))
        rss_1 = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        rss_2 = float(np.sum((y[~mask] - y[~mask].mean()) ** 2))
        k = 1
        n = n1 + n2
        chow = ((rss_p - (rss_1 + rss_2)) / k) / ((rss_1 + rss_2) / (n - 2 * k))
        pvalue = float(1 - stats.f.cdf(chow, k, n - 2 * k))
        rows.append(
            {
                "series": series,
                "test": "chow_mean_only",
                "break_date": config.FLOAT_TRANSITION_DATE,
                "chow_stat": float(chow),
                "pvalue": pvalue,
                "n_pre": n1,
                "n_post": n2,
            }
        )
        try:
            cusum = breaks_cusumolsresid(model.resid)
            rows[-1]["cusum_stat"] = float(cusum[0])
            rows[-1]["cusum_pvalue"] = float(cusum[1])
        except Exception as exc:  # noqa: BLE001
            rows[-1]["cusum_note"] = str(exc)
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "structural_break_chow.csv")
    return table


def ruptures_change_points(daily: pd.DataFrame, n_bkps: int = 3) -> pd.DataFrame:
    try:
        import ruptures as rpt
    except ImportError:
        table = pd.DataFrame([{"status": "ruptures not installed"}])
        write_csv(table, config.TABLES / "structural_break_ruptures.csv")
        return table

    work = daily.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    work = work.sort_values("date").reset_index(drop=True)
    rows = []
    for series, col in (
        ("usd_kzt_returns", "usd_kzt_log_return"),
        ("eur_kzt_returns", "eur_kzt_log_return"),
        ("usd_kzt_squared", None),
        ("eur_kzt_squared", None),
    ):
        if col is None:
            base = "usd_kzt_log_return" if series.startswith("usd") else "eur_kzt_log_return"
            signal = work[base].astype(float).to_numpy() ** 2
        else:
            signal = work[col].astype(float).to_numpy()
        algo = rpt.Binseg(model="l2").fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)
        for i, idx in enumerate(bkps[:-1]):
            date = work.loc[min(idx, len(work) - 1), "date"]
            rows.append(
                {
                    "series": series,
                    "method": "binseg_l2",
                    "breakpoint_order": i + 1,
                    "index": int(idx),
                    "date": pd.Timestamp(date).date().isoformat(),
                }
            )
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "structural_break_ruptures.csv")
    return table


def combine_structural_tables(daily: pd.DataFrame) -> pd.DataFrame:
    prepost = pre_post_comparison(daily)
    chow = chow_known_break(daily)
    brk = ruptures_change_points(daily)
    combined = []
    for _, row in prepost.iterrows():
        combined.append(
            {
                "test_family": "pre_post_moments_levene_welch",
                "series": row["series"],
                "statistic": row["levene_stat"],
                "pvalue": row["levene_pvalue"],
                "note": (
                    f"Levene variance test; Welch mean p={row['welch_t_pvalue']:.4g}; "
                    f"pre std={row['pre_std']:.5f}; post std={row['post_std']:.5f}"
                ),
            }
        )
    for _, row in chow.iterrows():
        combined.append(
            {
                "test_family": row.get("test", "chow"),
                "series": row.get("series"),
                "statistic": row.get("chow_stat"),
                "pvalue": row.get("pvalue"),
                "note": f"known break {row.get('break_date')}",
            }
        )
    if "date" in brk.columns:
        for _, row in brk.iterrows():
            combined.append(
                {
                    "test_family": "ruptures_binseg",
                    "series": row["series"],
                    "statistic": np.nan,
                    "pvalue": np.nan,
                    "note": f"breakpoint {row['breakpoint_order']} at {row['date']}",
                }
            )
    table = pd.DataFrame(combined)
    write_csv(table, config.TABLES / "structural_break_tests.csv")
    return table
