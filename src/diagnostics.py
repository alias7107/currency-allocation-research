"""Statistical diagnostics for USD/KZT and EUR/KZT log returns."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf

from . import config
from .utils import slice_window, write_csv

SERIES = {
    "usd_kzt": "usd_kzt_log_return",
    "eur_kzt": "eur_kzt_log_return",
}

WINDOWS = {
    "A_full": (config.WINDOW_A_START, config.WINDOW_A_END),
    "B_baseline": (config.BASELINE_START, config.BASELINE_END),
    "C_recent": (config.RECENT_START, config.RECENT_END),
}


def _ann_vol(std: float, factor: int | None = None) -> float:
    factor = factor or config.ANNUALIZATION_FACTOR
    return float(std * np.sqrt(factor))


def descriptive_row(values: pd.Series, series_name: str, window_name: str) -> dict:
    x = values.dropna().astype(float)
    return {
        "window": window_name,
        "series": series_name,
        "N": int(len(x)),
        "mean": float(x.mean()) if len(x) else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "annualized_volatility": _ann_vol(float(x.std(ddof=1))) if len(x) > 1 else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
        "max": float(x.max()) if len(x) else np.nan,
        "p1": float(x.quantile(0.01)) if len(x) else np.nan,
        "p5": float(x.quantile(0.05)) if len(x) else np.nan,
        "p95": float(x.quantile(0.95)) if len(x) else np.nan,
        "p99": float(x.quantile(0.99)) if len(x) else np.nan,
        "skewness": float(x.skew()) if len(x) > 2 else np.nan,
        "excess_kurtosis": float(x.kurtosis()) if len(x) > 3 else np.nan,
    }


def descriptive_statistics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_name, (start, end) in WINDOWS.items():
        sample = slice_window(daily, start, end)
        for label, col in SERIES.items():
            rows.append(descriptive_row(sample[col], label, window_name))
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "descriptive_statistics.csv")
    _write_latex(table, config.TABLES / "descriptive_statistics.tex")
    return table


def _write_latex(table: pd.DataFrame, path) -> None:
    cols = list(table.columns)
    header = " & ".join(cols) + r" \\"
    lines = [
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, float):
                cells.append(f"{val:.6f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def distribution_tests(daily: pd.DataFrame) -> pd.DataFrame:
    sample = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    rows = []
    for label, col in SERIES.items():
        x = sample[col].dropna().astype(float)
        jb_stat, jb_p = stats.jarque_bera(x)
        rows.append(
            {
                "series": label,
                "window": "B_baseline",
                "N": int(len(x)),
                "skewness": float(x.skew()),
                "excess_kurtosis": float(x.kurtosis()),
                "jarque_bera_stat": float(jb_stat),
                "jarque_bera_pvalue": float(jb_p),
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "distribution_tests.csv")
    return table


def dependence_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window_name, (start, end) in WINDOWS.items():
        sample = slice_window(daily, start, end).dropna(
            subset=["usd_kzt_log_return", "eur_kzt_log_return"]
        )
        pearson = sample["usd_kzt_log_return"].corr(sample["eur_kzt_log_return"], method="pearson")
        spearman = sample["usd_kzt_log_return"].corr(sample["eur_kzt_log_return"], method="spearman")
        rows.append(
            {
                "window": window_name,
                "N": int(len(sample)),
                "pearson": float(pearson),
                "spearman": float(spearman),
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "return_correlations.csv")
    return table


def rolling_correlations(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    for w in config.ROLLING_CORR_WINDOWS:
        work[f"corr_{w}d"] = (
            work["usd_kzt_log_return"].rolling(w).corr(work["eur_kzt_log_return"])
        )
    write_csv(work[["date"] + [f"corr_{w}d" for w in config.ROLLING_CORR_WINDOWS]],
              config.DATA_INTERIM / "rolling_correlations.csv")
    return work


def _ljung_box_rows(series: pd.Series, transform: str, currency: str) -> list[dict]:
    x = series.dropna().astype(float)
    if transform == "abs":
        x = x.abs()
    elif transform == "squared":
        x = x ** 2
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = acorr_ljungbox(x, lags=list(config.LJUNG_BOX_LAGS), return_df=True)
    for lag in config.LJUNG_BOX_LAGS:
        rows.append(
            {
                "series": currency,
                "transform": transform,
                "lag": int(lag),
                "ljung_box_stat": float(lb.loc[lag, "lb_stat"]),
                "ljung_box_pvalue": float(lb.loc[lag, "lb_pvalue"]),
            }
        )
    return rows


def autocorrelation_tests(daily: pd.DataFrame) -> pd.DataFrame:
    sample = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    rows = []
    acf_rows = []
    for label, col in SERIES.items():
        x = sample[col].dropna().astype(float)
        for transform in ("raw", "abs", "squared"):
            rows.extend(_ljung_box_rows(x, transform, label))
            y = x if transform == "raw" else (x.abs() if transform == "abs" else x ** 2)
            values = acf(y, nlags=config.ACF_MAX_LAG, fft=True)
            for lag, val in enumerate(values):
                if lag == 0:
                    continue
                acf_rows.append(
                    {
                        "series": label,
                        "transform": transform,
                        "lag": lag,
                        "acf": float(val),
                    }
                )
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "autocorrelation_tests.csv")
    write_csv(pd.DataFrame(acf_rows), config.TABLES / "acf_values.csv")
    return table


def arch_tests(daily: pd.DataFrame) -> pd.DataFrame:
    sample = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    rows = []
    for label, col in SERIES.items():
        x = sample[col].dropna().astype(float)
        for lag in config.ARCH_LM_LAGS:
            stat, pvalue, _, _ = het_arch(x, nlags=lag)
            rows.append(
                {
                    "series": label,
                    "lags": int(lag),
                    "arch_lm_stat": float(stat),
                    "arch_lm_pvalue": float(pvalue),
                }
            )
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "arch_tests.csv")
    return table


def garch_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    """Fit diagnostic GARCH(1,1); not a final simulation model."""
    try:
        from arch import arch_model
    except ImportError:
        table = pd.DataFrame(
            [{"series": "both", "status": "arch package not installed", "note": "skipped"}]
        )
        write_csv(table, config.TABLES / "garch11_diagnostics.csv")
        return table

    sample = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    rows = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    _ = rng  # seed reserved for any stochastic fitting options
    for label, col in SERIES.items():
        x = sample[col].dropna().astype(float) * 100  # percent scale for numerical stability
        try:
            model = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
            fitted = model.fit(disp="off")
            params = fitted.params
            omega = float(params.get("omega", np.nan))
            alpha = float(params.get("alpha[1]", np.nan))
            beta = float(params.get("beta[1]", np.nan))
            resid = pd.Series(fitted.std_resid).dropna()
            jb_stat, jb_p = stats.jarque_bera(resid)
            lb_raw = acorr_ljungbox(resid, lags=[10], return_df=True)
            lb_sq = acorr_ljungbox(resid ** 2, lags=[10], return_df=True)
            rows.append(
                {
                    "series": label,
                    "status": "ok",
                    "omega": omega,
                    "alpha": alpha,
                    "beta": beta,
                    "alpha_plus_beta": alpha + beta,
                    "std_resid_jarque_bera_stat": float(jb_stat),
                    "std_resid_jarque_bera_pvalue": float(jb_p),
                    "std_resid_ljung_box_10_pvalue": float(lb_raw.loc[10, "lb_pvalue"]),
                    "std_resid_sq_ljung_box_10_pvalue": float(lb_sq.loc[10, "lb_pvalue"]),
                    "note": "Diagnostic GARCH(1,1) only; not used for allocation.",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"series": label, "status": "failed", "note": str(exc)})
    table = pd.DataFrame(rows)
    write_csv(table, config.TABLES / "garch11_diagnostics.csv")
    return table


def extreme_returns(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    work["usd_prev"] = work["usd_kzt"].shift(1)
    work["eur_prev"] = work["eur_kzt"].shift(1)
    parts = []
    n = config.TOP_EXTREMES
    parts.append(_top(work, "usd_kzt_log_return", "usd_kzt", "usd_prev", n, True, "USD"))
    parts.append(_top(work, "usd_kzt_log_return", "usd_kzt", "usd_prev", n, False, "USD"))
    parts.append(_top(work, "eur_kzt_log_return", "eur_kzt", "eur_prev", n, True, "EUR"))
    parts.append(_top(work, "eur_kzt_log_return", "eur_kzt", "eur_prev", n, False, "EUR"))
    table = pd.concat(parts, ignore_index=True)
    write_csv(table, config.TABLES / "extreme_returns.csv")
    return table


def _top(
    work: pd.DataFrame,
    ret_col: str,
    price_col: str,
    prev_col: str,
    n: int,
    positive: bool,
    currency: str,
) -> pd.DataFrame:
    s = work.dropna(subset=[ret_col, prev_col])
    picked = s.nlargest(n, ret_col) if positive else s.nsmallest(n, ret_col)
    out = picked[["date", prev_col, price_col, ret_col]].copy()
    out.columns = ["date", "price_before", "price_after", "log_return"]
    out["currency"] = currency
    out["direction"] = "positive" if positive else "negative"
    return out


def methodology_window_sensitivity(daily: pd.DataFrame) -> pd.DataFrame:
    baseline = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    included = baseline.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"])
    excluded = included[included["methodology_window_2016"] == 0]

    def _block(df: pd.DataFrame, label: str) -> dict:
        return {
            "sample": label,
            "N": int(len(df)),
            "usd_mean": float(df["usd_kzt_log_return"].mean()),
            "usd_std": float(df["usd_kzt_log_return"].std(ddof=1)),
            "usd_ann_vol": _ann_vol(float(df["usd_kzt_log_return"].std(ddof=1))),
            "usd_skew": float(df["usd_kzt_log_return"].skew()),
            "usd_exkurt": float(df["usd_kzt_log_return"].kurtosis()),
            "eur_mean": float(df["eur_kzt_log_return"].mean()),
            "eur_std": float(df["eur_kzt_log_return"].std(ddof=1)),
            "eur_ann_vol": _ann_vol(float(df["eur_kzt_log_return"].std(ddof=1))),
            "eur_skew": float(df["eur_kzt_log_return"].skew()),
            "eur_exkurt": float(df["eur_kzt_log_return"].kurtosis()),
            "pearson": float(df["usd_kzt_log_return"].corr(df["eur_kzt_log_return"])),
            "spearman": float(
                df["usd_kzt_log_return"].corr(df["eur_kzt_log_return"], method="spearman")
            ),
        }

    table = pd.DataFrame(
        [
            _block(included, "baseline_including_2016_window"),
            _block(excluded, "baseline_excluding_2016_window"),
        ]
    )
    write_csv(table, config.TABLES / "methodology_window_sensitivity.csv")
    return table


def monthly_versus_daily(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    base_d = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    m = monthly.copy()
    m["month"] = pd.to_datetime(m["month"])
    base_m = m[
        (m["month"] >= pd.Timestamp(config.BASELINE_START))
        & (m["month"] <= pd.Timestamp(config.BASELINE_END))
    ]
    rows = [
        descriptive_row(base_d["usd_kzt_log_return"], "usd_daily_log", "B_baseline"),
        descriptive_row(base_d["eur_kzt_log_return"], "eur_daily_log", "B_baseline"),
        descriptive_row(base_m["usd_kzt_monthly_return"], "usd_monthly_simple", "B_baseline"),
        descriptive_row(base_m["eur_kzt_monthly_return"], "eur_monthly_simple", "B_baseline"),
    ]
    table = pd.DataFrame(rows)
    # monthly annualized vol uses 12 observations
    for i, row in table.iterrows():
        if "monthly" in row["series"] and pd.notna(row["std"]):
            table.loc[i, "annualized_volatility"] = float(row["std"] * np.sqrt(12))
    write_csv(table, config.TABLES / "monthly_vs_daily_descriptives.csv")
    return table
