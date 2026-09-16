"""Log returns, flags, and monthly compounding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .utils import write_csv


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    float_dt = pd.Timestamp(config.FLOAT_TRANSITION_DATE)
    meth_start = pd.Timestamp(config.METHODOLOGY_WINDOW_START)
    meth_end = pd.Timestamp(config.METHODOLOGY_WINDOW_END)
    out["post_float"] = (out["date"] >= float_dt).astype(int)
    out["methodology_window_2016"] = (
        (out["date"] >= meth_start) & (out["date"] <= meth_end)
    ).astype(int)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["weekday"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    return out


def log_return(price: pd.Series) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce")
    return np.log(price / price.shift(1))


def build_daily_processed(levels: pd.DataFrame, calendar: str | None = None) -> pd.DataFrame:
    """Sort, optionally restrict to weekdays, then compute log returns."""
    calendar = calendar or config.ESTIMATION_CALENDAR
    out = add_flags(levels)
    out = out.sort_values("date")
    if calendar == "weekdays":
        out = out.loc[out["is_weekend"] == 0].copy()
    out["usd_kzt_log_return"] = log_return(out["usd_kzt"])
    out["eur_kzt_log_return"] = log_return(out["eur_kzt"])
    cols = [
        "date",
        "usd_kzt",
        "eur_kzt",
        "usd_kzt_log_return",
        "eur_kzt_log_return",
        "post_float",
        "methodology_window_2016",
        "year",
        "month",
        "is_weekend",
    ]
    out = out[cols].reset_index(drop=True)
    write_csv(out, config.DATA_PROCESSED / "fx_daily_2010_2025.csv")
    return out


def compound_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.dropna(subset=["usd_kzt_log_return", "eur_kzt_log_return"]).copy()
    work["date"] = pd.to_datetime(work["date"])
    work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    grouped = work.groupby("month", as_index=False).agg(
        usd_sum=("usd_kzt_log_return", "sum"),
        eur_sum=("eur_kzt_log_return", "sum"),
        n_days=("date", "size"),
    )
    grouped["usd_kzt_monthly_return"] = np.exp(grouped["usd_sum"]) - 1
    grouped["eur_kzt_monthly_return"] = np.exp(grouped["eur_sum"]) - 1
    monthly = grouped[["month", "usd_kzt_monthly_return", "eur_kzt_monthly_return", "n_days"]]
    write_csv(monthly, config.DATA_PROCESSED / "fx_monthly_2010_2025.csv")
    return monthly
