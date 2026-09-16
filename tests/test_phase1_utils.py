"""Deterministic unit tests for Phase 1 utilities (no live NBK calls)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_cleaning import identify_currency_column, identify_date_column
from src.returns import compound_monthly, log_return
from src.utils import parse_mixed_date, parse_number


def test_log_return_two_prices():
    s = pd.Series([100.0, 110.0, 99.0])
    r = log_return(s)
    assert np.isnan(r.iloc[0])
    assert np.isclose(r.iloc[1], np.log(1.1))
    assert np.isclose(r.iloc[2], np.log(99.0 / 110.0))


def test_monthly_compounding_identity():
    days = pd.date_range("2020-01-01", periods=3, freq="B")
    r = pd.Series([0.01, -0.005, 0.002])
    daily = pd.DataFrame(
        {
            "date": days,
            "usd_kzt": np.nan,
            "eur_kzt": np.nan,
            "usd_kzt_log_return": r,
            "eur_kzt_log_return": r,
        }
    )
    monthly = compound_monthly(daily)
    expected = float(np.exp(r.sum()) - 1)
    assert np.isclose(monthly.loc[0, "usd_kzt_monthly_return"], expected)
    assert np.isclose(monthly.loc[0, "eur_kzt_monthly_return"], expected)


def test_currency_column_identification():
    cols = ["Date", "USD_quant", "USD", "EUR_quant", "EUR"]
    assert identify_currency_column(cols, "USD") == "USD"
    assert identify_currency_column(cols, "EUR") == "EUR"
    assert identify_date_column(cols) == "Date"
    ru = ["Дата", "ДОЛЛАР США", "ЕВРО"]
    assert identify_currency_column(ru, "USD") == "ДОЛЛАР США"
    assert identify_currency_column(ru, "EUR") == "ЕВРО"


def test_duplicate_date_keep_last_logic():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
            "usd_kzt": [1.0, 2.0, 3.0],
            "source_file": ["a.xlsx", "b.xlsx", "b.xlsx"],
        }
    )
    out = df.sort_values(["date", "source_file"]).drop_duplicates(subset=["date"], keep="last")
    assert len(out) == 2
    assert float(out.loc[out["date"] == "2020-01-02", "usd_kzt"].iloc[0]) == 2.0


def test_parse_decimal_comma_and_dates():
    assert parse_number("187,45") == 187.45
    assert parse_mixed_date("20.08.2015") == pd.Timestamp("2015-08-20")
    assert parse_mixed_date("2015-08-20") == pd.Timestamp("2015-08-20")
