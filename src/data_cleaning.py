"""Parse heterogeneous NBK workbooks and validate the unified daily panel."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_acquisition import list_raw_files
from .utils import parse_mixed_date, parse_number, write_csv


USD_HEADER_TOKENS = ("usd", "доллар сша", "доллар", "ақш доллары", "us dollar")
EUR_HEADER_TOKENS = ("eur", "евро", "еуро", "euro")
DATE_HEADER_TOKENS = ("date", "дата", "күн")
QUANT_TOKENS = ("quant", "qty", "единиц", "units", "кол")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def identify_currency_column(columns: list[str], currency: str) -> str | None:
    """Map a header list to USD or EUR; prefers exact code, then language names.

    Quant columns are ignored so that rate columns are selected.
    """
    tokens = USD_HEADER_TOKENS if currency.upper() == "USD" else EUR_HEADER_TOKENS
    code = currency.lower()
    ranked: list[tuple[int, str]] = []
    for col in columns:
        n = _norm(col)
        if any(q in n for q in QUANT_TOKENS) and code not in n.replace("_quant", ""):
            continue
        if n == code or (n.startswith(f"{code}_") and "quant" not in n):
            ranked.append((0, col))
        elif code in n and "quant" not in n:
            ranked.append((1, col))
        elif any(tok in n for tok in tokens) and "quant" not in n:
            ranked.append((2, col))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][1]


def identify_quant_column(columns: list[str], currency: str) -> str | None:
    code = currency.lower()
    for col in columns:
        n = _norm(col)
        if code in n and any(q in n for q in QUANT_TOKENS):
            return col
    return None


def identify_date_column(columns: list[str]) -> str | None:
    for col in columns:
        n = _norm(col)
        if any(tok == n or tok in n for tok in DATE_HEADER_TOKENS):
            return col
    return columns[0] if columns else None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frames = []
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
            parsed = _promote_header(raw)
            if parsed is not None and not parsed.empty:
                frames.append(parsed)
        if not frames:
            raise ValueError(f"No usable sheet in {path.name}")
        return pd.concat(frames, ignore_index=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported raw file type: {path.name}")


def _promote_header(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Find a header row containing a date column and a USD/EUR token."""
    max_scan = min(15, len(raw))
    for i in range(max_scan):
        headers = [str(x) if pd.notna(x) else f"col_{j}" for j, x in enumerate(raw.iloc[i].tolist())]
        joined = " ".join(_norm(h) for h in headers)
        has_date = any(tok in joined for tok in DATE_HEADER_TOKENS)
        has_fx = any(tok in joined for tok in USD_HEADER_TOKENS + EUR_HEADER_TOKENS)
        if has_date and has_fx:
            body = raw.iloc[i + 1 :].copy()
            body.columns = headers
            body = body.dropna(how="all")
            return body
    # fallback: first row
    headers = [str(x) if pd.notna(x) else f"col_{j}" for j, x in enumerate(raw.iloc[0].tolist())]
    body = raw.iloc[1:].copy()
    body.columns = headers
    return body.dropna(how="all")


def parse_nbk_file(path: Path) -> pd.DataFrame:
    table = _read_table(path)
    columns = [str(c) for c in table.columns]
    date_col = identify_date_column(columns)
    usd_col = identify_currency_column(columns, "USD")
    eur_col = identify_currency_column(columns, "EUR")
    usd_q = identify_quant_column(columns, "USD")
    eur_q = identify_quant_column(columns, "EUR")
    if date_col is None or usd_col is None or eur_col is None:
        raise ValueError(
            f"Could not identify USD/EUR/date columns in {path.name}: {columns}"
        )
    out = pd.DataFrame(
        {
            "date": table[date_col].map(parse_mixed_date),
            "usd_raw": table[usd_col].map(parse_number),
            "eur_raw": table[eur_col].map(parse_number),
            "usd_quant": table[usd_q].map(parse_number) if usd_q else 1.0,
            "eur_quant": table[eur_q].map(parse_number) if eur_q else 1.0,
            "source_file": path.name,
        }
    )
    out["usd_quant"] = out["usd_quant"].replace(0, np.nan).fillna(1.0)
    out["eur_quant"] = out["eur_quant"].replace(0, np.nan).fillna(1.0)
    out["usd_kzt"] = out["usd_raw"] / out["usd_quant"]
    out["eur_kzt"] = out["eur_raw"] / out["eur_quant"]
    out = out.dropna(subset=["date"])
    return out[["date", "usd_kzt", "eur_kzt", "source_file"]]


def load_and_merge_raw() -> pd.DataFrame:
    files = list_raw_files()
    if not files:
        raise FileNotFoundError(
            "No files in data/raw/. Run acquisition or drop yearly NBK Excel files there."
        )
    frames = []
    errors = []
    for path in files:
        try:
            frames.append(parse_nbk_file(path))
        except Exception as exc:  # noqa: BLE001 - collect parse failures for the log
            errors.append({"filename": path.name, "error": str(exc)})
    if errors:
        write_csv(pd.DataFrame(errors), config.DIAGNOSTICS_DIR / "raw_parse_errors.csv")
    if not frames:
        raise RuntimeError("All raw files failed to parse.")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["date", "source_file"]).drop_duplicates(subset=["date"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    write_csv(merged, config.DATA_INTERIM / "fx_daily_merged_raw.csv")
    return merged


def validate_panel(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    flags = []

    dupes = work[work["date"].duplicated(keep=False)]
    if not dupes.empty:
        flags.append(("duplicate_dates", len(dupes)))
        write_csv(dupes, config.DIAGNOSTICS_DIR / "duplicate_dates.csv")

    malformed = work[work["date"].isna()]
    if not malformed.empty:
        flags.append(("malformed_dates", len(malformed)))

    unsorted = not work["date"].is_monotonic_increasing
    flags.append(("unsorted_before_sort", int(unsorted)))

    for col in ("usd_kzt", "eur_kzt"):
        flags.append((f"{col}_missing", int(work[col].isna().sum())))
        flags.append((f"{col}_zero", int((work[col] == 0).sum())))
        flags.append((f"{col}_negative", int((work[col] < 0).sum())))
        flags.append((f"{col}_duplicated_price_runs", int((work[col] == work[col].shift()).sum())))

    aligned = work.dropna(subset=["usd_kzt", "eur_kzt"]).copy()
    missing_alignment = work[work["usd_kzt"].isna() | work["eur_kzt"].isna()]
    write_csv(missing_alignment, config.DIAGNOSTICS_DIR / "missing_currency_alignment.csv")

    start, end = aligned["date"].min(), aligned["date"].max()
    full_days = pd.date_range(start, end, freq="D")
    present = pd.DatetimeIndex(aligned["date"].unique())
    missing_dates = full_days.difference(present)
    missing_report = pd.DataFrame({"missing_date": missing_dates})
    if not missing_report.empty:
        missing_report["weekday"] = missing_report["missing_date"].dt.day_name()
    write_csv(missing_report, config.DIAGNOSTICS_DIR / "missing_dates.csv")

    aligned["dow"] = aligned["date"].dt.dayofweek
    n_sat = int((aligned["dow"] == 5).sum())
    n_sun = int((aligned["dow"] == 6).sum())
    weekdays = aligned[aligned["dow"] < 5]
    weekday_span = pd.bdate_range(weekdays["date"].min(), weekdays["date"].max())
    weekday_gaps = weekday_span.difference(pd.DatetimeIndex(weekdays["date"]))
    write_csv(pd.DataFrame({"weekday_gap": weekday_gaps}), config.DIAGNOSTICS_DIR / "weekday_gaps.csv")

    tmp = aligned.sort_values("date")
    tmp["usd_r"] = np.log(tmp["usd_kzt"] / tmp["usd_kzt"].shift(1))
    tmp["eur_r"] = np.log(tmp["eur_kzt"] / tmp["eur_kzt"].shift(1))
    zero_usd = int((tmp["usd_r"] == 0).sum())
    zero_eur = int((tmp["eur_r"] == 0).sum())

    tmp["usd_extreme"] = tmp["usd_r"].abs() >= config.EXTREME_JUMP_ABS_LOG_RETURN
    tmp["eur_extreme"] = tmp["eur_r"].abs() >= config.EXTREME_JUMP_ABS_LOG_RETURN
    extremes = tmp[tmp["usd_extreme"] | tmp["eur_extreme"]][
        ["date", "usd_kzt", "eur_kzt", "usd_r", "eur_r", "usd_extreme", "eur_extreme"]
    ]
    write_csv(extremes, config.DIAGNOSTICS_DIR / "flagged_extreme_jumps.csv")

    summary = {
        "n_raw_rows": int(len(work)),
        "n_aligned": int(len(aligned)),
        "start": None if pd.isna(start) else pd.Timestamp(start).date().isoformat(),
        "end": None if pd.isna(end) else pd.Timestamp(end).date().isoformat(),
        "duplicate_date_rows": int(len(dupes)),
        "malformed_dates": int(len(malformed)),
        "missing_calendar_dates": int(len(missing_dates)),
        "n_saturdays": n_sat,
        "n_sundays": n_sun,
        "n_weekday_gaps": int(len(weekday_gaps)),
        "n_zero_usd_returns_full_calendar": zero_usd,
        "n_zero_eur_returns_full_calendar": zero_eur,
        "n_flagged_extreme_jumps": int(len(extremes)),
        "flags": flags,
    }
    write_csv(pd.DataFrame([summary]), config.DIAGNOSTICS_DIR / "validation_summary.csv")
    return summary
