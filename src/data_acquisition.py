"""Acquire official daily USD/KZT and EUR/KZT rates from the National Bank of Kazakhstan.

Preferred source: NBK Excel export of Daily Official (Market) Foreign Exchange Rates.
Raw files are never overwritten. Place manually downloaded yearly workbooks in
``data/raw/`` using the pattern ``nbk_official_rates_YYYY.xlsx`` (or .xls/.csv)
and they will be merged automatically.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from . import config
from .utils import ensure_directories, sha256_file, utc_today, write_csv


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": config.USER_AGENT})


def raw_year_path(year: int, suffix: str = "xlsx") -> Path:
    return config.DATA_RAW / f"nbk_official_rates_{year}.{suffix}"


def download_year_excel(year: int, overwrite: bool = False) -> dict:
    """Download one calendar year from the official NBK Excel export."""
    ensure_directories()
    dest = raw_year_path(year)
    url = config.NBK_EXCEL_EXPORT
    begin = f"01.01.{year}"
    end = f"31.12.{year}"
    params = [
        ("beginDate", begin),
        ("endDate", end),
        ("rates[]", str(config.NBK_USD_RATE_ID)),
        ("rates[]", str(config.NBK_EUR_RATE_ID)),
    ]
    record = {
        "source": "National Bank of Kazakhstan",
        "file_or_endpoint": url,
        "download_date": utc_today(),
        "year": year,
        "number_of_records": None,
        "currencies_found": None,
        "status": "pending",
        "notes": "",
        "local_path": str(dest),
    }
    if dest.exists() and not overwrite:
        record["status"] = "skipped_existing"
        record["notes"] = "Raw file already present; not overwritten."
        return record
    try:
        response = SESSION.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "spreadsheet" not in content_type and not response.content.startswith(b"PK"):
            record["status"] = "failed"
            record["notes"] = (
                f"Unexpected content type {content_type!r}; "
                "place a manual yearly file in data/raw/."
            )
            return record
        dest.write_bytes(response.content)
        record["status"] = "downloaded"
        record["notes"] = f"beginDate={begin}; endDate={end}; rates USD={config.NBK_USD_RATE_ID}, EUR={config.NBK_EUR_RATE_ID}"
        return record
    except requests.RequestException as exc:
        record["status"] = "failed"
        record["notes"] = f"Download error: {exc}"
        return record


def acquire_nbk_yearly_excel(
    start_year: int | None = None,
    end_year: int | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    start_year = start_year or pd.Timestamp(config.RAW_START).year
    end_year = end_year or pd.Timestamp(config.RAW_END).year
    rows = []
    for year in range(start_year, end_year + 1):
        row = download_year_excel(year, overwrite=overwrite)
        rows.append(row)
        time.sleep(config.REQUEST_PAUSE_SECONDS)
    log = pd.DataFrame(rows)
    write_csv(log, config.DIAGNOSTICS_DIR / "data_acquisition_log.csv")
    return log


def list_raw_files() -> list[Path]:
    ensure_directories()
    files = []
    for pattern in ("*.xlsx", "*.xls", "*.csv", "*.xml"):
        files.extend(sorted(config.DATA_RAW.glob(pattern)))
    return [p for p in files if p.is_file() and not p.name.startswith(".")]


def build_raw_file_manifest() -> pd.DataFrame:
    rows = []
    for path in list_raw_files():
        rows.append(
            {
                "filename": path.name,
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "date_acquired": datetime.utcfromtimestamp(path.stat().st_mtime).date().isoformat(),
                "source_url": config.NBK_EXCEL_EXPORT if path.suffix.lower() in {".xlsx", ".xls"} else "",
            }
        )
    manifest = pd.DataFrame(rows)
    write_csv(manifest, config.DIAGNOSTICS_DIR / "raw_file_manifest.csv")
    return manifest


def fetch_rss_date(day: pd.Timestamp) -> dict:
    """Official per-date RSS service (recent years only on current NBK site)."""
    fdate = day.strftime("%d.%m.%Y")
    url = f"{config.NBK_RSS_RATES}?fdate={fdate}"
    response = SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    info = root.findtext("info")
    if info:
        return {"date": day, "usd_kzt": None, "eur_kzt": None, "note": info, "url": url}
    rates = {}
    quants = {}
    for item in root.findall("item"):
        code = (item.findtext("title") or "").strip().upper()
        rate = item.findtext("description")
        quant = item.findtext("quant") or "1"
        if code in config.TARGET_CURRENCIES:
            rates[code] = float(str(rate).replace(",", "."))
            quants[code] = float(str(quant).replace(",", "."))
    usd = rates.get("USD") / quants.get("USD", 1) if "USD" in rates else None
    eur = rates.get("EUR") / quants.get("EUR", 1) if "EUR" in rates else None
    return {"date": day, "usd_kzt": usd, "eur_kzt": eur, "note": "", "url": url}
