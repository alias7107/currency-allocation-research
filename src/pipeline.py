"""End-to-end Phase 1 runner: acquire, clean, diagnose, report."""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .data_acquisition import acquire_nbk_yearly_excel, build_raw_file_manifest
from .data_cleaning import load_and_merge_raw, validate_panel
from .diagnostics import (
    arch_tests,
    autocorrelation_tests,
    dependence_table,
    descriptive_statistics,
    distribution_tests,
    extreme_returns,
    garch_diagnostics,
    methodology_window_sensitivity,
    monthly_versus_daily,
    rolling_correlations,
)
from .returns import build_daily_processed, compound_monthly
from .structural_breaks import combine_structural_tables
from .utils import ensure_directories, slice_window
from .visualization import save_all_figures


def _sig(p: float, alpha: float = 0.05) -> str:
    if pd.isna(p):
        return "n/a"
    return "reject H0 at 5%" if p < alpha else "do not reject H0 at 5%"


def write_data_quality_report(summary: dict, daily: pd.DataFrame) -> None:
    manifest_path = config.DIAGNOSTICS_DIR / "raw_file_manifest.csv"
    acq_path = config.DIAGNOSTICS_DIR / "data_acquisition_log.csv"
    text = f"""# Data Quality Report (Phase 1)

## Raw files acquired

See `{manifest_path.name}` for SHA256 hashes and `{acq_path.name}` for the download log.

Source: National Bank of Kazakhstan, Daily Official (Market) Foreign Exchange Rates.
Excel export: `{config.NBK_EXCEL_EXPORT}`
USD rate id (page checkbox): {config.NBK_USD_RATE_ID}; EUR rate id: {config.NBK_EUR_RATE_ID}.

## Coverage

- Requested raw window: {config.RAW_START} to {config.RAW_END}
- Observed aligned start/end: {summary.get('start')} to {summary.get('end')}
- Aligned observations (both USD and EUR present): {summary.get('n_aligned')}
- Duplicate date rows: {summary.get('duplicate_date_rows')}
- Malformed dates: {summary.get('malformed_dates')}
- Missing calendar dates (including weekends if unpublished): {summary.get('missing_calendar_dates')}

## Weekend treatment

Empirical counts on the published aligned calendar (before weekday filtering):

- Saturdays: {summary.get('n_saturdays')}
- Sundays: {summary.get('n_sundays')}
- Weekday gaps (business-day calendar vs published weekdays): {summary.get('n_weekday_gaps')}
- Zero USD log returns on the full published calendar: {summary.get('n_zero_usd_returns_full_calendar')}
- Zero EUR log returns on the full published calendar: {summary.get('n_zero_eur_returns_full_calendar')}

NBK Excel extracts for 2015 include Saturday and Sunday rows that repeat the previous official rate.
Estimation therefore uses `{config.ESTIMATION_CALENDAR}` observations (Monday–Friday as published).
Returns are computed between consecutive retained dates (Friday-to-Monday is one observation).
Volatility is **not** computed on repeated weekend zeros.

## Currency alignment

Intersection of dates with both USD/KZT and EUR/KZT is retained.
Dates missing one currency are listed in `missing_currency_alignment.csv`.
No forward-fill was applied before missingness diagnostics.

## Suspicious values

Extreme absolute log returns (|r| ≥ {config.EXTREME_JUMP_ABS_LOG_RETURN}) are flagged in
`flagged_extreme_jumps.csv` and listed in `extreme_returns.csv`. They are **not** deleted.

## Methodology window (6 January 2016 – 1 February 2016)

NBK states that official and market calculation methods differed in this window.
A flag `methodology_window_2016` is stored. Sensitivity statistics are in
`methodology_window_sensitivity.csv`. The window is **not** dropped from the baseline automatically.

## Structural-break notes

Assumed policy date in `config.FLOAT_TRANSITION_DATE` = {config.FLOAT_TRANSITION_DATE}
(NBK Annual Report 2015: free float from 20 August 2015).
Formal tests are in `structural_break_tests.csv`. Detected change-points need not coincide with the policy date.

## Unresolved issues

- Kazakhstan Open Data (`data.egov.kz`) requires an API key and was not used.
- The official RSS service (`get_rates.cfm`) did not return 2010–2021 dates in this environment; yearly Excel export was used instead.
- Rate checkbox ids are site-specific; if NBK changes ids, update `config.py`.
- 2025 coverage stops at {config.RAW_END} by design even if later dates exist on the website.

Processed estimation sample N (weekday log returns, non-missing):
USD {int(daily['usd_kzt_log_return'].notna().sum())}, dates {daily['date'].min()} to {daily['date'].max()}.
"""
    (config.DIAGNOSTICS_DIR / "data_quality_report.md").write_text(text, encoding="utf-8")


def write_method_recommendation(daily: pd.DataFrame) -> None:
    base = slice_window(daily, config.BASELINE_START, config.BASELINE_END)
    dist = pd.read_csv(config.TABLES / "distribution_tests.csv")
    acorr = pd.read_csv(config.TABLES / "autocorrelation_tests.csv")
    arch = pd.read_csv(config.TABLES / "arch_tests.csv")
    dep = pd.read_csv(config.TABLES / "return_correlations.csv")
    brk = pd.read_csv(config.TABLES / "structural_break_tests.csv")
    garch = pd.read_csv(config.TABLES / "garch11_diagnostics.csv")
    corr_roll = pd.read_csv(config.DATA_INTERIM / "rolling_correlations.csv")

    def pval(df, **filt):
        q = df
        for k, v in filt.items():
            q = q[q[k] == v]
        return float(q["ljung_box_pvalue"].iloc[0]) if len(q) else float("nan")

    def archp(series, lags=10):
        q = arch[(arch["series"] == series) & (arch["lags"] == lags)]
        return float(q["arch_lm_pvalue"].iloc[0]) if len(q) else float("nan")

    usd_jb = float(dist.loc[dist["series"] == "usd_kzt", "jarque_bera_pvalue"].iloc[0])
    eur_jb = float(dist.loc[dist["series"] == "eur_kzt", "jarque_bera_pvalue"].iloc[0])
    pearson_b = float(dep.loc[dep["window"] == "B_baseline", "pearson"].iloc[0])
    spearman_b = float(dep.loc[dep["window"] == "B_baseline", "spearman"].iloc[0])
    roll_std = float(corr_roll["corr_90d"].std(skipna=True))

    q1 = (
        "No. Baseline diagnostics show fat tails and (typically) volatility clustering; "
        "raw-return serial correlation is assessed separately below."
    )
    usd_raw = pval(acorr, series="usd_kzt", transform="raw", lag=10)
    eur_raw = pval(acorr, series="eur_kzt", transform="raw", lag=10)
    usd_abs = pval(acorr, series="usd_kzt", transform="abs", lag=10)
    usd_sq = pval(acorr, series="usd_kzt", transform="squared", lag=10)
    eur_abs = pval(acorr, series="eur_kzt", transform="abs", lag=10)
    eur_sq = pval(acorr, series="eur_kzt", transform="squared", lag=10)

    levene = brk[brk["test_family"] == "pre_post_moments_levene_welch"]
    levene_note = "; ".join(
        f"{r.series} Levene p={r.pvalue:.4g}" for r in levene.itertuples()
    )

    garch_note = garch.to_csv(index=False)

    text = f"""# Simulation Method Recommendation (Phase 1 evidence only)

This note does **not** implement a Monte Carlo allocator. It records what the
baseline sample ({config.BASELINE_START} to {config.BASELINE_END}, weekday official rates)
implies for later simulation design.

EUR/KZT is an NBK **cross rate**. Correlation with USD/KZT is a mechanical
dependence statistic for joint simulation, not evidence of two independent markets.

## Question 1 — Are daily returns approximately i.i.d.?

{q1}

## Question 2 — Autocorrelation in raw returns?

- USD/KZT Ljung–Box lag 10 p-value: {usd_raw:.4g} ({_sig(usd_raw)})
- EUR/KZT Ljung–Box lag 10 p-value: {eur_raw:.4g} ({_sig(eur_raw)})

If raw-return tests fail to reject white noise while |r| and r² do not, mean
returns may be weakly dependent even if volatility is not.

## Question 3 — Autocorrelation in absolute/squared returns?

- USD |r| LB10 p={usd_abs:.4g}; r² LB10 p={usd_sq:.4g}
- EUR |r| LB10 p={eur_abs:.4g}; r² LB10 p={eur_sq:.4g}

Persistent dependence in |r| or r² is the usual signature of volatility clustering
and is the main statistical case for **block** or **volatility-aware** resampling.

## Question 4 — ARCH effects?

- USD ARCH-LM (10 lags) p={archp('usd_kzt'):.4g} ({_sig(archp('usd_kzt'))})
- EUR ARCH-LM (10 lags) p={archp('eur_kzt'):.4g} ({_sig(archp('eur_kzt'))})

Diagnostic GARCH(1,1):

```
{garch_note}
```

## Question 5 — Approximate Gaussianity?

Jarque–Bera p-values (baseline): USD {usd_jb:.4g}; EUR {eur_jb:.4g}.
Visual histograms/Q-Q plots are supporting exhibits only. Large excess kurtosis
plus JB rejection would make a Gaussian parametric Monte Carlo a poor default.

## Question 6 — USD–EUR dependence?

Baseline Pearson {pearson_b:.4f}; Spearman {spearman_b:.4f}.
Standard deviation of 90-day rolling correlation: {roll_std:.4f}.
Rolling correlation is a descriptive stability check, not a structural model.

## Question 7 — August 2015 structural break?

Documented policy date: {config.FLOAT_TRANSITION_DATE}.
{levene_note}
Ruptures dates are reported in `structural_break_tests.csv` without forcing them
onto the policy date. A material pre/post variance change supports keeping the
**post-float window as the primary estimation sample** (Window B).

## Question 8 — Approaches that deserve Phase 2 evaluation

Based only on the tests above, Phase 2 should **compare** (not pre-commit to):

1. **i.i.d. historical bootstrap** — simple benchmark if raw returns are close to uncorrelated; may understate clustered tails if ARCH is strong.
2. **Moving-block / stationary bootstrap** — justified if |r| and r² dependence is statistically meaningful.
3. **Multivariate Gaussian parametric simulation** — usually a straw man if JB rejects and kurtosis is large.
4. **Student-t (or other fat-tailed) parametric joint model** — deserves comparison if tails are heavy but a parametric structure is still wanted.
5. **GARCH-based (possibly DCC/GO-GARCH) simulation** — deserves comparison if ARCH-LM rejects and α+β is high in the diagnostic GARCH(1,1).

No single winner is selected here. Window B remains the intended estimation sample;
Window C is a robustness check; the 2016 methodology window stays in-sample unless
Phase 2 sensitivity says otherwise.

Baseline weekday observations used: {int(base['usd_kzt_log_return'].notna().sum())}.
"""
    (config.DIAGNOSTICS_DIR / "simulation_method_recommendation.md").write_text(
        text, encoding="utf-8"
    )


def run_phase1(download: bool = True, overwrite_raw: bool = False) -> dict:
    ensure_directories()
    if download:
        acquire_nbk_yearly_excel(overwrite=overwrite_raw)
    build_raw_file_manifest()
    merged = load_and_merge_raw()
    summary = validate_panel(merged)
    # Full calendar copy for weekend documentation
    from .returns import add_flags, log_return

    calendar = add_flags(merged)
    calendar["usd_kzt_log_return"] = log_return(calendar["usd_kzt"])
    calendar["eur_kzt_log_return"] = log_return(calendar["eur_kzt"])
    calendar.to_csv(config.DATA_INTERIM / "fx_daily_full_calendar.csv", index=False)

    daily = build_daily_processed(merged)
    monthly = compound_monthly(daily)
    descriptive_statistics(daily)
    distribution_tests(daily)
    dependence_table(daily)
    corr_df = rolling_correlations(daily)
    autocorrelation_tests(daily)
    arch_tests(daily)
    garch_diagnostics(daily)
    combine_structural_tables(daily)
    methodology_window_sensitivity(daily)
    extreme_returns(daily)
    monthly_versus_daily(daily, monthly)
    save_all_figures(daily, corr_df)
    write_data_quality_report(summary, daily)
    write_method_recommendation(daily)
    return {"validation": summary, "n_daily": int(len(daily)), "n_monthly": int(len(monthly))}


if __name__ == "__main__":
    print(json.dumps(run_phase1(), indent=2, default=str))
