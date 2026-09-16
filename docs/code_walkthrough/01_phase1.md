# 01 — Phase 1: данные и диагностика

См. мастер-документ §4 и `src/pipeline.py`.

## Entry

`scripts/run_phase1.py` → `run_phase1(download=True, overwrite_raw=False)`  
**Файл:** `src/pipeline.py:234–265`

## Фактическая последовательность

1. **`acquire_nbk_yearly_excel`** (`src/data_acquisition.py:80–94`)  
   Для каждого года `RAW_START`…`RAW_END` вызывает `download_year_excel`.  
   Если файл уже есть — `skipped_existing` (raw не затирается).  
   Лог: `results/diagnostics/data_acquisition_log.csv`.

2. **`build_raw_file_manifest`** (`data_acquisition.py:105–119`) — SHA256.

3. **`load_and_merge_raw`** (`data_cleaning.py:135–156`)  
   - `parse_nbk_file` определяет колонки date/USD/EUR/quant  
   - `usd_kzt = usd_raw / usd_quant`  
   - concat всех годов, `drop_duplicates(subset=["date"], keep="last")`  
   - пишет `data/interim/fx_daily_merged_raw.csv`

4. **`validate_panel`** (`data_cleaning.py:159+`) — missing dates, weekends, alignment CSV.

5. Full-calendar copy с returns (для документации weekend zeros).

6. **`build_daily_processed`** (`returns.py:34–57`)  
   - `add_flags` (post_float, methodology_window, is_weekend)  
   - если `ESTIMATION_CALENDAR=="weekdays"`: drop Sat/Sun  
   - `log_return = np.log(P/P.shift(1))`  
   - пишет `data/processed/fx_daily_2010_2025.csv`

7. **`compound_monthly`** (`returns.py:60–73`) — *календарные* месяцы:  
   `exp(sum daily log returns)-1`.  
   **Это не household month.** Household использует 21 weekday (`cashflows.daily_paths_to_monthly_rates`).

8. Diagnostics (`src/diagnostics.py`): descriptive_statistics, distribution_tests (Jarque–Bera), dependence_table, rolling_correlations, autocorrelation_tests (Ljung–Box), arch_tests, garch_diagnostics, extreme_returns, monthly_versus_daily.

9. **`combine_structural_tables`** (`structural_breaks.py`) — Chow/ruptures/Levene-Welch вокруг float date.

10. **`save_all_figures`**, **`write_data_quality_report`**, **`write_method_recommendation`**.

## Важные выражения

```python
np.log(price / price.shift(1))
```
Один Friday→Monday return после удаления weekends.

```python
work.groupby("month").agg(usd_sum=("usd_kzt_log_return","sum"))
np.exp(usd_sum) - 1
```
Календарный monthly simple return из суммы log returns.

## Выходы Phase 1 (для статьи)

| Файл | Содержание |
|------|------------|
| `results/tables/descriptive_statistics.csv` | моменты returns |
| `results/tables/distribution_tests.csv` | JB и др. |
| `results/tables/arch_tests.csv` / `autocorrelation_tests.csv` | ARCH/LB |
| `results/tables/structural_break_tests.csv` | разрывы |
| `results/diagnostics/data_quality_report.md` | качество данных |
| `results/diagnostics/simulation_method_recommendation.md` | что сравнивать в Phase 2A |

## Неиспользуемое в runner

- `fetch_rss_date` (`data_acquisition.py:122+`) — есть в модуле, **не** вызывается из `run_phase1`.
