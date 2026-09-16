# Data lineage и файловая карта проекта

Read-only разбор по **фактическому коду** (без изменений и без новых прогонов).
Дата снимка: 2026-08-29.

---

## 1. Общая карта проекта

```
data/raw/nbk_official_rates_YYYY.xlsx          ← NBK Excel (скачивание или вручную)
    ↓  src/data_acquisition.py, src/data_cleaning.py
data/interim/fx_daily_merged_raw.csv
data/interim/fx_daily_full_calendar.csv
data/interim/rolling_correlations.csv
    ↓  src/returns.py, src/diagnostics.py, src/pipeline.py
data/processed/fx_daily_2010_2025.csv        ← ГЛАВНЫЙ processed FX (Phase 1)
data/processed/fx_monthly_2010_2025.csv
    ↓  src/simulation/validation.py::load_baseline_bundle()
    ↓  src/simulation/model_comparison.py::run_phase2a()
results/simulation/tables/*.csv              ← сравнение DGP (Phase 2A)
results/simulation/diagnostics/phase2a_run_summary.json  (recommended_L=20)
    ↓  [пути НЕ сохраняются на диск в Phase 2A; L=20 берётся из config HH_BLOCK_LENGTH]
    ↓  src/household/calibration.py::generate_common_fx_paths()
results/household/cache/sb20_daily_levels.npy  ← 10k × 1260 дней (Phase 2B1)
results/household/tables/stress_calibration.csv
    ↓  [Phase 2B2 читает cache + config; calibration.csv только для verify]
    ↓  src/optimization/pipeline.py::run_phase2b2()
results/optimization/tables/optimal_allocations.csv   ← LEGACY (n_surface=5280)
results/optimization/tables/coarse_grid_results.csv
    ↓  src/robustness/grid_patch.py + src/robustness/pipeline.py
results/robustness/grid_patch/corrected_optimal_allocations.csv  ← CANONICAL optima
results/robustness/tables/robustness_results.csv
    ↓  src/final/pipeline.py::run_final_cleanup()
results/final/tables/*.csv                     ← publication layer
results/final/audit/*.csv, *.md
results/final/canonical_specification.json
```

**Критический переход Phase 2A → 2B1:** Phase 2A **не записывает** 10 000 FX-путей. Phase 2B1 **генерирует заново** SB20 с тем же seed=42 и сохраняет в `.npy`. Phase 2B2/2C **переиспользуют** этот cache через `load_or_generate_fx_levels()`.

**Критический переход Phase 2B1 → 2B2:** baseline household (E/I=0.95, B=1) задаётся в **`src/config.py`**, а **не** читается из JSON Phase 2B1. Файл `stress_calibration.csv` используется только для **проверки воспроизводимости** benchmark-стратегий.

---

## 2. Phase 1 — FX diagnostics

### Команда запуска

```bash
PYTHONPATH=. python scripts/run_phase1.py
```

### Entry point

| Уровень | Функция / модуль |
|---------|------------------|
| Script | `scripts/run_phase1.py` → `run_phase1(download=True, overwrite_raw=False)` |
| Pipeline | `src/pipeline.py::run_phase1()` |

### Главные модули

| Модуль | Роль |
|--------|------|
| `src/data_acquisition.py` | Скачивание NBK Excel, manifest, log |
| `src/data_cleaning.py` | Merge raw, validate, diagnostics |
| `src/returns.py` | Weekday filter, log returns, monthly compound |
| `src/diagnostics.py` | Descriptive stats, ARCH, ACF, GARCH |
| `src/structural_breaks.py` | Chow, ruptures, pre/post |
| `src/visualization.py` | Phase 1 figures |
| `src/config.py` | Окна, даты, пути, NBK IDs |

### INPUT FILES

| Файл | Путь | Кто создаёт | Что внутри | Для чего |
|------|------|-------------|------------|----------|
| NBK Excel (по годам) | `data/raw/nbk_official_rates_YYYY.xlsx` | `acquire_nbk_yearly_excel()` или вручную | Официальные USD/KZT, EUR/KZT | Источник всех курсов |
| Config | `src/config.py` | репозиторий | RAW_START/END, BASELINE_START, FLOAT_TRANSITION_DATE, NBK rate IDs | Окна, правила |
| (нет входных CSV) | — | — | — | Phase 1 стартует с raw Excel |

**Config-параметры, влияющие на Phase 1:**

- `RAW_START`, `RAW_END` — окно загрузки
- `BASELINE_START` = 2015-08-20, `RECENT_START` = 2020-01-01
- `ESTIMATION_CALENDAR` = `"weekdays"` — суббота/воскресенье исключаются
- `FLOAT_TRANSITION_DATE`, `METHODOLOGY_WINDOW_*`
- `NBK_USD_RATE_ID=5`, `NBK_EUR_RATE_ID=6`

### PROCESSING (пошагово)

1. **Acquire:** `acquire_nbk_yearly_excel()` — скачивает Excel по годам (не перезаписывает существующие raw).
2. **Manifest:** `build_raw_file_manifest()` — SHA256 каждого raw-файла.
3. **Merge:** `load_and_merge_raw()` — читает все `data/raw/nbk_official_rates_*`, парсит даты/числа → `fx_daily_merged_raw.csv`.
4. **Validate:** `validate_panel()` — пропуски, дубликаты, выходные, extreme jumps.
5. **Full calendar:** log returns на полном календаре (включая weekend rows) → `fx_daily_full_calendar.csv`.
6. **Processed daily:** `build_daily_processed()` — weekday filter + log returns → **`data/processed/fx_daily_2010_2025.csv`**.
7. **Monthly:** `compound_monthly()` → `data/processed/fx_monthly_2010_2025.csv`.
8. **Diagnostics:** descriptive, distribution, ACF, ARCH, GARCH, structural breaks → `results/tables/`, `results/diagnostics/`.
9. **Figures:** `save_all_figures()` → `results/figures/`.
10. **Reports:** `data_quality_report.md`, `simulation_method_recommendation.md`.

### OUTPUT FILES

#### A. Core data outputs

| Файл | Путь | Содержимое | Используется дальше? | Кем |
|------|------|------------|----------------------|-----|
| **Processed daily FX** | `data/processed/fx_daily_2010_2025.csv` | date, usd_kzt, eur_kzt, log returns, flags | **ДА — главный вход Phase 2A+** | `load_baseline_bundle()` |
| Processed monthly | `data/processed/fx_monthly_2010_2025.csv` | monthly compounded returns | Нет в pipeline 2A+ | только Phase 1 report |
| Merged raw | `data/interim/fx_daily_merged_raw.csv` | aligned USD/EUR levels | Нет напрямую | audit/repro |
| Full calendar | `data/interim/fx_daily_full_calendar.csv` | levels + returns incl. weekends | Нет в 2A+ | diagnostics only |
| Rolling corr interim | `data/interim/rolling_correlations.csv` | rolling USD-EUR corr | Phase 1 report only | `write_method_recommendation()` |

#### B. Diagnostics (`results/diagnostics/`)

| Файл | Назначение | Читается дальше? |
|------|------------|-----------------|
| `raw_file_manifest.csv` | SHA256 raw files | Final cleanup (repro manifest) |
| `data_acquisition_log.csv` | log загрузок | Report only |
| `validation_summary.csv` | panel validation | Report only |
| `missing_dates.csv`, `missing_currency_alignment.csv` | пропуски | Report only |
| `weekday_gaps.csv`, `duplicate_dates.csv` | calendar QA | Report only |
| `flagged_extreme_jumps.csv` | \|r\| ≥ 0.05 | Report only |
| `raw_parse_errors.csv` | parse failures | Report only |
| `data_quality_report.md` | human report | Final docs reference |
| `simulation_method_recommendation.md` | Phase 1 → 2A rationale | Reference only |

#### C. Tables (`results/tables/`)

| Файл | Содержимое | Читается дальше? |
|------|------------|-----------------|
| **`descriptive_statistics.csv`** | mean, SD, vol по окнам A/B/C | **Final cleanup** → `table_fx_descriptive_statistics.csv` |
| **`distribution_tests.csv`** | JB, skew, kurtosis | **Final cleanup** (Jarque-Bera p) |
| `return_correlations.csv` | Pearson/Spearman | Phase 1 report; Final (indirect) |
| `autocorrelation_tests.csv`, `acf_values.csv` | LB tests | Report only |
| `arch_tests.csv`, `garch11_diagnostics.csv` | volatility clustering | Report only |
| `structural_break_*.csv` | Chow, ruptures, pre/post | Report only |
| `extreme_returns.csv`, `methodology_window_sensitivity.csv` | tails, 2016 window | Report only |
| `monthly_vs_daily_descriptives.csv` | comparison | Report only |
| `descriptive_statistics.tex` | LaTeX export | Publication helper |

#### D. Figures (`results/figures/`)

`fig01_usd_kzt_level` … `fig12_eur` — Phase 1 only; Final cleanup генерирует **свои** figures из processed CSV.

### Главный результат Phase 1

**`data/processed/fx_daily_2010_2025.csv`** — единственный processed-файл, который читают все последующие фазы через `load_baseline_bundle()`.

---

## 3. Phase 2A — FX simulation model comparison

### Команда

```bash
PYTHONPATH=. python scripts/run_phase2a.py
```

### Entry point

`src/simulation/model_comparison.py::run_phase2a()`

### INPUT

| Источник | Путь / функция | Детали |
|----------|----------------|--------|
| Processed FX | `data/processed/fx_daily_2010_2025.csv` | через `load_baseline_bundle()` |
| Historical returns | `bundle.returns` | фильтр `BASELINE_START`–`BASELINE_END`, weekday, dropna |
| S0 (starting levels) | `bundle.s0` | **последняя строка** baseline window: usd_kzt, eur_kzt |
| Baseline date filter | `config.BASELINE_START`, `config.BASELINE_END` | в `validation.py::load_baseline_bundle()` |
| OOS window | `OOS_TRAIN_*`, `OOS_EVAL_*` | pseudo-OOS таблица |
| Float shock | `FLOAT_SHOCK_SENSITIVITY_*` | sensitivity only |

**Phase 2A НЕ читает** CSV из `results/simulation/` или `results/tables/` — только processed FX + config.

### Модели (создаются in-memory)

| Ключ в коде | Тип | Block L |
|-------------|-----|---------|
| `iid_bootstrap` | IID bootstrap | — |
| `stationary_bootstrap_L5` | Stationary bootstrap | 5 |
| `stationary_bootstrap_L10` | Stationary bootstrap | 10 |
| `stationary_bootstrap_L20` | Stationary bootstrap | 20 |
| `moving_block_L10` | Moving block bootstrap | 10 |
| `student_t` | Bivariate Student-t | — |
| `gaussian` | Bivariate Gaussian | — |

### PROCESSING

1. `load_baseline_bundle()` → hist (N×2 returns), s0, dates.
2. `build_models()` + `fit_all()` — fit всех моделей на hist.
3. `simulate_all(n_paths=10000, horizon=252, seed=42)` — **1Y paths in memory**.
4. Validation tables: marginal, tail, dependence, temporal, vol clustering, multihorizon, pseudo-OOS, float shock.
5. `score_models()` → `model_scores.csv`; `recommend_block_length()` → L=20 among {5,10,20}.
6. Figures → `results/simulation/figures/`.
7. Reports → `model_comparison_report.md`, `phase2a_run_summary.json`.

### In-memory (не на диск)

- `sims` — dict model_name → (10000, 252, 2) returns — **не сохраняется**
- `sims_temporal` — первые 400 paths — **не сохраняется**
- Fitted model objects — **не сохраняются**

### OUTPUT FILES

| Файл | Путь | Используется дальше? | Canonical? |
|------|------|----------------------|------------|
| `model_scores.csv` | `results/simulation/tables/` | **Final cleanup** | YES (via final table) |
| `marginal_validation.csv` … `float_shock_sensitivity.csv` | `results/simulation/tables/` | Final cleanup (components) | YES |
| `student_t_parameters.csv` | tables | Report | DEBUG |
| `convergence_analysis.csv` | tables | Report | DEBUG |
| `phase2a_run_summary.json` | diagnostics | **Final repro manifest** | YES |
| `politis_white_block_length.json` | diagnostics | Report | DEBUG |
| `model_comparison_report.md` | diagnostics | Human read | REPORT |
| Figures figA–figI | `results/simulation/figures/` | Phase 2A paper | FIGURE (legacy phase) |

### Сохраняются ли 10 000 FX paths?

**НЕТ.** Phase 2A симулирует paths только для validation (horizon=252 days). Следующие фазы **генерируют заново**:

| Фаза | Функция | Seed | Block | Horizon | Cache |
|------|---------|------|-------|---------|-------|
| 2B1 | `generate_common_fx_paths()` | 42 | L=20 | 1260 days (5Y) | `results/household/cache/sb20_daily_levels.npy` |
| 2B2 | `load_or_generate_fx_levels()` | 42 | L=20 | 1260 | читает cache 2B1 если exists |
| 2C baseline | `baseline_sb20_levels()` | 42 | L=20 | 1260 | тот же cache |
| 2C alt DGP | `generate_fx_levels()` | 4210/4256/4240 | varies | 1260 | `results/robustness/cache/*.npy` |
| 2C recent | `generate_fx_levels()` | 4220 | L=20 | 1260 | `SB20_recent_2020_2025_daily_levels.npy` |

**Воспроизводимость:** одинаковые `bundle.returns` (из processed CSV) + `random_seed` + `HH_BLOCK_LENGTH=20` + `N_PATHS_PRIMARY=10000`.

### Главный результат Phase 2A

**Рекомендация L=20** (в `phase2a_run_summary.json`, поле `recommended_L`) и таблицы validation. **Не** persisted simulation paths.

---

## 4. Phase 2B1 — Household calibration

### Команда

```bash
PYTHONPATH=. python scripts/run_phase2b1.py
```

### Entry point

`src/household/calibration.py::run_phase2b1()`

### INPUT

| Источник | Откуда | Прямое чтение файла Phase 2A? |
|----------|--------|-------------------------------|
| Historical returns | `load_baseline_bundle()` ← processed CSV | **Нет** — только тот же processed FX |
| Block length L=20 | `config.HH_BLOCK_LENGTH` | **Нет** — config, не phase2a JSON |
| Seed | `config.HH_RANDOM_SEED` (=42) | config |
| Profiles P1–P4 | `src/household/profiles.py` | code |
| E/I grid | `config.HH_EXPENSE_INCOME_RATIOS` = (0.80, 0.95, 1.05) | config |
| B grid | `config.HH_BUFFER_MONTHS` = (1, 3, 6, 12) | config |
| Horizons | `config.HH_HORIZON_YEARS` = (1, 3, 5) | config |
| Strategies B1–B6 | `strategy_catalog(profile)` | code |

**Phase 2B1 не читает** `results/simulation/*` напрямую.

### PROCESSING — calibration loop

```
for horizon_years in {1,3,5}:
  monthly_fx = daily_paths_to_monthly_rates(levels[:, :days+1])
  for profile in P1..P4:
    for expense_income_ratio in {0.80, 0.95, 1.05}:
      for buffer_months in {1, 3, 6, 12}:
        for strategy in B1..B6 (deduped):
          simulate_household(monthly_fx, cashflows, weights, W0)
          → shortfall_probability, CVaR95, terminal wealth stats
```

684 конфигурации = 4 profiles × 3 E/I × 4 B × 3 horizons × ~6 strategies (minus dedup).

In-memory:
- `levels` (10000, 1261, 2) daily FX levels
- `monthly` (10000, n_months, 2) per horizon slice
- `table` DataFrame 684 rows

### OUTPUT FILES

| Файл | Путь | Содержимое | Читается Phase 2B2? |
|------|------|------------|---------------------|
| **`stress_calibration.csv`** | `results/household/tables/` | все 684 rows metrics | **ДА — только verify benchmarks** |
| **`sb20_daily_levels.npy`** | `results/household/cache/` | (10000, 1261, 2) daily levels | **ДА — Phase 2B2/2C cache** |
| `phase2b1_run_summary.json` | diagnostics | s0, candidate selection | Final cleanup audit; **НЕ** runtime input 2B2 |
| `calibration_report.md` | diagnostics | human report + candidate JSON | Report |
| `example_path_accounting.csv` | diagnostics | trace P4 example | DEBUG |
| Figures figA–figE | `results/household/figures/` | calibration plots | FIGURE |

### Где зафиксирован B=1?

- **Calibration rule** в `select_baseline_candidate()` выбирает cell E/I=0.95, **B=1**, horizon=3 (см. `phase2b1_run_summary.json`).
- **Optimization baseline** задаётся в **`config.OPT_BUFFER_MONTHS = 1`** — это то, что использует Phase 2B2.

### Phase 2B2 читает baseline из файла Phase 2B1 или из config?

**Из config.** Доказательство по коду:

```python
# src/optimization/pipeline.py — optimize_all_cells()
cf, E0, W0 = primary_cashflows(profile, s0)
# primary_cashflows использует config.OPT_EXPENSE_INCOME_RATIO, OPT_BUFFER_MONTHS

# src/optimization/evaluator.py — primary_cashflows()
E0 = config.OPT_EXPENSE_INCOME_RATIO * I0   # 0.95
W0 = config.OPT_BUFFER_MONTHS * E0            # 1 * 0.95
```

Phase 2B2 **читает файл** `stress_calibration.csv` только здесь:

```python
# src/optimization/pipeline.py::run_phase2b2()
p2b1 = pd.read_csv(config.HH_TABLES / "stress_calibration.csv")
repro = verify_phase2b1_benchmarks(daily_levels, s0, p2b1)  # sanity check, raise if fail
```

`phase2b1_run_summary.json` с `"candidate": {"buffer_months": 1, ...}` — **документация**, не runtime config loader.

### Главный результат Phase 2B1

1. **`results/household/cache/sb20_daily_levels.npy`** — shared FX paths для 2B2/2C.
2. **`stress_calibration.csv`** — полная calibration surface + verify target для 2B2.

---

## 5. Phase 2B2 — CVaR optimization

### Команда

```bash
PYTHONPATH=. python scripts/run_phase2b2.py
```

### Entry point

`src/optimization/pipeline.py::run_phase2b2(run_full_1pct_check=True)`

### INPUT

| Источник | Файл / config | Тип |
|----------|---------------|-----|
| FX paths | `load_or_generate_fx_levels()` → cache `.npy` или regenerate | **file or regenerate** |
| S0 | из cache row 0 или `bundle.s0` | derived |
| Profiles | `src/household/profiles.py` | code |
| E/I, B, I0 | `config.OPT_EXPENSE_INCOME_RATIO=0.95`, `OPT_BUFFER_MONTHS=1` | **config** |
| Horizons | `config.OPT_HORIZON_YEARS` | config |
| Grid | integer bp: coarse 200bp, refined 100bp | code (`grid.py`) |
| CVaR alpha | `config.OPT_CVAR_ALPHA=0.95` | config |
| Rebalance | `config.HH_REBALANCE_FREQUENCY="monthly"` | config |
| Cost | `config.HH_CONVERSION_COST=0.0` | config |
| Verify table | `stress_calibration.csv` | **file (check only)** |

### PROCESSING

1. Load FX cache / generate SB20 seed 42.
2. `verify_phase2b1_benchmarks()` — сверка с calibration table.
3. For each profile × horizon:
   - Coarse grid 1326 portfolios (2% step)
   - Local refined grid (~100bp neighborhood)
   - `merge_result_frames()` — integer bp dedup
   - argmin CVaR, near-optimal region, identification class
4. Benchmark strategies B1–B6, paired bootstrap uncertainty.
5. 50k confirmation (seed 424242, P2/P4 3Y).
6. Optional full 1% simplex check P2/P4 3Y.

### OUTPUT FILES

| Файл | Путь | Phase 2C? | Canonical? |
|------|------|-----------|------------|
| **`optimal_allocations.csv`** | `results/optimization/tables/` | read for **comparison only** | **LEGACY** (n_surface inflated 5280) |
| `coarse_grid_results.csv` | tables | figures in Final | LEGACY (surface data OK) |
| `refined_grid_results.csv` | tables | no | LEGACY |
| `benchmark_comparison.csv` | tables | **Final cleanup** (benchmark table) | YES (metrics OK) |
| `horizon_allocation_comparison.csv` | tables | duplicate of optimal | LEGACY |
| `optimization_uncertainty.csv` | tables | no | REPORT |
| `confirmation_50k.csv` | tables | no | REPORT |
| `global_1pct_check_P2_P4_3Y.csv` | tables | no | DEBUG |
| `phase2b1_reproduction_check.csv` | diagnostics | no | DEBUG |
| `phase2b2_run_summary.json` | diagnostics | repro | DEBUG |
| `optimization_report.md` | diagnostics | human | REPORT |
| Figures fig01–fig07 | `results/optimization/figures/` | no | FIGURE (legacy) |

**Исправленный integer-bp output** создаётся в Phase 2C:

| Файл | Canonical? |
|------|------------|
| `results/robustness/grid_patch/corrected_optimal_allocations.csv` | **YES** |
| `results/robustness/grid_patch/grid_patch_comparison.csv` | audit |
| `results/robustness/grid_patch/corrected_near_optimal_regions.csv` | YES |

### Главный результат Phase 2B2

`optimal_allocations.csv` + `benchmark_comparison.csv` — но **canonical optima** для публикации = **corrected** из Phase 2C grid patch.

---

## 6. Phase 2C — Grid patch + robustness

### Команда

```bash
PYTHONPATH=. python scripts/run_phase2c.py
```

### Entry point

`src/robustness/pipeline.py::run_phase2c()`

### Что читает из Phase 2B2

| Файл | Как используется |
|------|------------------|
| `optimal_allocations.csv` | **grid_patch** — сравнение old vs corrected (не перезаписывает) |
| FX cache | `baseline_sb20_levels()` → `household/cache/sb20_daily_levels.npy` |

**Не читает** coarse/refined CSV из 2B2 для optimization — **пересчитывает** через `optimize_cell()`.

### Robustness cases

#### R0 — Baseline reference
- **INPUT:** baseline SB20 cache, E/I=0.95, B=1, monthly, c=0
- **CHANGED:** nothing (reference)
- **OUTPUT:** rows in `robustness_results.csv`, spec `baseline_SB20_EI95_B1_monthly_c0`

#### R1 — Household stress
- **INPUT:** same FX cache
- **CHANGED:** E/I=1.05, B=3 (`config.ROB_ALT_EI`, `ROB_ALT_B`)
- **OUTPUT:** spec `EI105_B3`, `EI105_B3_P1_control`

#### R2 — DGP alternatives
- **INPUT:** regenerate from processed FX with alt seeds/blocks
- **CHANGED:** SB10 (seed 4210), SB56 (seed 4256), IID (seed 4240)
- **OUTPUT:** `dgp_sensitivity.csv`; caches `results/robustness/cache/{SB10,SB56,IID}_daily_levels.npy`

#### R3 — Recent window
- **INPUT:** `load_baseline_bundle(RECENT_START, RECENT_END)`
- **CHANGED:** estimation 2020-01-01 – 2025-12-31, seed 4220
- **OUTPUT:** `window_sensitivity.csv`; cache `SB20_recent_2020_2025_daily_levels.npy`

#### R4 — Rebalancing
- **INPUT:** baseline FX
- **CHANGED:** `rebalance_frequency` ∈ {quarterly, none}
- **OUTPUT:** `rebalancing_sensitivity.csv`

#### R5 — Transaction costs
- **INPUT:** baseline FX
- **CHANGED:** `conversion_cost` ∈ {0.0025, 0.005, 0.01}; cross with rebalance freqs
- **OUTPUT:** `transaction_cost_sensitivity.csv`

### OUTPUT FILES (Phase 2C)

| Файл | Путь | Canonical? |
|------|------|------------|
| **`corrected_optimal_allocations.csv`** | `results/robustness/grid_patch/` | **YES** |
| `grid_patch_comparison.csv` | grid_patch | audit |
| **`robustness_results.csv`** | `results/robustness/tables/` | **YES** |
| `robustness_summary.csv` | tables | YES |
| `expense_vs_income_matching.csv` | tables | YES |
| `dgp/window/rebalancing/transaction_cost_sensitivity.csv` | tables | YES (slices) |
| `confirmation_50k_robustness.csv` | tables | REPORT |
| `baseline_reproduction_gate.csv` | diagnostics | audit |
| `phase2c_run_summary.json` | diagnostics | YES (classification) |
| `robustness_report.md`, `grid_patch_report.md` | diagnostics | REPORT |
| Caches `*.npy` | `results/robustness/cache/` | REGENERATABLE |
| Figures figR1–figR6 | `results/robustness/figures/` | FIGURE (legacy phase) |

### Главный результат Phase 2C

1. **Corrected optima** (grid patch).
2. **`robustness_results.csv`** + classification **RESULT B**.

---

## 7. Final Cleanup / Publication Freeze

### Команда

```bash
PYTHONPATH=. .venv/bin/python scripts/run_final_cleanup.py
```

### Entry point

`src/final/pipeline.py::run_final_cleanup()`

**Не перезапускает** Phase 1–2C. Читает существующие outputs + lightweight recompute для audits.

### Canonical final files → источники

| Final file | Источник данных | Phase | Исходный файл / функция |
|------------|-----------------|-------|-------------------------|
| `table_fx_descriptive_statistics.csv` | Phase 1 descriptives | 1 | `results/tables/descriptive_statistics.csv` + `distribution_tests.csv` |
| `table_simulation_model_comparison.csv` | Phase 2A scores | 2A | `results/simulation/tables/model_scores.csv` + validation tables |
| `table_optimal_allocations.csv` | Corrected optima | 2C | `results/robustness/grid_patch/corrected_optimal_allocations.csv` |
| `table_benchmark_3y.csv` | Benchmarks | 2B2 | `results/optimization/tables/benchmark_comparison.csv` |
| `table_robustness_summary.csv` | Robustness | 2C | `results/robustness/tables/robustness_results.csv` |
| `canonical_specification.json` | All configs + summaries | 1–2C | `src/config.py` + `phase2a/2b1/2c_run_summary.json` |
| `FINAL_RESULTS_SUMMARY.md` | synthesized | all | final tables + gate values |
| `PAPER_SAFE_CLAIMS.md` | rules | all | manual classification |
| `ASSUMPTIONS_AND_LIMITATIONS.md` | rules | all | manual |
| `PUBLICATION_ARTIFACT_MANIFEST.md` | mapping | all | manual |
| `audit/grid_integrity.csv` | code assert | — | `src/optimization/grid.py` |
| `audit/grid_reconciliation.csv` | old vs corrected | 2B2+2C | `optimal_allocations.csv` vs `grid_patch_comparison.csv` |
| `audit/reproducibility_manifest.csv` | hashes/metrics | 1–2C | existing diagnostics |
| `audit/*.md` | lightweight recompute | — | `src/final/pipeline.py` audits |
| `figures/figure*.pdf/png` | processed FX + phase tables | 1–2C | `build_publication_figures()` |

---

## 8. File dependency graph (текстовый)

```
data/raw/nbk_official_rates_YYYY.xlsx
    ↓  data_acquisition.acquire_nbk_yearly_excel()
    ↓  data_cleaning.load_and_merge_raw()
data/interim/fx_daily_merged_raw.csv
    ↓  returns.build_daily_processed()
data/processed/fx_daily_2010_2025.csv  ←──────────────┐
    ↓  validation.load_baseline_bundle()               │
    ├─→ Phase 2A (in-memory sims, tables only)         │
    ├─→ Phase 2B1 generate_common_fx_paths()           │
    │       ↓                                          │
    │   results/household/cache/sb20_daily_levels.npy  │
    │       ↓                                          │
    ├─→ Phase 2B2 load_or_generate_fx_levels()         │
    │       ↓                                          │
    │   results/optimization/tables/*.csv (LEGACY)     │
    │       ↓                                          │
    └─→ Phase 2C grid_patch + robustness               │
            ↓  (+ robustness/cache/*.npy alt DGPs)     │
        results/robustness/grid_patch/corrected_*.csv  │
        results/robustness/tables/robustness_*.csv     │
            ↓                                          │
        results/final/*  (CANONICAL publication)  ←────┘
              (figures also read processed CSV directly)
```

**Изолированные ветки (не feed forward):**

- `results/tables/arch_tests.csv`, `garch11_diagnostics.csv` → только Phase 1 reports
- `results/simulation/figures/*` → Phase 2A only
- `results/household/figures/*` → Phase 2B1 only
- `results/optimization/figures/*` → Phase 2B2 only (Final делает новые figures)

---

## 9. Таблица всех важных файлов

| Phase | File | Type | Created by | Read by | Purpose | Canonical? |
|-------|------|------|------------|---------|---------|------------|
| — | `data/raw/nbk_official_rates_*.xlsx` | RAW | NBK / acquire | Phase 1 | Source rates | RAW |
| 1 | `data/processed/fx_daily_2010_2025.csv` | OUTPUT | `returns.build_daily_processed` | 2A,2B1,2C,Final | Core FX panel | YES |
| 1 | `data/processed/fx_monthly_2010_2025.csv` | OUTPUT | `returns.compound_monthly` | — | Monthly stats | DEBUG |
| 1 | `data/interim/fx_daily_merged_raw.csv` | INTERMEDIATE | `data_cleaning` | — | Audit | DEBUG |
| 1 | `results/tables/descriptive_statistics.csv` | OUTPUT | `diagnostics.descriptive_statistics` | Final | FX descriptives | YES |
| 1 | `results/tables/distribution_tests.csv` | OUTPUT | `diagnostics.distribution_tests` | Final | JB tests | YES |
| 1 | `results/diagnostics/raw_file_manifest.csv` | OUTPUT | `data_acquisition` | Final audit | SHA256 | DEBUG |
| 2A | `results/simulation/tables/model_scores.csv` | OUTPUT | `model_comparison.score_models` | Final | DGP ranking | YES |
| 2A | `results/simulation/diagnostics/phase2a_run_summary.json` | OUTPUT | `run_phase2a` | Final | L=20, nu | YES |
| 2A | (10k sim paths) | INTERMEDIATE | `simulate_all` | — | **not saved** | — |
| 2B1 | `results/household/cache/sb20_daily_levels.npy` | OUTPUT | `generate_common_fx_paths` | 2B2,2C | FX paths cache | YES |
| 2B1 | `results/household/tables/stress_calibration.csv` | OUTPUT | `run_stress_calibration` | 2B2 verify | 684 configs | YES |
| 2B1 | `results/household/diagnostics/phase2b1_run_summary.json` | OUTPUT | `run_phase2b1` | Final audit | candidate doc | REPORT |
| 2B2 | `results/optimization/tables/optimal_allocations.csv` | OUTPUT | `optimize_all_cells` | 2C patch, Final reconcile | optima | **LEGACY** |
| 2B2 | `results/optimization/tables/benchmark_comparison.csv` | OUTPUT | `optimize_all_cells` | Final | benchmarks | YES |
| 2B2 | `results/optimization/tables/coarse_grid_results.csv` | OUTPUT | `optimize_all_cells` | Final figures | CVaR surface | LEGACY |
| 2C | `results/robustness/grid_patch/corrected_optimal_allocations.csv` | OUTPUT | `grid_patch.run_grid_patch` | **Final** | canonical optima | **YES** |
| 2C | `results/robustness/tables/robustness_results.csv` | OUTPUT | `run_robustness_suite` | Final | robustness | YES |
| 2C | `results/robustness/cache/SB10_daily_levels.npy` | OUTPUT | `generate_fx_levels` | 2C only | alt DGP cache | REGENERATABLE |
| Final | `results/final/tables/table_optimal_allocations.csv` | OUTPUT | `run_final_cleanup` | paper | publication | **YES** |
| Final | `results/final/canonical_specification.json` | OUTPUT | `run_final_cleanup` | paper | frozen spec | **YES** |
| — | `src/config.py` | CONFIG | repo | all phases | parameters | YES |

---

## 10. Что можно удалить, а что нельзя

### 1. MUST KEEP (никогда не удалять для воспроизводимости)

- `data/raw/nbk_official_rates_*.xlsx` — **RAW, never delete**
- `data/processed/fx_daily_2010_2025.csv`
- `src/` — весь код
- `results/robustness/grid_patch/corrected_optimal_allocations.csv`
- `results/final/` — publication layer

### 2. KEEP FOR REPRODUCIBILITY

- `results/household/cache/sb20_daily_levels.npy` — ускоряет exact reproduction 2B2/2C
- `results/household/tables/stress_calibration.csv`
- `results/optimization/tables/*` (включая LEGACY)
- `results/robustness/tables/*`
- `results/simulation/tables/*`
- `results/tables/*`, `results/diagnostics/*`
- Phase run summaries (`phase2a/2b1/2b2/2c_run_summary.json`)

### 3. LEGACY / superseded (не цитировать в статье, но хранить)

- `results/optimization/tables/optimal_allocations.csv` — **n_surface=5280** (float-key duplicates); optima/CVaR совпадают с corrected
- `results/optimization/tables/horizon_allocation_comparison.csv` — дубликат optimal
- `config.PRIMARY_BLOCK_LENGTH=10` — legacy name; фактически HH_BLOCK_LENGTH=20

### 4. REGENERATABLE (можно пересоздать командой фазы)

- `results/household/cache/sb20_daily_levels.npy` — regenerate Phase 2B1
- `results/robustness/cache/*.npy` — regenerate Phase 2C
- `results/final/*` — regenerate `run_final_cleanup.py`
- Все figures (`results/figures/`, `results/*/figures/`)
- `data/interim/*`

### 5. RAW — never delete

- `data/raw/*`

---

## DOCUMENTATION VS ACTUAL CODE

| Документация / ожидание | Фактический код |
|-------------------------|-----------------|
| «Phase 2B2 использует baseline, выбранный Phase 2B1» | **2B2 берёт E/I и B из `config.OPT_*`**, не парсит `phase2b1_run_summary.json`. Calibration candidate — документальный. |
| «select_baseline_candidate предпочитает B≈6» | Tie-break `abs(B-6)` **вторичен** после E/I; при E/I=0.95 выбирается **B=1** (см. calibration_report caveat). |
| «Phase 2A сохраняет primary simulation paths» | **Paths не пишутся на диск**; только metrics tables. |
| «lowest_overall_score = primary DGP» | `phase2a_run_summary.json`: lowest = `moving_block_L10`, но **recommended_L=20** по temporal/vol criteria. |
| `PRIMARY_BLOCK_LENGTH=10` в config | Household/opt используют **`HH_BLOCK_LENGTH=20`**. |
| `optimal_allocations.csv` n_surface | **5280 inflated** vs corrected **5151/1473**; economic optima unchanged (grid patch). |
| Phase 2C «читает» 2B2 optimization results | **Пересчитывает** optima via `optimize_cell()`; читает old file **только для сравнения** в grid_patch. |

---

## Краткие итоги по фазам

| Phase | Main input | Main output |
|-------|------------|-------------|
| **1** | `data/raw/nbk_official_rates_*.xlsx` | `data/processed/fx_daily_2010_2025.csv` |
| **2A** | processed CSV → baseline returns | `phase2a_run_summary.json`, `model_scores.csv` (no path file) |
| **2B1** | processed CSV + config L=20 seed 42 | `sb20_daily_levels.npy`, `stress_calibration.csv` |
| **2B2** | `.npy` cache + config E/I/B | `optimal_allocations.csv` (**LEGACY counts**), `benchmark_comparison.csv` |
| **2C** | `.npy` + processed CSV (alt windows) | `corrected_optimal_allocations.csv`, `robustness_results.csv` |
| **Final** | corrected + phase tables | `results/final/` |

**Canonical publication layer:** `results/final/`

**Critical inconsistencies found:** Phase 2B2 не загружает baseline household из JSON Phase 2B1 (только config); Phase 2A не persist paths; `optimal_allocations.csv` содержит legacy `n_surface` — использовать `corrected_optimal_allocations.csv` или `results/final/tables/table_optimal_allocations.csv`.
