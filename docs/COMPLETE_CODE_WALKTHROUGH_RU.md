# Полный разбор кода: currency-allocation-research

**Документ:** `docs/COMPLETE_CODE_WALKTHROUGH_RU.md`  
**Режим:** только чтение исходников; исследовательский код **не** изменялся.  
**Язык:** русский; идентификаторы Python, имена файлов, библиотеки и уравнения — на английском.

Связанные файлы по фазам: `docs/code_walkthrough/00_architecture.md` … `07_end_to_end_trace.md`.

---

## 0. Как пользоваться этим гайдом

1. Сначала §1 (общая картина) и §2 (дерево).
2. Затем `src/config.py` (§3).
3. Потом фазы 1 → 2A → 2B1 → 2B2 → 2C.
4. В конце: end-to-end след, toy-пример, тесты, риски.

Каждое утверждение об имплементации опирается на реальный файл. Если план исследования и код расходятся — это явно помечено.

---

## 1. Общая картина (bird’s-eye view)

### 1.1. Что входит в систему

- **Эмпирические данные:** официальные курсы NBK USD/KZT и EUR/KZT (Excel по годам).
- **Модельные допущения:** профили домохозяйства P1–P4, отношение E/I, буфер B, горизонты, сетка весов, α для CVaR, DGP (bootstrap), costs/rebalance в Phase 2C.

### 1.2. Преобразования

```
RAW NBK Excel
  → Phase 1: очистка, weekday returns, диагностика
  → Phase 2A: joint FX Monte Carlo (bootstrap / Student-t / Gaussian)
  → Phase 2B1: household cash-flow engine + stress calibration
  → Phase 2B2: CVaR grid optimization по весам (KZT, USD, EUR)
  → Phase 2C: integer-bp grid patch + robustness
  → FINAL RESULTS (CSV/MD/figures under results/)
```

### 1.3. Главные промежуточные объекты

| Объект | Типичная форма | Где |
|--------|----------------|-----|
| Daily levels + returns | DataFrame ~тысячи строк | `data/processed/fx_daily_2010_2025.csv` |
| Joint history | `(T, 2)` ≈ (2705, 2) | `load_baseline_bundle()` |
| Simulated returns | `(n_paths, H, 2)` | симуляторы |
| Simulated levels | `(n_paths, H+1, 2)` | `returns_to_levels` |
| Monthly FX | `(n_paths, M, 2)` | `daily_paths_to_monthly_rates` |
| Path losses | `(n_paths,)` | `cumulative_shortfall` |
| Weight grid | bp integers sum 10000 | `generate_simplex_grid` |

### 1.4. Финальные выходы

- Phase 1: tables/figures/diagnostics (описательная статистика, recommendation).
- Phase 2A: `results/simulation/` (validation + model comparison).
- Phase 2B1: `results/household/` (stress_calibration, calibration_report).
- Phase 2B2: `results/optimization/` (optima, benchmarks, uncertainty).
- Phase 2C: `results/robustness/` (grid_patch + robustness tables/report).

### 1.5. Research questions ↔ код

| RQ | Содержание | Где в коде |
|----|------------|------------|
| RQ1 | Как аллоцировать сбережения | Phase 2B2 `optimize_all_cells`, CVaR argmin |
| RQ2 | Зависимость от валют дохода/расхода | профили P1–P4 + сравнение |
| RQ3 | Зависимость от горизонта | `OPT_HORIZON_YEARS` / loops |
| RQ4 | Лучше ли оптимум бенчмарков | `benchmark_comparison` |
| RQ5 | Чувствительность | Phase 2C `run_robustness_suite` |

### 1.6. Диаграмма entry points (реальные имена)

```
scripts/run_phase1.py
  → src.pipeline.run_phase1(...)
      → acquire_nbk_yearly_excel / build_raw_file_manifest
      → load_and_merge_raw / validate_panel
      → build_daily_processed / compound_monthly
      → diagnostics.* / structural_breaks / visualization
      → data_quality_report.md, simulation_method_recommendation.md

scripts/run_phase2a.py
  → src.simulation.model_comparison.run_phase2a()
      → load_baseline_bundle
      → build_models / fit_all / simulate_all
      → *_validation / score_models / write_report

scripts/run_phase2b1.py
  → src.household.calibration.run_phase2b1()
      → generate_common_fx_paths (SB L=20)
      → run_stress_calibration → simulate_household
      → select_baseline_candidate / write_calibration_report

scripts/run_phase2b2.py
  → src.optimization.pipeline.run_phase2b2()
      → load_or_generate_fx_levels
      → optimize_all_cells (grid → CVaR)
      → uncertainty / 50k confirmation / figures / report

scripts/run_phase2c.py
  → src.robustness.pipeline.run_phase2c()
      → run_grid_patch (integer bp)
      → verify_baseline_gate
      → run_robustness_suite (R1–R5)
      → figures / robustness_report.md
```

---

## 2. Дерево проекта (релевантные файлы)

### A. ENTRY POINTS / SCRIPTS
- `scripts/run_phase1.py` — запуск Phase 1.
- `scripts/run_phase2a.py` — запуск Phase 2A.
- `scripts/run_phase2b1.py` — запуск Phase 2B1.
- `scripts/run_phase2b2.py` — запуск Phase 2B2 (+ console summary).
- `scripts/run_phase2c.py` — запуск Phase 2C (+ console summary).

### B. CONFIGURATION
- `src/config.py` — **единственный** центральный конфиг всех фаз.

### C. DATA
- `data/raw/` — yearly NBK Excel (не перезаписываются без `overwrite`).
- `data/interim/` — merged/full calendar.
- `data/processed/fx_daily_2010_2025.csv` — weekday estimation sample.
- `data/processed/fx_monthly_2010_2025.csv` — календарная месячная агрегация (Phase 1; household использует **другую** 21-day логику).

### D. PHASE 1 SOURCE
- `src/pipeline.py` — оркестрация Phase 1.
- `src/data_acquisition.py` — скачивание NBK Excel / RSS helper.
- `src/data_cleaning.py` — парсинг/merge/validate.
- `src/returns.py` — log returns, flags, monthly compound.
- `src/diagnostics.py` — JB, ACF, ARCH, rolling corr, etc.
- `src/structural_breaks.py` — тесты на разрыв.
- `src/visualization.py` — фигуры Phase 1.
- `src/utils.py` — directories, sha256, write_csv, slice_window.

### E. PHASE 2A SOURCE
- `src/simulation/base.py` — `JointReturnSimulator`, `rng`.
- `src/simulation/iid_bootstrap.py` — IID joint bootstrap.
- `src/simulation/stationary_bootstrap.py` — SB + moving-block + Politis–White heuristic.
- `src/simulation/student_t.py` — bivariate Student-t + Gaussian.
- `src/simulation/path_builder.py` — returns → levels.
- `src/simulation/validation.py` — `load_baseline_bundle`, stats helpers.
- `src/simulation/model_comparison.py` — `run_phase2a` и все validation tables.

### F. PHASE 2B1 SOURCE
- `src/household/profiles.py` — P1–P4 + `strategy_catalog`.
- `src/household/cashflows.py` — nominal amounts + daily→monthly.
- `src/household/portfolio.py` — initial holdings / rebalance (+ costs).
- `src/household/settlement.py` — `simulate_household`.
- `src/household/risk_metrics.py` — shortfall, VaR, CVaR.
- `src/household/calibration.py` — `run_phase2b1`.

### G. PHASE 2B2 SOURCE
- `src/optimization/grid.py` — integer bp simplex (**каноническая** сетка).
- `src/optimization/evaluator.py` — evaluate weights/grid, FX load.
- `src/optimization/near_optimal.py` — near-opt + identification.
- `src/optimization/uncertainty.py` — paired bootstrap.
- `src/optimization/pipeline.py` — `run_phase2b2`.

### H. PHASE 2C SOURCE
- `src/robustness/grid_patch.py` — Part A integrity.
- `src/robustness/optimize.py` — cell optimizer.
- `src/robustness/scenarios.py` — DGP/window FX generation.
- `src/robustness/classify.py` — robustness labels + final A/B/C/D.
- `src/robustness/pipeline.py` — `run_phase2c`.

### I. TESTS
- `tests/test_phase1_utils.py`, `test_*bootstrap*`, `test_path_builder.py`, `test_student_t.py`,
  household tests, `test_optimization.py`, `test_phase2c.py` (всего **57** `test_*`).

### J. NOTEBOOKS
- `notebooks/01_…`–`09_…` — exploratory; **не** являются canonical entry points финального pipeline (канон = `scripts/run_phase*.py`).

### K. GENERATED RESULTS
- `results/`, `results/simulation/`, `results/household/`, `results/optimization/`, `results/robustness/`.

### L. UNUSED / LEGACY / EXPERIMENTAL
- `fetch_rss_date` в `data_acquisition.py` — helper; Phase 1 runner **не** вызывает его в `run_phase1`.
- `PRIMARY_BLOCK_LENGTH = 10` в config — исторический default; household/opt используют `HH_BLOCK_LENGTH = 20`.
- Старая float-дедупликация сетки **заменена** integer bp в `grid.py`; старые inflated counts остались только в **сохранённых** CSV Phase 2B2 (`n_surface=5280`), не в текущем генераторе сетки.
- Notebooks — экспериментальный/педагогический слой.

---

## 3. Конфигурация (`src/config.py`)

Ниже — research-relevant константы. Полный файл: строки 1–281.

### Даты / окна
| NAME | VALUE | MEANING | WHERE USED |
|------|-------|---------|------------|
| `RAW_START`/`RAW_END` | 2010-01-01 / 2025-12-31 | окно сырых данных | acquisition |
| `BASELINE_START` | 2015-08-20 | post-float estimation | `load_baseline_bundle`, diagnostics |
| `RECENT_START` | 2020-01-01 | Window C / Phase 2C R3 | robustness |
| `FLOAT_TRANSITION_DATE` | 2015-08-20 | официальный float | flags, structural breaks |
| `METHODOLOGY_WINDOW_*` | 2016-01-06…02-01 | NBK method change | flags, sensitivity |
| `OOS_TRAIN_*` / `OOS_EVAL_*` | …2019 / 2020… | pseudo-OOS | Phase 2A |
| `FLOAT_SHOCK_SENSITIVITY_*` | Aug 2015 | узкий shock exclude | Phase 2A shock table |

### Phase 1 stats
| NAME | VALUE | MEANING |
|------|-------|---------|
| `ANNUALIZATION_FACTOR` | 252 | weekday year |
| `ESTIMATION_CALENDAR` | `"weekdays"` | drop Sat/Sun before returns |
| `ACF_MAX_LAG`, `LJUNG_BOX_LAGS`, `ARCH_LM_LAGS` | 30; (5,10,20,30); (5,10,20) | diagnostics |
| `EXTREME_JUMP_ABS_LOG_RETURN` | 0.05 | flag only, not delete |
| `RANDOM_SEED` | 42 | Phase 1 RNG if needed |

### NBK
| NAME | VALUE | MEANING |
|------|-------|---------|
| `NBK_EXCEL_EXPORT` | URL | yearly Excel |
| `NBK_USD_RATE_ID` / `NBK_EUR_RATE_ID` | 5 / 6 | checkbox ids |
| `TARGET_CURRENCIES` | `("USD","EUR")` | |

### Phase 2A
| NAME | VALUE | MEANING |
|------|-------|---------|
| `SIM_RANDOM_SEED` | 42 | primary sim seed |
| `HORIZON_1Y/3Y/5Y_DAYS` | 252/756/1260 | weekday horizons |
| `N_PATHS_PRIMARY` | 10000 | main MC size |
| `N_PATHS_CONVERGENCE` | 1k…50k | convergence table |
| `STATIONARY_BLOCK_LENGTHS` | (5,10,20) | compared L |
| `PRIMARY_BLOCK_LENGTH` | **10** | default candidate label; **не** финальный household L |
| `MULTIHORIZON_WINDOWS` | (5,21,63,252) | multihorizon validation |

### Phase 2B1
| NAME | VALUE | MEANING |
|------|-------|---------|
| `HH_BLOCK_LENGTH` | **20** | accepted primary DGP |
| `HH_N_PATHS` | 10000 | |
| `HH_WEEKDAYS_PER_MONTH` | 21 | household month |
| `HH_I0` | 1.0 | normalized income |
| `HH_EXPENSE_INCOME_RATIOS` | (0.80,0.95,1.05) | stress grid |
| `HH_BUFFER_MONTHS` | (1,3,6,12) | |
| `HH_HORIZON_YEARS` | (1,3,5) | |
| `HH_CVAR_ALPHA` | 0.95 | |
| `HH_CONVERSION_COST` | 0.0 | default |
| `HH_REBALANCE_FREQUENCY` | `"monthly"` | default |
| `HH_ACCOUNTING_ATOL` | 1e-8 | |
| `HH_WEALTH_DEPLETE_TOL` | 1e-10 | zero holdings if shortfall |

### Phase 2B2
| NAME | VALUE | MEANING |
|------|-------|---------|
| `OPT_EXPENSE_INCOME_RATIO` | 0.95 | primary env |
| `OPT_BUFFER_MONTHS` | 1 | primary env W0=0.95 |
| `OPT_COARSE_STEP` / `OPT_REFINED_STEP` | 0.02 / 0.01 | → 200/100 bp |
| `OPT_NEAR_OPTIMAL_REL_TOL_*` | 0.01 / 0.05 | |
| `OPT_WELL_IDENTIFIED_MAX_FRAC` | 0.05 | ID class A |
| `OPT_NOT_IDENTIFIED_MIN_FRAC` | 0.50 | ID class C |
| `OPT_BOOTSTRAP_REPLICATES` / `SEED` | 1000 / 4242 | |
| `OPT_CONFIRM_N_PATHS` / `SEED` | 50000 / 424242 | |

### Phase 2C
| NAME | VALUE | MEANING |
|------|-------|---------|
| `ROB_COARSE_STEP_BP` / `REFINED` | 200 / 100 | |
| `ROB_ALT_EI` / `ROB_ALT_B` | 1.05 / 3 | R1 |
| `ROB_DGP_SPECS` | SB20/10/56 + IID + seeds | R2 |
| `ROB_CONVERSION_COSTS` | 0, 0.25%, 0.5%, 1% | R5 |
| `ROB_REBALANCE_FREQS` | monthly/quarterly/none | R4 |
| `ROB_GATE_P2_CVAR` / `P4` | ≈1.187 / ≈0.508 | reproduction gate |
| `TOTAL_BP` | 10000 | в `grid.py`, не в config |

**Важно:** Phase 2B2 **не** читает автоматически JSON кандидата из 2B1; среда зашита в `OPT_*`.

---

## 4. Phase 1 — пошаговый walkthrough

### Запуск
```bash
PYTHONPATH=. python scripts/run_phase1.py
```
`scripts/run_phase1.py` вызывает `run_phase1(download=True, overwrite_raw=False)`  
(`src/pipeline.py:234–265`).

### Шаги `run_phase1`
1. `ensure_directories()` — создать data/results dirs.
2. `acquire_nbk_yearly_excel(overwrite=False)` — `src/data_acquisition.py:80–94`: цикл годов, `download_year_excel`; существующие raw **не** перезаписываются.
3. `build_raw_file_manifest()` — SHA256 манифест.
4. `load_and_merge_raw()` — `data_cleaning.py:135–156`: parse каждого файла, concat, `drop_duplicates(date, keep="last")` → `fx_daily_merged_raw.csv`.
5. `validate_panel(merged)` — missing/weekend/alignment diagnostics.
6. Full-calendar copy + log returns (для документации weekends).
7. `build_daily_processed(merged)` — **weekdays only**, log returns → `fx_daily_2010_2025.csv`.
8. `compound_monthly(daily)` — календарные месяцы (≠ household 21-day).
9. Батарея diagnostics + structural breaks + figures.
10. `write_data_quality_report` + `write_method_recommendation`.

### `log_return` (`src/returns.py:29–31`)
```python
return np.log(price / price.shift(1))
```
На weekday-календаре Friday→Monday — **один** return (промежуточные weekend rows уже удалены).

### Мини-пример return
S_{t-1}=500, S_t=510 → r = ln(510/500) ≈ 0.0198.

### EUR/KZT
Это NBK **cross rate**; высокая корреляция с USD/KZT — ожидаема механически; код явно предупреждает об этом в recommendation.

---

## 5. Phase 2A — симуляция

### Запуск
`scripts/run_phase2a.py` → `run_phase2a()` (`model_comparison.py:773–823`).

### Поток
1. `load_baseline_bundle()` → history `(T,2)`, `s0` = last levels in window.
2. `build_models()` — IID, SB L∈{5,10,20}, moving-block L=10, Student-t, Gaussian.
3. Fit + simulate 10k × 252 days (primary validation horizon).
4. Validation tables + scores + figures + `model_comparison_report.md`.
5. `recommend_block_length` выбирает L среди {5,10,20} по ACF/vol fit.
6. Politis–White heuristic на `|r_USD|` пишется в JSON (диагностика; L≈56 не baseline).

**Фактически принятый household DGP:** SB **L=20** (`HH_BLOCK_LENGTH`), даже если `PRIMARY_BLOCK_LENGTH=10` или recommendation иной — решение зафиксировано в household config после Phase 2A review.

### IID (`iid_bootstrap.py:27–36`)
```python
idx = rng.integers(0, len(hist), size=(n_paths, horizon_days))
return hist[idx]
```
Каждый день независимо, но **строка** `(r_USD, r_EUR)` целиком → contemporaneous dependence сохраняется, serial — разрушается.

### Path builder (`path_builder.py:8–31`)
```python
levels[:, 0, :] = s0
levels[:, 1:, :] = s0 * np.exp(np.cumsum(r, axis=1))
```
Форма: returns `(N,H,2)` → levels `(N,H+1,2)`.

Toy: S0=500, r=(0.01, −0.02, 0.03) →  
S1=500·e^{0.01}, S2=500·e^{0.01−0.02}, S3=500·e^{0.01−0.02+0.03}.

---

## 6. Stationary bootstrap — детально

**Файл:** `src/simulation/stationary_bootstrap.py`  
**Ядро индексов:** `_index_paths` строки **35–64**.

### Параметры
- L = `expected_block_length`
- `p = 1/L` (`geometric_success_probability`, строки 29–32)

### Алгоритм (векторизованный по paths)
1. `u = rng.random((n_paths, horizon_days))`
2. `starts = rng.integers(0, n_obs, size=(n_paths, horizon_days))` — кандидаты новых стартов
3. `idx[:, 0] = starts[:, 0]`
4. Для t=1..H-1:
   - `cont = u[:, t] > p` — продолжить блок?
   - если да: `(idx[:, t-1] + 1) % n_obs` (circular wrap)
   - если нет: `starts[:, t]` (новый блок)

5. `return hist[idx]` — **joint** rows: USD и EUR всегда из одной исторической даты.

### Почему joint
Независимое сэмплирование USD и EUR разрушило бы эмпирическую зависимость (~Pearson≈0.89 в baseline) и cross-rate структуру.

### Toy (L=3 → p≈0.333), история индексов 0…9
Возможная последовательность: `4,5,6, 1,2, 9,0,1,…`  
На каждом шаге копируется пара `(r_USD, r_EUR)` с этим индексом.

### Seed
`JointReturnSimulator.rng` → `np.random.default_rng(seed)` (`base.py:35–37`). Seed 42 → воспроизводимые индексы при тех же данных.

### Moving-block
Тот же `_index_paths` с `moving_block_length=L`: фиксированные блоки длины L, circular.

---

## 7. Phase 2B1 — household engine

### Запуск
`run_phase2b1()` (`calibration.py:354–378`):
1. `generate_common_fx_paths` — SB20, 10k×1260, seed 42, cache `sb20_daily_levels.npy`
2. `run_stress_calibration` — полный factorial
3. figures, `select_baseline_candidate`, report

### Профили (`profiles.py`)
| Profile | Income | Expenses |
|---------|--------|----------|
| P1 | 100% KZT | 100% KZT |
| P2 | 100% USD | 100% KZT |
| P3 | 50/50 KZT/USD | 100% KZT |
| P4 | 50/50 USD/EUR | 80% KZT / 20% EUR |

### Нормализация (`cashflows.py:38–58`)
При I0=1, S_USD0=500, qI_usd=0.5:
`I_usd = 0.5 * 1 / 500 = 0.001` (единиц USD).  
KZT-эквивалент при S0: `0.001 * 500 = 0.5` = доля.  
Деление на S0 нужно, чтобы **номинал** в иностранной валюте давал нужную KZT-стоимость на t=0.

### Daily → monthly (`cashflows.py:70–87`)
`daily_levels`: `(N, days+1, 2)` включая t=0.  
Month m uses index `(m+1)*21`.  
756 days → 36 months → shape `(N, 36, 2)`.

### Settlement (`settlement.py:81–232`) — один месяц
1. FX `S_u, S_e`
2. `wealth_before = a_kzt + a_usd*S_u + a_eur*S_e`
3. Добавить income в native currencies
4. Заплатить native expenses `min(holding, expense)`
5. Gaps → G в KZT; W = leftover wealth
6. `take = min(G, W/(1+c))`, fee=`c*take`, liquidate `frac=(take+fee)/W` pro-rata
7. `shortfall = G - take`
8. Optionally rebalance (`monthly`/`quarterly`/`none`) with buy-volume cost

Conservation check:  
`wealth_before + inc - funded - liq_fee ≈ wealth_after`  
(ошибки ~1e-13…1e-15 — float noise).

### Risk metrics (`risk_metrics.py`)
- `cumulative_shortfall` = sum months → L_s
- `shortfall_probability` = P(L>0)
- `wealth_depletion_probability` = P(terminal ≤ tol)
- `empirical_var` = k-th largest loss
- `empirical_cvar` = mean of largest k losses  
  k = ceil((1-α)N); N=10000, α=0.95 → **k=500**

Toy losses `[0,0,1,4,10]`, α=0.8 → k=ceil(0.2*5)=1 → CVaR=10, VaR=10.  
При большой массе нулей VaR95 может быть 0, а CVaR95>0 (как у P2 3Y).

### Calibration grid
Loops: horizons × profiles × E/I × B × strategies.  
После dedup бенчмарков (~684 строки в CSV; точное число = len table).  
`select_baseline_candidate` (`calibration.py:199–248`): P1 all-KZT P(SF)≤5% и FX benchmark в 5–40%; предпочтение horizon=3, E/I≈0.95, **B≈6**.

**Расхождение:** Phase 2B2 использует `OPT_BUFFER_MONTHS=1` (accepted research choice), а не обязательно `candidate["buffer_months"]` из этой сортировки. Код 2B2 **не** импортирует выбор автоматически.

---

## 8. Phase 2B2 — оптимизация

### Запуск
`run_phase2b2()` (`optimization/pipeline.py`).

### Это НЕ machine learning
Путь: **grid → household sim → path losses → CVaR → argmin**. Нет нейросети, нет “обучения” весов градиентом.

### Integer bp grid (`grid.py:17–53`)
`TOTAL_BP=10000`.  
`(8000,2000,0)` → 80% KZT, 20% USD, 0% EUR.  
Число точек: `(n+1)(n+2)/2` где n=10000/step → 2%→1326, 1%→5151.

### Coarse + refine (`pipeline.optimize_all_cells`)
1. Evaluate 2% grid
2. `local_refined_grid_bp` ±500 bp around near-best coarse
3. `merge_result_frames` по integer keys
4. `pick_argmin` (min CVaR, tie-break max KZT)

### Старый баг float-ключей
Раньше coarse∪refined дедуплицировали по float `w_kzt` → дубликаты вроде 0.14 vs 0.14000000000000001 → `n_surface=5280` вместо 5151.  
**Сейчас** канон — integer bp; старые CSV Phase 2B2 сохранены для аудита; Phase 2C `grid_patch` пересчитал counts.

### Near-optimal / identification (`near_optimal.py`)
- 1%: CVaR ≤ 1.01×CVaR*
- 5%: ≤ 1.05×CVaR*
- Если CVaR*≈0: абсолютный допуск
- A: frac1 ≤ 0.05 и CVaR*>0  
- C: frac1≥0.50 или (CVaR≈0 и frac5≥0.50)  
- иначе B

### Paired bootstrap (`uncertainty.py:12–76`)
Одинаковые resampled path indices для всех портфелей → разности CVaR.  
CI excludes 0 ⇒ ranking clear **в смысле MC uncertainty**, не “истина о экономике”.

### 50k confirmation
Не полный grid; кандидаты + бенчмарки; `stable_flag` если CVaR близка или best=`opt_10k`.

---

## 9. Phase 2C — robustness

### Запуск
`run_phase2c()`:
1. `run_grid_patch` — сравнение old vs corrected; STOP если economic change
2. `verify_baseline_gate` — P2/P4 CVaR gate
3. `run_robustness_suite` R0–R5
4. figures + report + final A/B/C/D

### Dimensions
| Dim | Что меняется | Что фиксируется |
|-----|--------------|-----------------|
| R1 | E/I=1.05, B=3 | SB20, monthly, c=0 |
| R2 | DGP SB10/56/IID (+seeds) | household baseline |
| R3 | окно 2020–2025 returns | SB20, baseline HH |
| R4 | rebalance monthly/quarterly/none | c=0 |
| R5 | c∈{0,0.25%,0.5%,1%} + cross 0.5%×freq | SB20 baseline HH |

### Costs
Liquidation: `fee=c*take`, gross-up `W/(1+c)`.  
Rebalance: `fee=c * buy_volume` (one-way).  
Один charge через KZT-numeraire — без двойного USD→KZT→EUR.

### No-rebalance
Пропуск `rebalance_holdings_*`: новый income остаётся в валюте поступления; поверхность CVaR может стать плоской → UNIDENTIFIED при том же argmin 100% KZT.

### Final class B
`final_paper_classification`: если есть UNIDENTIFIED при остальных ROBUST_EXACT → **RESULT B**.

---

## 10. End-to-end след одной величины

```
NBK Excel cell USD/KZT
 → parse_nbk_file / load_and_merge_raw
 → build_daily_processed → usd_kzt_log_return
 → load_baseline_bundle → hist[t,0]
 → StationaryBootstrapSimulator.simulate → r_path[s,d,0]
 → returns_to_levels → S_path[s,d,0]
 → daily_paths_to_monthly_rates → S_month[s,m,0]
 → nominal_cashflows → I_usd = q*I0/S0
 → simulate_household: income I_usd, expenses E_kzt, shortfall
 → cumulative_shortfall → L[s]
 → empirical_cvar(L) → CVaR95(w)
 → generate_simplex_grid / evaluate_grid / pick_argmin → w*
 → classify_robustness / final_paper_classification → RESULT B
```

---

## 11. EDUCATIONAL TOY EXAMPLE — NOT EMPIRICAL RESULTS

Упрощение: 1 path, 2 валюты (KZT, USD), 3 месяца, логика как в `simulate_household` с c=0, monthly rebalance.

**Параметры:** I0=1, E0=0.95, W0=0.95, P2 (100% USD income, 100% KZT expense), S0=500,  
`I_usd=1/500=0.002`, `E_kzt=0.95`.

### Allocation A: 100% KZT
t=0: a_kzt=0.95, a_usd=0.

Month 1, S=500:  
income → a_usd=0.002; wealth mark = 0.95+0.002*500=1.95  
pay E_kzt=0.95 from KZT → a_kzt=0, a_usd=0.002, W=1.0  
gap=0 → shortfall=0  
rebalance to 100% KZT: a_kzt=1.0, a_usd=0.

Month 2, S=550: income adds 0.002 USD; pay 0.95 KZT; leftover foreign liquidated if needed; rebalance to all-KZT.

Month 3: аналогично. При достаточном буфере/surplus shortfall часто 0 на спокойных путях.

### Allocation B: 100% USD
t=0: a_usd=0.95/500=0.0019.  
Каждый месяц income USD + expenses KZT ⇒ **обязательная** конвертация (pro-rata liquidation) для оплаты KZT gap.  
Если USD/KZT падает, KZT-ценность буфера падает → выше риск shortfall.  
Именно этот механизм объясняет, почему CVaR all-USD > CVaR all-KZT в baseline P2.

---

## 12. Библиотеки в ЭТОМ проекте

| Library | Роль здесь |
|---------|------------|
| **pandas** | NBK tables, diagnostics CSV, calibration grids, reports |
| **numpy** | симуляции, settlement vectorization, CVaR sort |
| **scipy** / **statsmodels** | JB, Ljung–Box, ACF, и др. в diagnostics/validation |
| **arch** | ARCH-LM / GARCH diagnostics (Phase 1); не полный DCC engine |
| **matplotlib** | все figures |
| **requests** | NBK Excel download |

---

## 13. Производительность

- Settlement: цикл по месяцам, внутри векторизация по `n_paths`.
- Optimization: цикл по весам сетки (1326+), каждый полный household run.
- 10k × 36m × 1326 ≈ очень дорого → один FX cache, common paths, refine только локально.
- Phase 2C не делает полный декартов продукт всех факторов.

---

## 14. Randomness / reproducibility

| Seed | Где | Зачем |
|------|-----|-------|
| 42 | SIM/HH/OPT/ROB baseline | primary paths |
| 4242 | OPT bootstrap | paired CVaR CI |
| 424242 | 50k confirm | separate draw |
| 4210/4256/4240/4220 | ROB DGP/window | documented alternate DGPs |

API: `np.random.default_rng(seed)`.

Common random numbers: одна матрица FX для всех весов внутри DGP.

---

## 15. Данные vs допущения

| ITEM | TYPE | SOURCE |
|------|------|--------|
| USD/KZT, EUR/KZT levels | empirical | NBK |
| Float date 2015-08-20 | institutional | NBK docs / config |
| E/I=0.95, B=1 | modeling / design | config OPT_* |
| 21 days/month | modeling approx | HH_WEEKDAYS_PER_MONTH |
| L=20 | selected DGP | HH_BLOCK_LENGTH |
| 10000 paths | computational | N_PATHS_PRIMARY |
| CVaR95 | risk objective | HH_CVAR_ALPHA |
| P4 20% EUR expenses | synthetic scenario | profiles.py |
| Transaction costs | sensitivity assumption | ROB_CONVERSION_COSTS |
| Bootstrap scenarios | resampled history | not forecasts |

---

## 16. Что легко неправильно понять

1. MC **не** предсказывает будущий курс на дату.
2. Bootstrap **переиспользует** историческое поведение returns.
3. 100% KZT — **условно** при принятых допущениях.
4. CVaR ≠ максимизация wealth.
5. Foreign income уже есть FX exposure.
6. EUR/KZT — cross-rate.
7. Более низкая дневная vol ≠ автоматически ниже household CVaR.
8. Argmin ≠ всегда sharply identified optimum.
9. P1 CVaR=0 ⇒ поверхность неидентифицирована, не «все портфели оптимальны экономически».
10. Paired bootstrap CI — про MC uncertainty, не про всю реальность.

---

## 17. Read-only code review

| Issue | Class |
|-------|-------|
| `OPT_BUFFER_MONTHS=1` не автосинхронизирован с `select_baseline_candidate` (предпочитает B≈6) | **MODERATE** (документировать; не баг симуляции) |
| `PRIMARY_BLOCK_LENGTH=10` vs `HH_BLOCK_LENGTH=20` — путаница имён | **MINOR** |
| Старые Phase 2B2 CSV содержат inflated `n_surface` | **MINOR** (аудит; код исправлен) |
| `fetch_rss_date` не в main Phase 1 path | **NO ISSUE** (helper) |
| Полный Cartesian robustness намеренно не делается | **NO ISSUE** |
| Look-ahead: s0 = last historical level; симуляция вперёд от него — OK для DGP, не forecast | **NO ISSUE / VERIFIED** |
| Float conservation ~1e-13 | **NO ISSUE / VERIFIED** |
| Matplotlib/fontcache abort risk в headless | **MINOR** (Phase 2C ставит Agg/MPLCONFIGDIR) |
| Дублирование evaluate logic 2B2 vs robustness.optimize | **MINOR** |

Critical issues, блокирующие корректность baseline CVaR/optima: **не найдены** при этом обзоре.

---

## 18. Финальные сводки

### A. Программа в 20 шагах
1. Скачать/прочитать NBK Excel.  
2. Слить USD/EUR по датам.  
3. Убрать weekends для estimation.  
4. Посчитать log returns.  
5. Диагностировать tails/ARCH/dependence/breaks.  
6. Загрузить baseline joint returns.  
7. Сравнить IID/SB/Student-t/Gaussian.  
8. Принять SB L=20.  
9. Сгенерировать 10k×1260 joint paths.  
10. Преобразовать returns→levels.  
11. Сжать в monthly (21 days).  
12. Задать профили и nominal cashflows.  
13. Прогнать stress grid E/I×B×horizon.  
14. Зафиксировать primary env E/I=0.95, B=1.  
15. Оценить simplex весов по CVaR.  
16. Найти argmin + near-optimal region.  
17. Сравнить с бенчмарками + bootstrap.  
18. Подтвердить 50k на ключевых клетках.  
19. Integer-bp grid patch.  
20. Robustness R1–R5 → RESULT B.

### B. 10 важнейших функций
1. `run_phase1` — `src/pipeline.py`  
2. `build_daily_processed` — `src/returns.py`  
3. `StationaryBootstrapSimulator.simulate` — `src/simulation/stationary_bootstrap.py`  
4. `returns_to_levels` — `src/simulation/path_builder.py`  
5. `load_baseline_bundle` — `src/simulation/validation.py`  
6. `nominal_cashflows` / `daily_paths_to_monthly_rates` — `src/household/cashflows.py`  
7. `simulate_household` — `src/household/settlement.py`  
8. `empirical_cvar` — `src/household/risk_metrics.py`  
9. `generate_simplex_grid` / `optimize_all_cells` — `grid.py` / `pipeline.py`  
10. `run_phase2c` / `classify_robustness` — `robustness/pipeline.py` / `classify.py`

### C. 15 идей перед защитой
1. Joint bootstrap vs independent.  
2. L и p=1/L.  
3. Levels = S0·exp(cumsum r).  
4. 21 weekday = 1 month.  
5. Native pay then pro-rata liquidation.  
6. Loss = sum shortfalls, not −wealth.  
7. CVaR = mean of worst 500 paths.  
8. Mass-at-zero: VaR=0, CVaR>0.  
9. Grid search ≠ ML.  
10. Integer bp uniqueness.  
11. Near-optimal region vs argmin.  
12. Identification A/B/C.  
13. Expense- vs income-matching.  
14. Robustness ≠ exact weight equality.  
15. What is data vs assumption.

---

*Конец мастер-документа. Детали фаз — в `docs/code_walkthrough/`.*
